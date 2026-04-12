import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_csv('signals_type_b.csv')
df_2026 = df[df['year'] == 2026].copy()
df_2026['date'] = pd.to_datetime(df_2026['date'])
df_2026['exit_date'] = df_2026.apply(lambda r: r['date'] + pd.Timedelta(days=int(r['days'])), axis=1)
df_2026 = df_2026.sort_values('date').reset_index(drop=True)

SLIPPAGE_PCT = 0.10
COMMISSION   = 2.0
MAX_POS      = 99  # без ограничений
START        = 100_000.0

account = START
open_pos = []
trades = []

for _, row in df_2026.iterrows():
    entry_date = row['date']
    open_pos = [p for p in open_pos if p['exit_date'] > entry_date]

    if len(open_pos) >= MAX_POS:
        continue

    risk_dollar    = account * 0.01
    entry          = row['entry']
    sl             = row['sl']
    risk_per_share = entry - sl
    if risk_per_share <= 0:
        continue

    shares    = max(1, int(risk_dollar / risk_per_share))
    pos_value = shares * entry
    slip      = pos_value * SLIPPAGE_PCT / 100
    pnl_net   = pos_value * row['pnl_pct'] / 100 - COMMISSION - slip
    account  += pnl_net

    exit_date = row['exit_date']
    open_pos.append({'exit_date': exit_date})

    trades.append({
        'Дата входа':  entry_date.strftime('%d.%m'),
        'Дата выхода': exit_date.strftime('%d.%m'),
        'Дней':        int(row['days']),
        'Тикер':       row['symbol'],
        'Вход $':      round(entry, 2),
        'Стоп $':      round(sl, 2),
        'Акций':       shares,
        'Позиция $':   round(pos_value),
        'Риск $':      round(risk_dollar),
        'P&L %':       round(row['pnl_pct'], 1),
        'Slip+Com $':  round(slip + COMMISSION),
        'P&L чистый $': round(pnl_net),
        'Итог':        'WIN' if row['win'] else 'LOSS',
        'Счёт $':      round(account),
    })

df_out = pd.DataFrame(trades)
print(df_out.to_string(index=False))

wins   = sum(1 for t in trades if t['Итог'] == 'WIN')
losses = len(trades) - wins
total_pnl   = round(account - START)
total_slip  = sum(t['Slip+Com $'] for t in trades)

print(f"\n{'='*70}")
print(f"Старт:     ${START:,.0f}")
print(f"Итог:      ${account:,.0f}")
print(f"Прибыль:   ${total_pnl:+,.0f}  ({(account/START-1)*100:.1f}%)")
print(f"Сделок:    {len(trades)}  |  WIN: {wins}  LOSS: {losses}  |  WR: {wins/len(trades)*100:.0f}%")
print(f"Slippage+комиссии итого: ${total_slip:,.0f}")
