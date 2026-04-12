import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_csv('signals_type_b.csv')
df_2025 = df[df['year'] == 2025].copy()
df_2025['date'] = pd.to_datetime(df_2025['date'])
df_2025['exit_date'] = df_2025.apply(lambda r: r['date'] + pd.Timedelta(days=int(r['days'])), axis=1)
df_2025 = df_2025.sort_values('date').reset_index(drop=True)

SLIPPAGE_PCT = 0.10
COMMISSION   = 2.0
MAX_POS      = 99
MAX_POS_PCT  = 0.20  # макс 20% счёта на позицию
START        = 4_000.0

account = START
open_pos = []
trades = []

for _, row in df_2025.iterrows():
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

    # Кап 20% счёта
    max_pos_value = account * MAX_POS_PCT
    if pos_value > max_pos_value:
        shares    = max(1, int(max_pos_value / entry))
        pos_value = shares * entry

    slip    = pos_value * SLIPPAGE_PCT / 100
    pnl_net = pos_value * row['pnl_pct'] / 100 - COMMISSION - slip
    account += pnl_net

    open_pos.append({'exit_date': row['exit_date']})
    trades.append({
        'Дата':    entry_date.strftime('%d.%m'),
        'Тикер':   row['symbol'],
        'Акций':   shares,
        'Позиция': round(pos_value),
        'P&L%':    round(row['pnl_pct'], 1),
        'P&L$':    round(pnl_net),
        'Итог':    'WIN' if row['win'] else 'LOSS',
        'Счёт':    round(account),
    })

print('| # | Дата  | Тикер | Акций | Позиция | P&L%   | P&L$  | Итог | Счёт   |')
print('|---|-------|-------|-------|---------|--------|-------|------|--------|')
for i, t in enumerate(trades, 1):
    print(f"| {i:<2}| {t['Дата']} | {t['Тикер']:<5} | {t['Акций']:<5} | ${t['Позиция']:<6} | {t['P&L%']:>+5}% | {t['P&L$']:>+5}$ | {t['Итог']:<4} | ${t['Счёт']:,} |")

wins   = sum(1 for t in trades if t['Итог'] == 'WIN')
losses = len(trades) - wins
print(f"\nСтарт: $4,000  |  Итог: ${round(account):,}  |  {(account/START-1)*100:+.1f}%")
print(f"Сделок: {len(trades)}  |  WIN: {wins}  LOSS: {losses}  |  WR: {wins/len(trades)*100:.0f}%")
