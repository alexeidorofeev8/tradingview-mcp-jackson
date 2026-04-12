import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_csv('signals_type_b.csv')
df_2026 = df[df['year'] == 2026].copy()
df_2026['date'] = pd.to_datetime(df_2026['date'])
df_2026['exit_date'] = df_2026.apply(lambda r: r['date'] + pd.Timedelta(days=int(r['days'])), axis=1)
df_2026 = df_2026.sort_values('date').reset_index(drop=True)

account = 10000.0
risk_pct = 0.01
max_positions = 4
trades = []
open_positions = []

for _, row in df_2026.iterrows():
    entry_date = row['date']
    still_open = []
    for pos in open_positions:
        if pos['exit_date'] <= entry_date:
            pnl = pos['risk_dollar'] * (pos['pnl_pct'] / pos['risk_pct_entry'])
            account += pnl
        else:
            still_open.append(pos)
    open_positions = still_open

    if len(open_positions) >= max_positions:
        continue

    risk_dollar = account * risk_pct
    entry_price = row['entry']
    sl_price = row['sl']
    risk_per_share = entry_price - sl_price
    if risk_per_share <= 0:
        continue
    shares = max(1, int(risk_dollar / risk_per_share))
    position_value = shares * entry_price
    pnl_dollar = shares * entry_price * row['pnl_pct'] / 100
    risk_pct_entry = (risk_per_share * shares / position_value) * 100

    trades.append({
        'date_in': entry_date.strftime('%d.%m'),
        'ticker': row['symbol'],
        'entry': round(entry_price, 2),
        'sl': round(sl_price, 2),
        'shares': shares,
        'pos': round(position_value),
        'risk': round(risk_dollar),
        'pnl_pct': round(row['pnl_pct'], 1),
        'pnl_usd': round(pnl_dollar),
        'result': 'WIN' if row['win'] else 'LOSS',
        'days': int(row['days']),
        'account_after': 0
    })

    open_positions.append({
        'exit_date': row['exit_date'],
        'pnl_pct': row['pnl_pct'],
        'risk_dollar': risk_dollar,
        'risk_pct_entry': risk_pct_entry,
        'symbol': row['symbol']
    })

account2 = 10000.0
for t in trades:
    account2 += t['pnl_usd']
    t['account_after'] = round(account2)

print('| # | Дата  | Тикер | Вход    | Стоп    | Акций | Позиция  | Риск | P&L%   | P&L$   | Итог | Счёт    |')
print('|---|-------|-------|---------|---------|-------|----------|------|--------|--------|------|---------|')
for i, t in enumerate(trades, 1):
    res = 'WIN ' if t['result']=='WIN' else 'LOSS'
    print(f"| {i:<2}| {t['date_in']} | {t['ticker']:<5} | ${t['entry']:<7} | ${t['sl']:<7} | {t['shares']:<5} | ${t['pos']:<7} | ${t['risk']:<3} | {t['pnl_pct']:>+5}% | {t['pnl_usd']:>+5}$ | {res} | ${t['account_after']:,} |")

wins = sum(1 for t in trades if t['result']=='WIN')
losses = len(trades) - wins
print()
print(f"Старт: $10,000  |  Итог: ${round(account2):,}  |  +{round((account2/10000-1)*100,1)}%")
print(f"Сделок: {len(trades)}  |  WIN: {wins}  |  LOSS: {losses}  |  WR: {round(wins/len(trades)*100)}%")
