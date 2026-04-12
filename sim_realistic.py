"""
Реалистичная симуляция 2016-2026
- Старт: $10,000
- MAX_POSITIONS: 8
- Комиссия: $1 вход + $1 выход = $2/сделку
- Slippage: 0.05% вход + 0.05% выход = 0.10% round-trip
- Риск: 1% счёта на сделку
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_csv('signals_type_b.csv')
df = df[df['year'] >= 2016].copy()
df['date'] = pd.to_datetime(df['date'])
df['exit_date'] = df.apply(lambda r: r['date'] + pd.Timedelta(days=int(r['days'])), axis=1)
df = df.sort_values('date').reset_index(drop=True)

SLIPPAGE_PCT = 0.10   # 0.10% round-trip (0.05% вход + 0.05% выход)
COMMISSION   = 2.0    # $1 + $1
MAX_POS      = 8
START        = 10_000.0

account = START
open_pos = []
trades_taken = 0
wins = 0
total_commission = 0
total_slippage = 0

yearly = {}

for _, row in df.iterrows():
    entry_date = row['date']
    open_pos = [p for p in open_pos if p['exit_date'] > entry_date]

    if len(open_pos) >= MAX_POS:
        continue

    risk_dollar  = account * 0.01
    entry        = row['entry']
    sl           = row['sl']
    risk_per_share = entry - sl
    if risk_per_share <= 0:
        continue

    shares = max(1, int(risk_dollar / risk_per_share))
    pos_value = shares * entry

    # Slippage уменьшает P&L (покупаем дороже, продаём дешевле)
    slip_dollar = pos_value * SLIPPAGE_PCT / 100
    pnl_dollar  = pos_value * row['pnl_pct'] / 100 - COMMISSION - slip_dollar

    account += pnl_dollar
    trades_taken += 1
    total_commission += COMMISSION
    total_slippage += slip_dollar
    if row['win']:
        wins += 1

    year = entry_date.year
    if year not in yearly:
        yearly[year] = {'start': account - pnl_dollar, 'end': account}
    else:
        yearly[year]['end'] = account

    open_pos.append({'exit_date': row['exit_date']})

years = 10.3
cagr = (account / START) ** (1/years) - 1
wr = wins / trades_taken * 100

print("=" * 55)
print("РЕАЛИСТИЧНАЯ СИМУЛЯЦИЯ 2016–2026")
print("Старт $10,000 | 8 позиций | Комиссия + Slippage")
print("=" * 55)

print(f"\n{'Год':<6} {'Счёт к концу года':>20}")
print(f"{'─'*30}")
prev = START
for yr in sorted(yearly.keys()):
    end = yearly[yr]['end']
    chg = (end/prev - 1)*100
    print(f"{yr:<6} ${end:>14,.0f}   ({chg:>+.1f}%)")
    prev = end

print(f"\n{'─'*55}")
print(f"Сделок:          {trades_taken}")
print(f"Win Rate:        {wr:.1f}%")
print(f"Комиссии итого:  ${total_commission:,.0f}")
print(f"Slippage итого:  ${total_slippage:,.0f}")
print(f"{'─'*55}")
print(f"ИТОГОВЫЙ СЧЁТ:   ${account:,.0f}")
print(f"Прибыль:         ${account-START:,.0f} (+{(account/START-1)*100:.0f}x)")
print(f"CAGR:            {cagr*100:.1f}%")
print(f"{'─'*55}")
print(f"\nС поправкой на survivorship bias (-10%/год):")
adj_cagr = cagr - 0.10
adj_account = START * (1 + adj_cagr) ** years
print(f"Реалистичный CAGR: {adj_cagr*100:.1f}%")
print(f"Реалистичный счёт: ${adj_account:,.0f}")
