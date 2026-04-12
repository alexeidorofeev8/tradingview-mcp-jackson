"""
Сравнение MAX_POSITIONS = 4 vs 8 vs 12
на реальных сигналах 2016-2026
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

df = pd.read_csv('signals_type_b.csv')
df = df[df['year'] >= 2016].copy()
df['date'] = pd.to_datetime(df['date'])
df['exit_date'] = df.apply(lambda r: r['date'] + pd.Timedelta(days=int(r['days'])), axis=1)
df = df.sort_values('date').reset_index(drop=True)

def simulate(max_pos, label):
    account = 4000.0
    commission = 2.0
    open_pos = []  # list of exit_date
    trades_taken = 0
    trades_skipped = 0
    wins = 0

    for _, row in df.iterrows():
        entry_date = row['date']

        # Закрываем истёкшие позиции
        open_pos = [p for p in open_pos if p['exit_date'] > entry_date]

        # Нет места — пропускаем
        if len(open_pos) >= max_pos:
            trades_skipped += 1
            continue

        risk_dollar = account * 0.01
        entry = row['entry']
        sl = row['sl']
        risk_per_share = entry - sl
        if risk_per_share <= 0:
            continue

        shares = max(1, int(risk_dollar / risk_per_share))
        pnl_dollar = shares * entry * row['pnl_pct'] / 100 - commission
        account += pnl_dollar
        trades_taken += 1
        if row['win']:
            wins += 1

        open_pos.append({'exit_date': row['exit_date']})

    years = 10.3
    cagr = (account / 4000) ** (1/years) - 1
    wr = wins / trades_taken * 100 if trades_taken > 0 else 0

    print(f"\n{'─'*50}")
    print(f"MAX_POSITIONS = {max_pos}  |  {label}")
    print(f"{'─'*50}")
    print(f"Сделок взято:    {trades_taken}")
    print(f"Сделок скипнуто: {trades_skipped}")
    print(f"Win Rate:        {wr:.1f}%")
    print(f"Итоговый счёт:   ${account:,.0f}")
    print(f"CAGR:            {cagr*100:.1f}%")
    return trades_taken, trades_skipped, wr, account, cagr

print("СРАВНЕНИЕ MAX_POSITIONS | Бэктест 2016–2026 | Старт $4,000")

r4  = simulate(4,  "текущая стратегия")
r6  = simulate(6,  "умеренное расширение")
r8  = simulate(8,  "агрессивное расширение")
r12 = simulate(12, "без ограничений")

print(f"\n{'='*55}")
print(f"{'Позиций':<10} {'Сделок':<10} {'WR':>6} {'CAGR':>8} {'Счёт 2026':>14}")
print(f"{'─'*55}")
for lbl, r in [("4 (базовый)", r4), ("6", r6), ("8", r8), ("12 (без огр.)", r12)]:
    print(f"{lbl:<14} {r[0]:<10} {r[2]:>5.1f}%  {r[4]*100:>6.1f}%  ${r[3]:>12,.0f}")
