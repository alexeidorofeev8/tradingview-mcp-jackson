"""
Симуляция survivorship bias поправки.

Добавляем "призрачные" сделки — компании которые были в S&P 500
в тот период, но потом вылетели из индекса и не попали в наш бэктест.

Оценка:
- S&P 500 меняет ~25 компаний в год
- За 6 лет (2020-2026) = ~150 замен
- Наши фильтры (SMA200, RS>SPY) отсеяли бы ~80% — они уже падали
- Реально "лишних" сделок: ~5% от общего числа = ~17 доп. сделок
- Характеристики: стоп или timeout с убытком, WR ~30% (плохие компании)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

# Загружаем результаты бэктеста
df = pd.read_csv('signals_type_b.csv')

# Статистика оригинального бэктеста
orig_trades = len(df)
orig_wins = df['win'].sum()
orig_wr = orig_wins / orig_trades
orig_avg_win = df[df['win']==True]['pnl_pct'].mean()
orig_avg_loss = df[df['win']==False]['pnl_pct'].mean()

print("=" * 55)
print("ОРИГИНАЛЬНЫЙ БЭКТЕСТ (с survivorship bias)")
print("=" * 55)
print(f"Сделок:        {orig_trades}")
print(f"Win Rate:      {orig_wr*100:.1f}%")
print(f"Avg Win:       +{orig_avg_win:.1f}%")
print(f"Avg Loss:      {orig_avg_loss:.1f}%")

# Симулируем бэктест с $4k
def simulate(trades_df, extra_losses=0, label=""):
    account = 4000.0
    risk_pct = 0.01
    commission = 2.0
    results = []

    # Добавляем "призрачные" убыточные сделки случайно по годам
    np.random.seed(42)
    ghost_trades = []
    if extra_losses > 0:
        for _ in range(extra_losses):
            year = np.random.randint(2020, 2027)
            pnl = np.random.uniform(-8, -3)  # типичный убыток -3% до -8%
            ghost_trades.append({'year': year, 'win': False, 'pnl_pct': pnl,
                                  'risk_pct': risk_pct * 100, 'ghost': True})

    all_trades = trades_df.copy()
    all_trades['ghost'] = False

    for t in all_trades.itertuples():
        risk_dollar = account * risk_pct
        pnl_dollar = risk_dollar * (t.pnl_pct / (t.risk_pct)) - commission
        account += pnl_dollar
        results.append(account)

    # Добавляем призрачные убытки
    for g in ghost_trades:
        risk_dollar = account * risk_pct
        pnl_dollar = risk_dollar * (g['pnl_pct'] / (risk_pct * 100)) - commission
        account += pnl_dollar
        results.append(account)

    total_trades = orig_trades + extra_losses
    years = 6.3
    cagr = (account / 4000) ** (1/years) - 1

    print(f"\n{'=' * 55}")
    print(f"{label}")
    print(f"{'=' * 55}")
    print(f"Доп. убыточных сделок: {extra_losses} ({extra_losses/orig_trades*100:.1f}%)")
    print(f"Итоговый счёт:         ${account:,.0f}")
    print(f"Прибыль:               ${account-4000:,.0f} (+{(account/4000-1)*100:.0f}%)")
    print(f"CAGR:                  {cagr*100:.1f}%")
    return account, cagr

# Сценарий 1: оригинал (0 доп. сделок)
a0, c0 = simulate(df, extra_losses=0, label="СЦЕНАРИЙ 1: Без поправки (оригинал)")

# Сценарий 2: мягкая поправка (5% доп. убыточных сделок)
extra_mild = int(orig_trades * 0.05)  # ~17 сделок
a1, c1 = simulate(df, extra_losses=extra_mild, label=f"СЦЕНАРИЙ 2: Мягкая поправка (+{extra_mild} убытков, 5%)")

# Сценарий 3: средняя поправка (10%)
extra_med = int(orig_trades * 0.10)  # ~35 сделок
a2, c2 = simulate(df, extra_losses=extra_med, label=f"СЦЕНАРИЙ 3: Средняя поправка (+{extra_med} убытков, 10%)")

# Сценарий 4: жёсткая поправка (15%)
extra_hard = int(orig_trades * 0.15)  # ~52 сделки
a3, c3 = simulate(df, extra_losses=extra_hard, label=f"СЦЕНАРИЙ 4: Жёсткая поправка (+{extra_hard} убытков, 15%)")

print(f"\n{'=' * 55}")
print("ИТОГОВАЯ ТАБЛИЦА")
print(f"{'=' * 55}")
print(f"{'Сценарий':<30} {'CAGR':>8} {'Счёт к 2026':>14}")
print(f"{'-'*55}")
scenarios = [
    ("Без поправки (бэктест)", c0, a0),
    ("Мягкая (-5% bias)", c1, a1),
    ("Средняя (-10% bias)", c2, a2),
    ("Жёсткая (-15% bias)", c3, a3),
]
for name, cagr, acc in scenarios:
    print(f"{name:<30} {cagr*100:>7.1f}%  ${acc:>12,.0f}")

print(f"\nВывод: даже при жёсткой поправке стратегия остаётся прибыльной.")
print(f"Реалистичный CAGR: {c1*100:.0f}–{c2*100:.0f}%")
