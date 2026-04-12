"""
Сравнение: без плеча vs 2x плечо — с реальными комиссиями IBKR и процентами по марже.

Три сценария:
  A — Базовый:           4 слота, $2 плоско   (как в оригинальном бэктесте)
  B — Реалистичный:      4 слота, IBKR $0.005/акцию, без маржи
  C — 2x плечо:          8 слотов, IBKR $0.005/акцию + исторические проценты по марже
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# ─── Настройки ────────────────────────────────────────────────────────────────
CSV_PATH      = 'd:/projects/trading/signals_type_b_2010.csv'
START_BALANCE = 10_000.0
RISK_PCT      = 0.01
TEST_START    = '2010-01-01'
TEST_END      = '2026-04-11'

# Исторические ставки IBKR для счётов до $100k:
# Ставка ФРС (среднегодовая) + 1.5% IBKR спред
IBKR_RATES = {
    2010: 0.0175,  # 0.25% FFR + 1.5%
    2011: 0.0160,  # 0.10% FFR + 1.5%
    2012: 0.0164,  # 0.14% FFR + 1.5%
    2013: 0.0161,  # 0.11% FFR + 1.5%
    2014: 0.0159,  # 0.09% FFR + 1.5%
    2015: 0.0163,  # 0.13% FFR + 1.5%
    2016: 0.0190,  # 0.40% FFR + 1.5%
    2017: 0.0250,  # 1.00% FFR + 1.5%
    2018: 0.0341,  # 1.91% FFR + 1.5%
    2019: 0.0366,  # 2.16% FFR + 1.5%
    2020: 0.0159,  # 0.09% FFR + 1.5%
    2021: 0.0157,  # 0.07% FFR + 1.5%
    2022: 0.0318,  # 1.68% FFR + 1.5%
    2023: 0.0652,  # 5.02% FFR + 1.5%  ← дорого!
    2024: 0.0680,  # 5.30% FFR + 1.5%  ← дорого!
    2025: 0.0600,  # ~4.5% FFR + 1.5%  (оценка)
    2026: 0.0550,  # ~4.0% FFR + 1.5%  (оценка)
}


# ─── Функция комиссии IBKR ────────────────────────────────────────────────────
def ibkr_commission(pos_usd, entry_price):
    """
    IBKR Pro tiered: $0.005/акцию, min $1 за ордер, max 1% от суммы.
    Сделка = вход + выход = 2 ордера.
    """
    shares = pos_usd / entry_price
    comm_per_leg = max(1.0, shares * 0.005)
    comm_per_leg = min(comm_per_leg, pos_usd * 0.01)
    return comm_per_leg * 2  # оба ордера


# ─── Основная функция симуляции ───────────────────────────────────────────────
def simulate(trades_df, max_positions=4, use_leverage=False, flat_commission=False):
    """
    Прогоняет симуляцию на отсортированных сделках.

    flat_commission=True  → $2 фиксированно (сценарий A)
    flat_commission=False → IBKR per-share модель
    use_leverage=True     → занимаем 50% каждой позиции, платим маржинальный процент
    """
    df = trades_df.sort_values('date').copy()
    account  = START_BALANCE
    open_pos = []   # список дат закрытия открытых позиций
    rows = []
    total_commission = 0.0
    total_margin     = 0.0

    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)

        # Освобождаем слоты, которые уже закрылись
        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= max_positions:
            continue
        open_pos.append(x_date)

        # Размер позиции
        pos = (account * RISK_PCT) / (t['risk_pct'] / 100)

        # Комиссия
        if flat_commission:
            commission = 2.0
        else:
            commission = ibkr_commission(pos, t['entry'])

        # Маржинальные проценты
        if use_leverage:
            year = int(pd.to_datetime(t['date']).year)
            rate = IBKR_RATES.get(year, 0.065)
            borrowed = pos * 0.5                          # берём в долг 50% позиции
            margin_cost = borrowed * rate * (int(t['days']) / 365)
        else:
            margin_cost = 0.0

        pnl_usd  = pos * (t['pnl_pct'] / 100) - commission - margin_cost
        account += pnl_usd
        total_commission += commission
        total_margin     += margin_cost

        rows.append({
            'date':        t['date'],
            'year':        int(pd.to_datetime(t['date']).year),
            'pnl':         pnl_usd,
            'account':     account,
            'symbol':      t['symbol'],
            'win':         t['win'],
            'commission':  commission,
            'margin_cost': margin_cost,
        })

    result = pd.DataFrame(rows)
    return result, account, total_commission, total_margin


# ─── Отчёт по годам ───────────────────────────────────────────────────────────
def yearly_snapshot(sim_df):
    """Возвращает конечный баланс счёта на конец каждого года."""
    snap = {}
    for year, grp in sim_df.groupby('year'):
        snap[year] = grp['account'].iloc[-1]
    return snap


# ─── Главная функция ──────────────────────────────────────────────────────────
def main():
    print("=== Сравнение: без плеча vs 2x плечо (2010–2026) ===\n")

    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    print(f"Загружено сигналов: {len(df)}\n")

    years_total = (pd.Timestamp(TEST_END) - pd.Timestamp(TEST_START)).days / 365.25

    # ── Прогон трёх сценариев ────────────────────────────────────────────────
    sim_a, final_a, comm_a, marg_a = simulate(df, max_positions=4,
                                               use_leverage=False, flat_commission=True)
    sim_b, final_b, comm_b, marg_b = simulate(df, max_positions=4,
                                               use_leverage=False, flat_commission=False)
    sim_c, final_c, comm_c, marg_c = simulate(df, max_positions=8,
                                               use_leverage=True,  flat_commission=False)

    cagr_a = (final_a / START_BALANCE) ** (1 / years_total) - 1
    cagr_b = (final_b / START_BALANCE) ** (1 / years_total) - 1
    cagr_c = (final_c / START_BALANCE) ** (1 / years_total) - 1

    snap_a = yearly_snapshot(sim_a)
    snap_b = yearly_snapshot(sim_b)
    snap_c = yearly_snapshot(sim_c)

    all_years = sorted(set(snap_a) | set(snap_b) | set(snap_c))

    # ── Таблица по годам ─────────────────────────────────────────────────────
    print(f"{'Год':<6} {'Сделок A':>8} {'Сделок C':>8}  "
          f"{'A: $2 плоско':>14}  {'B: без плеча':>14}  {'C: 2x плечо':>14}")
    print("─" * 70)

    prev_a = START_BALANCE
    prev_b = START_BALANCE
    prev_c = START_BALANCE
    trades_per_year_a = sim_a.groupby('year').size().to_dict()
    trades_per_year_c = sim_c.groupby('year').size().to_dict()

    for yr in all_years:
        bal_a = snap_a.get(yr, prev_a)
        bal_b = snap_b.get(yr, prev_b)
        bal_c = snap_c.get(yr, prev_c)
        tr_a  = trades_per_year_a.get(yr, 0)
        tr_c  = trades_per_year_c.get(yr, 0)

        pct_a = (bal_a / prev_a - 1) * 100
        pct_b = (bal_b / prev_b - 1) * 100
        pct_c = (bal_c / prev_c - 1) * 100

        print(f"{yr:<6} {tr_a:>8} {tr_c:>8}  "
              f"${bal_a:>10,.0f} ({pct_a:+.0f}%)  "
              f"${bal_b:>10,.0f} ({pct_b:+.0f}%)  "
              f"${bal_c:>10,.0f} ({pct_c:+.0f}%)")

        prev_a, prev_b, prev_c = bal_a, bal_b, bal_c

    # ── Итоговая сводка ──────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("ИТОГОВАЯ СВОДКА")
    print("═" * 70)

    rows_summary = [
        ("A — Базовый (4 слота, $2 плоско)",       len(sim_a), final_a, cagr_a, comm_a, marg_a),
        ("B — Реалистичный (4 слота, IBKR, без маржи)", len(sim_b), final_b, cagr_b, comm_b, marg_b),
        ("C — 2x плечо (8 слотов, IBKR + маржа)", len(sim_c), final_c, cagr_c, comm_c, marg_c),
    ]
    for label, trades, final, cagr, comm, marg in rows_summary:
        print(f"\n  {label}")
        print(f"    Сделок:          {trades}")
        print(f"    Итог:            ${final:>12,.0f}")
        print(f"    CAGR:            {cagr*100:.1f}%")
        print(f"    Комиссии всего:  ${comm:>10,.0f}")
        print(f"    Маржа всего:     ${marg:>10,.0f}")
        print(f"    Потери (comm+margin): ${comm+marg:>8,.0f}")

    print("\n" + "─" * 70)
    print("СРАВНЕНИЕ ПОТЕРЬ:")
    print(f"  Реальные комиссии vs $2 плоско (B - A):  ${comm_b - comm_a:+,.0f}")
    print(f"  Стоимость маржи 2x плечо (сценарий C):   ${marg_c:>8,.0f}")
    print(f"  Выигрыш от 8 слотов vs 4 слота:          ${final_c - final_b:+,.0f}")
    print(f"  CAGR разница (C vs B):                   {(cagr_c - cagr_b)*100:+.1f}%")

    print("\n" + "═" * 70)
    print(f"  Вывод: {'2x плечо ВЫГОДНЕЕ' if final_c > final_b else '2x плечо НЕ ВЫГОДНЕЕ'} "
          f"(с учётом всех реальных затрат)")
    print("═" * 70 + "\n")


if __name__ == '__main__':
    main()
