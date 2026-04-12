"""
Симуляция с ежегодными пополнениями счёта.

Сценарии:
  A — $4,000 без пополнений (чистый бэктест)
  B — $4k старт + $6k через 6 мес + €10k/год, 4 слота, без плеча
  C — то же что B, но 8 слотов + 2x плечо + маржинальный процент
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

CSV_PATH  = 'd:/projects/trading/signals_type_b_2010.csv'
RISK_PCT  = 0.01

# Исторические курсы EUR/USD (среднегодовые)
EURUSD = {
    2017: 1.13, 2018: 1.18, 2019: 1.12, 2020: 1.14,
    2021: 1.18, 2022: 1.05, 2023: 1.08, 2024: 1.08,
    2025: 1.05, 2026: 1.08,
}
EUR_ANNUAL = 10_000  # €10k в год

# Маржинальные ставки IBKR (FFR + 1.5%)
IBKR_RATES = {
    2016: 0.0190, 2017: 0.0250, 2018: 0.0341, 2019: 0.0366,
    2020: 0.0159, 2021: 0.0157, 2022: 0.0318, 2023: 0.0652,
    2024: 0.0680, 2025: 0.0600, 2026: 0.0550,
}

MID_2016 = pd.Timestamp('2016-07-01')


def ibkr_comm(pos, entry):
    shares = pos / entry
    leg = max(1.0, shares * 0.005)
    return min(leg, pos * 0.01) * 2


def simulate(trades_df, start_balance, contributions, max_positions=4, use_leverage=False):
    """
    contributions: dict {pd.Timestamp: amount} — пополнения по датам
    """
    df = trades_df.sort_values('date').copy()

    account      = start_balance
    open_pos     = []
    rows         = []
    pending      = dict(contributions)   # копия, будем удалять выданные взносы
    total_added  = start_balance
    total_comm   = 0.0
    total_margin = 0.0

    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)

        # Проверяем — нужно ли добавить взнос перед этой сделкой
        for contrib_date in sorted(pending.keys()):
            if e_date >= contrib_date:
                account     += pending[contrib_date]
                total_added += pending[contrib_date]
                del pending[contrib_date]
                break  # только один взнос за раз

        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= max_positions:
            continue
        open_pos.append(x_date)

        pos    = (account * RISK_PCT) / (t['risk_pct'] / 100)
        comm   = ibkr_comm(pos, t['entry'])
        year   = e_date.year
        margin = (pos * 0.5) * IBKR_RATES.get(year, 0.06) * (int(t['days']) / 365) if use_leverage else 0.0

        pnl     = pos * (t['pnl_pct'] / 100) - comm - margin
        account += pnl
        total_comm   += comm
        total_margin += margin

        rows.append({
            'year':    year,
            'account': account,
            'win':     t['win'],
            'pnl':     pnl,
        })

    return pd.DataFrame(rows), account, total_added, total_comm, total_margin


def main():
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    df = df[df['year'] >= 2016]

    # ── Строим словарь взносов ────────────────────────────────────────
    contribs_b = {MID_2016: 6_000}   # +$6k через 6 мес
    for yr in range(2017, 2027):
        contribs_b[pd.Timestamp(f'{yr}-01-01')] = EUR_ANNUAL * EURUSD.get(yr, 1.08)

    contribs_c = dict(contribs_b)    # то же самое для сценария C

    # ── Прогон ────────────────────────────────────────────────────────
    sim_a, fin_a, inv_a, comm_a, marg_a = simulate(df, 4_000, {}, max_positions=4)
    sim_b, fin_b, inv_b, comm_b, marg_b = simulate(df, 4_000, contribs_b, max_positions=4)
    sim_c, fin_c, inv_c, comm_c, marg_c = simulate(df, 4_000, contribs_c, max_positions=8, use_leverage=True)

    years_total = (pd.Timestamp('2026-04-11') - pd.Timestamp('2016-01-01')).days / 365.25
    cagr = lambda fin, inv: (fin / inv) ** (1 / years_total) - 1

    # ── Таблица по годам ─────────────────────────────────────────────
    print('=' * 78)
    print('  СИМУЛЯЦИЯ С ПОПОЛНЕНИЯМИ — 2016–2026  (старт $4,000)')
    print('=' * 78)

    header = f"{'Год':<5}  {'Взнос':>9}  {'A: без пополн.':>16}  {'B: 4 слота':>16}  {'C: 8 слотов+плечо':>18}"
    print(f"\n{header}")
    print('─' * 78)

    prev_a = prev_b = prev_c = 4_000.0
    snap_a = {yr: g['account'].iloc[-1] for yr, g in sim_a.groupby('year')}
    snap_b = {yr: g['account'].iloc[-1] for yr, g in sim_b.groupby('year')}
    snap_c = {yr: g['account'].iloc[-1] for yr, g in sim_c.groupby('year')}

    total_contrib_shown = 0.0

    for yr in range(2016, 2027):
        # Взнос в этом году (для отображения)
        if yr == 2016:
            contrib_display = 6_000
        elif yr <= 2026:
            contrib_display = EUR_ANNUAL * EURUSD.get(yr, 1.08)
        else:
            contrib_display = 0
        total_contrib_shown += contrib_display

        bal_a = snap_a.get(yr, prev_a)
        bal_b = snap_b.get(yr, prev_b)
        bal_c = snap_c.get(yr, prev_c)

        pct_a = (bal_a / prev_a - 1) * 100
        pct_b = (bal_b / prev_b - 1) * 100 if prev_b > 0 else 0
        pct_c = (bal_c / prev_c - 1) * 100 if prev_c > 0 else 0

        marker = ' ←' if pct_c < pct_b else ''

        print(f"  {yr}  +${contrib_display:>7,.0f}"
              f"  ${bal_a:>10,.0f} ({pct_a:>+4.0f}%)"
              f"  ${bal_b:>10,.0f} ({pct_b:>+4.0f}%)"
              f"  ${bal_c:>13,.0f} ({pct_c:>+4.0f}%){marker}")

        prev_a, prev_b, prev_c = bal_a, bal_b, bal_c

    # ── Итоговая сводка ───────────────────────────────────────────────
    total_invested = 4_000 + 6_000 + sum(EUR_ANNUAL * EURUSD.get(yr, 1.08) for yr in range(2017, 2027))

    print('\n' + '=' * 78)
    print('  ИТОГ')
    print('=' * 78)

    for label, fin, inv, comm, marg, slots in [
        ('A — без пополнений, 4 слота',      fin_a, inv_a, comm_a, marg_a, 4),
        ('B — с пополнениями, 4 слота',       fin_b, inv_b, comm_b, marg_b, 4),
        ('C — с пополнениями, 8 слотов+плечо',fin_c, inv_c, comm_c, marg_c, 8),
    ]:
        profit = fin - inv
        mult   = fin / inv
        print(f'\n  {label}')
        print(f'    Вложено своих:     ${inv:>10,.0f}')
        print(f'    Итог счёта:        ${fin:>10,.0f}')
        print(f'    Чистая прибыль:    ${profit:>10,.0f}')
        print(f'    Множитель:              ×{mult:.1f}')
        print(f'    Комиссии:          ${comm:>10,.0f}')
        if marg > 0:
            print(f'    Маржинальные %:    ${marg:>10,.0f}')

    print(f'\n  Всего вложено за 10 лет (A vs B/C): ${4_000:,.0f} vs ${total_invested:,.0f}')
    print(f'  Разница в вложениях:                ${total_invested - 4_000:,.0f}')
    print(f'  Разница в итоге (B − A):            ${fin_b - fin_a:+,.0f}')
    print(f'  Разница в итоге (C − A):            ${fin_c - fin_a:+,.0f}')
    print('=' * 78 + '\n')


def project_forward():
    """
    Прогноз вперёд: 2026–2036, старт $4k, +$6k через 6 мес, +€10k/год.
    Использует средние годовые доходности из бэктеста 2016–2026.
    """
    # Средние годовые доходности из бэктеста (грубо, без плохих лет)
    # Берём реальные данные из бэктеста по годам:
    annual_returns_4slots = {
        2016: 0.27, 2017: 1.37, 2018: 0.12, 2019: 0.24,
        2020: 0.23, 2021: 0.28, 2022: -0.03, 2023: 0.27,
        2024: 0.44, 2025: 0.33,
    }
    annual_returns_8slots = {
        2016: 0.43, 2017: 3.54, 2018: 0.18, 2019: 0.26,
        2020: 0.39, 2021: 0.60, 2022: -0.05, 2023: 0.37,
        2024: 0.79, 2025: 0.40,
    }

    # Для прогноза используем среднее и медиану из исторических данных
    import numpy as np
    r4 = list(annual_returns_4slots.values())
    r8 = list(annual_returns_8slots.values())

    avg4 = np.mean(r4)   # ~0.333
    avg8 = np.mean(r8)   # ~0.691 (медиана лучше из-за 2017)
    med4 = np.median(r4) # без выброса 2017
    med8 = np.median(r8)

    # Взносы
    EURUSD_FUTURE = 1.08
    contribs = {
        2026.5: 6_000,     # +$6k через 6 мес (октябрь 2026)
        2027: 10_000 * EURUSD_FUTURE,
        2028: 10_000 * EURUSD_FUTURE,
        2029: 10_000 * EURUSD_FUTURE,
        2030: 10_000 * EURUSD_FUTURE,
        2031: 10_000 * EURUSD_FUTURE,
        2032: 10_000 * EURUSD_FUTURE,
        2033: 10_000 * EURUSD_FUTURE,
        2034: 10_000 * EURUSD_FUTURE,
        2035: 10_000 * EURUSD_FUTURE,
        2036: 10_000 * EURUSD_FUTURE,
    }

    def project(start, rate, contribs_dict):
        acc = start
        total_in = start
        rows = []
        for yr_f in [2026, 2026.5, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036]:
            contrib = contribs_dict.get(yr_f, 0)
            acc      += contrib
            total_in += contrib
            if yr_f != int(yr_f):   # промежуточный взнос (не годовой)
                rows.append({'yr': yr_f, 'contrib': contrib, 'acc': acc, 'total_in': total_in, 'mid': True})
                continue
            if yr_f == 2026:        # первые 9 месяцев года
                acc = acc * (1 + rate * 0.75)
            else:
                acc = acc * (1 + rate)
            rows.append({'yr': yr_f, 'contrib': contrib, 'acc': acc, 'total_in': total_in, 'mid': False})
        return rows

    rows_b_med = project(4_000, med4, contribs)
    rows_b_avg = project(4_000, avg4, contribs)
    rows_c_med = project(4_000, med8, contribs)
    rows_c_avg = project(4_000, avg8, contribs)

    print()
    print('=' * 80)
    print(f'  ПРОГНОЗ 2026–2036  (старт $4k, +$6k через 6 мес, +€10k/год)')
    print(f'  Медиана год.доход: 4 слота {med4*100:.0f}% / 8 слотов {med8*100:.0f}%')
    print(f'  Среднее год.доход: 4 слота {avg4*100:.0f}% / 8 слотов {avg8*100:.0f}%')
    print('=' * 80)
    print(f"\n  {'Год':<7}  {'Взнос':>9}  {'B медиана':>12}  {'B среднее':>12}  {'C медиана':>12}  {'C среднее':>12}")
    print('  ' + '─' * 72)

    for i in range(len(rows_b_med)):
        rm = rows_b_med[i]
        ra = rows_b_avg[i]
        cm = rows_c_med[i]
        ca = rows_c_avg[i]
        if rm['mid']:
            yr_s = '  └─ +$6k'
            print(f"  {yr_s:<9}  +${rm['contrib']:>6,.0f}")
            continue
        yr_s = str(int(rm['yr']))
        contrib_s = f"+${rm['contrib']:>6,.0f}" if rm['contrib'] > 0 else '        '
        print(f"  {yr_s:<7}  {contrib_s}  "
              f"${rm['acc']:>10,.0f}  "
              f"${ra['acc']:>10,.0f}  "
              f"${cm['acc']:>10,.0f}  "
              f"${ca['acc']:>10,.0f}")

    total_in = rows_b_med[-1]['total_in']
    print(f"\n  Всего вложено своих: ${total_in:,.0f}")
    print(f"  {'':7}  {'':9}  {'4сл/медиана':>12}  {'4сл/среднее':>12}  {'8сл/медиана':>12}  {'8сл/среднее':>12}")
    print(f"  Итог 2036:           "
          f"  ${rows_b_med[-1]['acc']:>10,.0f}  "
          f"${rows_b_avg[-1]['acc']:>10,.0f}  "
          f"${rows_c_med[-1]['acc']:>10,.0f}  "
          f"${rows_c_avg[-1]['acc']:>10,.0f}")

    for label, rows in [('4 слота медиана', rows_b_med), ('4 слота среднее', rows_b_avg),
                        ('8 слотов медиана', rows_c_med), ('8 слотов среднее', rows_c_avg)]:
        fin = rows[-1]['acc']
        inv = rows[-1]['total_in']
        print(f"  Прибыль ({label}): ${fin - inv:>10,.0f}  (×{fin/inv:.1f} на вложенное)")

    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
    project_forward()
