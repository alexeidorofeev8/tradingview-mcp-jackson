"""
Бэктест Type B — только 2026 год (январь–апрель)
Скачивает свежие данные, генерирует сигналы, сравнивает плечо vs без плеча.
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# ─── Даты ─────────────────────────────────────────────────────────────────────
DATA_START = '2022-01-01'   # 4 года истории для надёжного SMA200 и prior_high
TEST_START = '2026-01-01'
TEST_END   = '2026-04-12'

# ─── Параметры стратегии ──────────────────────────────────────────────────────
START_BALANCE = 10_000.0
RISK_PCT      = 0.01

SMA_FAST, SMA_SLOW       = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
SR_LOOKBACK_MIN          = 20
SR_LOOKBACK_MAX          = 80
MAX_ATR_PCT              = 5.0
MIN_PRICE                = 20
MIN_VOL                  = 500_000
EXCLUDE_SECTORS          = {'Real Estate', 'Utilities'}
EARNINGS_BUFFER          = 7
TOP_SECTORS              = 5
HOURLY_DISC              = 0.01

B_BREAKDOWN_BARS = 1
B_MAX_BREACH_PCT = 1.5
B_RECOVERY_BODY  = 0.5
B_RECOVERY_VOL   = 1.2
B_STOP_BUFFER    = 0.5

SECTOR_ETF = {
    'Information Technology': 'XLK', 'Financials': 'XLF',
    'Energy': 'XLE',                  'Health Care': 'XLV',
    'Industrials': 'XLI',             'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',        'Materials': 'XLB',
    'Communication Services': 'XLC',
}
EARNINGS_CACHE = 'd:/projects/trading/earnings_cache.json'

# Маржинальная ставка IBKR для 2026 (~4% ФРС + 1.5% IBKR)
MARGIN_RATE_2026 = 0.055


# ─── Вспомогательные функции ──────────────────────────────────────────────────
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_earnings():
    if os.path.exists(EARNINGS_CACHE):
        raw = json.load(open(EARNINGS_CACHE, encoding='utf-8'))
        return {s: set(v) for s, v in raw.items()}
    return {}


def build_sector_leaders(spx_close):
    etfs = list(SECTOR_ETF.values())
    raw = yf.download(etfs, start=DATA_START, progress=False,
                      auto_adjust=True, group_by='ticker')
    sec_prices = {}
    for sector, etf in SECTOR_ETF.items():
        try:
            sec_prices[sector] = raw[etf]['Close'].squeeze()
        except:
            pass

    idx = spx_close.index
    leaders = {}
    for i in range(RS_LEN, len(idx)):
        date = idx[i]
        spx_ret = spx_close.iloc[i] / spx_close.iloc[i - RS_LEN]
        scores = {}
        for sec, prices in sec_prices.items():
            p = prices.reindex(idx, method='ffill')
            p_now, p_ago = p.iloc[i], p.iloc[i - RS_LEN]
            if pd.isna(p_now) or pd.isna(p_ago) or p_ago == 0:
                continue
            scores[sec] = (p_now / p_ago) / spx_ret
        if len(scores) >= TOP_SECTORS:
            leaders[date] = set(sorted(scores, key=scores.get, reverse=True)[:TOP_SECTORS])
    return leaders


def run_symbol(sym, df_stock, spx_close, earnings, sec_leaders, stock_sec):
    try:
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)
        close  = df_stock['Close'].squeeze()
        high   = df_stock['High'].squeeze()
        low    = df_stock['Low'].squeeze()
        open_  = df_stock['Open'].squeeze()
        volume = df_stock['Volume'].squeeze()
        if len(close) < 220:
            return []

        sma50  = close.rolling(SMA_FAST).mean()
        sma200 = close.rolling(SMA_SLOW).mean()
        atr    = calc_atr(high, low, close, ATR_LEN)
        vol_ma = volume.rolling(VOL_LEN).mean()
        spx    = spx_close.reindex(close.index, method='ffill')
        rs     = (close / close.shift(RS_LEN)) / (spx / spx.shift(RS_LEN))

        prior_high  = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        local_min   = low.rolling(15).min().shift(1)
        recent_low  = low.shift(1).rolling(B_BREAKDOWN_BARS).min()

        false_break = (
            ((recent_low < prior_high) & (recent_low >= prior_high * (1 - B_MAX_BREACH_PCT / 100))) |
            ((recent_low < local_min)  & (recent_low >= local_min  * (1 - B_MAX_BREACH_PCT / 100)))
        )
        recovery = (
            ((close > prior_high) & (low.shift(1) < prior_high)) |
            ((close > local_min)  & (low.shift(1) < local_min))
        )
        strong_recovery = (
            (close > open_) &
            ((close - open_) / atr > B_RECOVERY_BODY) &
            (volume > vol_ma * B_RECOVERY_VOL)
        )
        trend_ok   = (close > sma200) & (sma50 > sma200)
        quality_ok = ((atr / close * 100) < MAX_ATR_PCT) & (rs > 1.0) & (vol_ma > MIN_VOL) & (close > MIN_PRICE)
        market_ok  = spx > spx.rolling(200).mean()

        sig_mask  = trend_ok & quality_ok & false_break & recovery & strong_recovery & market_ok
        date_mask = (close.index >= TEST_START) & (close.index <= TEST_END)

        results = []
        bars     = list(close.index)
        earn_set = earnings.get(sym, set())

        for sig_date in sig_mask[date_mask & sig_mask].index:
            sig_d = sig_date.date()
            if any(0 <= (pd.to_datetime(ed).date() - sig_d).days <= EARNINGS_BUFFER for ed in earn_set):
                continue

            idx_i = bars.index(sig_date)
            entry_price = close.iloc[idx_i] * (1 - HOURLY_DISC)

            lookback     = min(B_BREAKDOWN_BARS + 1, idx_i)
            recent_lows  = low.iloc[idx_i - lookback: idx_i]
            breakdown_low = recent_lows.min() if len(recent_lows) > 0 else low.iloc[idx_i]
            sl_init      = breakdown_low - atr.iloc[idx_i] * B_STOP_BUFFER

            risk     = entry_price - sl_init
            risk_pct = risk / entry_price * 100
            if risk <= 0 or risk_pct > 15:
                continue

            res_cands = [
                high.iloc[max(0, idx_i - lb):idx_i].max()
                for lb in [10, 20, 40, 60]
                if high.iloc[max(0, idx_i - lb):idx_i].max() > entry_price * 1.01
            ]
            if res_cands:
                rr = max(1.5, min(5.0, (min(res_cands) - entry_price) / risk))
                tp = entry_price + risk * rr
            else:
                tp = entry_price + risk * 3.0

            actual_rr = (tp - entry_price) / risk
            if actual_rr < 1.0:
                continue

            exit_p, reason, hold = None, 'timeout', 0
            for j in range(1, 21):
                if idx_i + j >= len(bars):
                    break
                bh = high.iloc[idx_i + j]
                bl = low.iloc[idx_i + j]
                bc = close.iloc[idx_i + j]
                bs = sma50.iloc[idx_i + j]
                ba = atr.iloc[idx_i + j]
                hold = j
                if bl <= sl_init:    exit_p = sl_init; reason = 'stop';   break
                if bh >= tp:         exit_p = tp;      reason = 'target'; break
                if bc < bs - ba*0.5: exit_p = bc;      reason = 'sma';    break
            if exit_p is None:
                exit_p = close.iloc[min(idx_i + hold, len(close) - 1)]

            results.append({
                'symbol':     sym,
                'sector':     stock_sec or '',
                'date':       sig_date,
                'entry':      entry_price,
                'exit':       exit_p,
                'sl':         sl_init,
                'tp':         tp,
                'rr':         actual_rr,
                'risk_pct':   risk_pct,
                'pnl_pct':    (exit_p - entry_price) / entry_price * 100,
                'win':        exit_p >= entry_price,
                'reason':     reason,
                'days':       hold,
                'year':       sig_date.year,
                'month':      sig_date.month,
                'top_sector': bool(
                    stock_sec and sig_date in sec_leaders and
                    stock_sec in sec_leaders[sig_date]
                ),
            })
        return results
    except Exception:
        return []


# ─── Модель комиссий IBKR ────────────────────────────────────────────────────
def ibkr_commission(pos_usd, entry_price):
    shares       = pos_usd / entry_price
    comm_per_leg = max(1.0, shares * 0.005)
    comm_per_leg = min(comm_per_leg, pos_usd * 0.01)
    return comm_per_leg * 2


# ─── Симуляция счёта ─────────────────────────────────────────────────────────
def simulate(trades_df, max_positions=4, use_leverage=False, flat_commission=False):
    df       = trades_df.sort_values('date').copy()
    account  = START_BALANCE
    open_pos = []
    rows     = []
    total_commission = 0.0
    total_margin     = 0.0

    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)
        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= max_positions:
            continue
        open_pos.append(x_date)

        pos = (account * RISK_PCT) / (t['risk_pct'] / 100)

        if flat_commission:
            commission = 2.0
        else:
            commission = ibkr_commission(pos, t['entry'])

        if use_leverage:
            margin_cost = (pos * 0.5) * MARGIN_RATE_2026 * (int(t['days']) / 365)
        else:
            margin_cost = 0.0

        pnl_usd  = pos * (t['pnl_pct'] / 100) - commission - margin_cost
        account += pnl_usd
        total_commission += commission
        total_margin     += margin_cost

        rows.append({
            'date':        str(t['date'])[:10],
            'month':       int(e_date.month),
            'symbol':      t['symbol'],
            'win':         bool(t['win']),
            'reason':      t['reason'],
            'pnl_pct':     t['pnl_pct'],
            'pos':         pos,
            'pnl':         pnl_usd,
            'commission':  commission,
            'margin_cost': margin_cost,
            'account':     account,
        })

    return pd.DataFrame(rows), account, total_commission, total_margin


# ─── Главная функция ──────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Бэктест Type B — 2026 (январь–апрель)")
    print("=" * 70)

    # 1. Список S&P500
    print("\nЗагружаем S&P500...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers, timeout=15)
    wiki = pd.read_html(io.StringIO(resp.text))[0]
    wiki = wiki[~wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols    = wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map = dict(zip(wiki['Symbol'].str.replace('.', '-', regex=False), wiki['GICS Sector']))
    print(f"  Акций: {len(symbols)}")

    # 2. SPY + секторные ETF
    print("Загружаем SPY + секторные ETF...")
    spx_raw = yf.download('SPY', start=DATA_START, progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close   = spx_raw['Close'].squeeze()
    sec_leaders = build_sector_leaders(spx_close)

    # 3. Даты отчётностей
    print("Загружаем кэш отчётов...")
    earnings = load_earnings()

    # 4. Котировки акций
    print(f"Загружаем котировки {len(symbols)} акций (2024–2026)...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw   = yf.download(batch, start=DATA_START, progress=False,
                            auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except:
                pass
        print(f"  [{min(i+50, len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    # 5. Генерация сигналов
    print("\nСканируем сигналы...")
    all_trades = []
    for sym, df in all_data.items():
        sec    = sector_map.get(sym)
        trades = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec)
        all_trades.extend(trades)

    if not all_trades:
        print("  Сигналов не найдено!")
        return

    df_all = pd.DataFrame(all_trades)
    df_all['date'] = pd.to_datetime(df_all['date'])
    print(f"  Всего сигналов: {len(df_all)}")
    print(f"  WR (все): {df_all['win'].mean()*100:.0f}%")

    # Сохраняем
    csv_path = 'd:/projects/trading/signals_type_b_2026.csv'
    df_all.to_csv(csv_path, index=False)
    print(f"  Сохранено: signals_type_b_2026.csv")

    # 6. Симуляция — 3 сценария
    sim_a, final_a, comm_a, marg_a = simulate(df_all, max_positions=4,
                                               use_leverage=False, flat_commission=True)
    sim_b, final_b, comm_b, marg_b = simulate(df_all, max_positions=4,
                                               use_leverage=False, flat_commission=False)
    sim_c, final_c, comm_c, marg_c = simulate(df_all, max_positions=8,
                                               use_leverage=True,  flat_commission=False)

    # 7. Вывод — сделки сценария B (самый реалистичный без плеча)
    print("\n" + "=" * 70)
    print("  ВСЕ СИГНАЛЫ 2026 (в хронологическом порядке)")
    print("=" * 70)
    months = {1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель'}
    cur_month = None
    for _, r in sim_a.iterrows():
        m = r['month']
        if m != cur_month:
            cur_month = m
            print(f"\n  ── {months.get(m, m)} ──")
        icon = "WIN ✓" if r['win'] else "LOSS✗"
        print(f"  {r['date']}  {r['symbol']:<6}  {icon}  {r['pnl_pct']:>+7.1f}%  "
              f"({r['reason']:<8})  счёт: ${r['account']:>9,.0f}")

    if not sim_a.empty:
        wr_a = sim_a['win'].mean() * 100
        print(f"\n  Итог: {len(sim_a)} сделок, WR {wr_a:.0f}%, "
              f"счёт: ${final_a:,.0f} (было $10,000, {(final_a/10000-1)*100:+.1f}%)")

    # 8. Итоговая таблица
    print("\n" + "=" * 70)
    print("  СРАВНЕНИЕ: без плеча vs 2x плечо (2026 год)")
    print("=" * 70)

    scenarios = [
        ("A — $2 плоско (baseline)",            sim_a, final_a, comm_a, marg_a),
        ("B — IBKR комиссия, без маржи",         sim_b, final_b, comm_b, marg_b),
        ("C — 2x плечо, IBKR + 5.5% маржа",     sim_c, final_c, comm_c, marg_c),
    ]
    for label, sim, final, comm, marg in scenarios:
        if sim.empty:
            print(f"\n  {label}: нет сделок")
            continue
        wr = sim['win'].mean() * 100
        chg = (final / START_BALANCE - 1) * 100
        print(f"\n  {label}")
        print(f"    Сделок:        {len(sim)}")
        print(f"    WR:            {wr:.0f}%")
        print(f"    Итог:          ${final:>10,.0f}  ({chg:+.1f}%)")
        print(f"    Комиссии:      ${comm:>8,.2f}")
        print(f"    Маржа:         ${marg:>8,.2f}")

    print("\n" + "─" * 70)
    if not sim_b.empty and not sim_c.empty:
        diff = final_c - final_b
        print(f"  Выигрыш 2x плечо vs без плеча:  ${diff:+,.0f}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
