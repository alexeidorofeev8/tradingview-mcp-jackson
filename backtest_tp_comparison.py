"""
Сравнение TP-множителей (R:R) — бэктест 2010–2026.
Загружает данные один раз, тестирует 7 сценариев параллельно.
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# ─── Параметры ───────────────────────────────────────────────────────────────
DATA_START  = '2008-01-01'
TEST_START  = pd.Timestamp('2010-01-01')
TEST_END    = pd.Timestamp('2026-04-11')

START_BALANCE = 10_000.0
RISK_PCT      = 0.01
COMMISSION    = 2.0

SMA_FAST, SMA_SLOW       = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
SR_LOOKBACK_MIN          = 20
SR_LOOKBACK_MAX          = 80
MAX_ATR_PCT              = 5.0
MIN_PRICE                = 20
MIN_VOL                  = 500_000
EXCLUDE_SECTORS          = {'Real Estate', 'Utilities'}
EARNINGS_BUFFER          = 7
HOURLY_DISC              = 0.01

B_BREAKDOWN_BARS = 1
B_MAX_BREACH_PCT = 1.5
B_RECOVERY_BODY  = 0.5
B_RECOVERY_VOL   = 1.2
B_STOP_BUFFER    = 0.5

EARNINGS_CACHE = 'd:/projects/trading/earnings_cache.json'

IBKR_RATES = {
    2010: 0.0175, 2011: 0.0160, 2012: 0.0164, 2013: 0.0161,
    2014: 0.0159, 2015: 0.0163, 2016: 0.0190, 2017: 0.0250,
    2018: 0.0341, 2019: 0.0366, 2020: 0.0159, 2021: 0.0157,
    2022: 0.0318, 2023: 0.0652, 2024: 0.0680, 2025: 0.0600,
    2026: 0.0550,
}

# TP сценарии: (метка, множитель | None=dynamic)
TP_SCENARIOS = [
    ('1.5R',    1.5),
    ('2.0R',    2.0),
    ('2.5R',    2.5),
    ('3.0R',    3.0),
    ('4.0R',    4.0),
    ('5.0R',    5.0),
    ('dynamic', None),   # сопротивление [1.5–5.0], дефолт 3.0R
]


# ─── Вспомогательные ─────────────────────────────────────────────────────────
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_earnings():
    if not os.path.exists(EARNINGS_CACHE):
        return {}
    raw = json.load(open(EARNINGS_CACHE, encoding='utf-8'))
    result = {}
    for s, v in raw.items():
        if isinstance(v, list):
            result[s] = set(v)
        elif isinstance(v, dict):
            result[s] = set(v.get('dates', []))
    return result


def get_sp500():
    resp = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15
    )
    df = pd.read_html(io.StringIO(resp.text))[0]
    df = df[~df['GICS Sector'].isin(EXCLUDE_SECTORS)]
    syms = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    secs = dict(zip(df['Symbol'].str.replace('.', '-', regex=False), df['GICS Sector']))
    return syms, secs


# ─── Симуляция выхода ─────────────────────────────────────────────────────────
def simulate_exit(high, low, close, sma50, atr, idx_i, sl, tp_val):
    """Возвращает (exit_price, reason, hold_days)."""
    exit_p, reason, hold = None, 'timeout', 0
    n = len(close)
    for j in range(1, 21):
        if idx_i + j >= n:
            break
        bh = high.iloc[idx_i + j]
        bl = low.iloc[idx_i + j]
        bc = close.iloc[idx_i + j]
        bs = sma50.iloc[idx_i + j]
        ba = atr.iloc[idx_i + j]
        hold = j
        if bl <= sl:           exit_p = sl;     reason = 'stop';   break
        if bh >= tp_val:       exit_p = tp_val; reason = 'target'; break
        if bc < bs - ba * 0.5: exit_p = bc;     reason = 'sma';    break
    if exit_p is None:
        exit_p = close.iloc[min(idx_i + hold, n - 1)]
    return exit_p, reason, hold


# ─── Сканирование одной акции ─────────────────────────────────────────────────
def run_symbol(sym, df_stock, spx_close, earnings):
    """Возвращает список сигналов с результатами выхода по всем TP-сценариям."""
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

        prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        local_min  = low.rolling(15).min().shift(1)
        recent_low = low.shift(1).rolling(B_BREAKDOWN_BARS).min()

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

        earn_set = earnings.get(sym, set())
        bars = list(close.index)
        results = []

        for sig_date in sig_mask[date_mask & sig_mask].index:
            sig_d = sig_date.date()
            if any(0 <= (pd.to_datetime(ed).date() - sig_d).days <= EARNINGS_BUFFER for ed in earn_set):
                continue

            idx_i = bars.index(sig_date)
            entry = close.iloc[idx_i] * (1 - HOURLY_DISC)

            lookback = min(B_BREAKDOWN_BARS + 1, idx_i)
            recent_lows = low.iloc[idx_i - lookback: idx_i]
            breakdown_low = recent_lows.min() if len(recent_lows) > 0 else low.iloc[idx_i]
            sl = breakdown_low - atr.iloc[idx_i] * B_STOP_BUFFER

            risk = entry - sl
            risk_pct = risk / entry * 100
            if risk <= 0 or risk_pct > 15:
                continue

            # Dynamic TP (как в оригинальном бэктесте)
            res_cands = [
                high.iloc[max(0, idx_i - lb):idx_i].max()
                for lb in [10, 20, 40, 60]
                if high.iloc[max(0, idx_i - lb):idx_i].max() > entry * 1.01
            ]
            if res_cands:
                dynamic_rr = max(1.5, min(5.0, (min(res_cands) - entry) / risk))
            else:
                dynamic_rr = 3.0
            dynamic_tp = entry + risk * dynamic_rr

            # Симулируем выход для каждого TP-сценария
            exits = {}
            for label, mult in TP_SCENARIOS:
                tp_val = entry + risk * mult if mult is not None else dynamic_tp
                exit_p, reason, hold = simulate_exit(high, low, close, sma50, atr, idx_i, sl, tp_val)
                pnl_pct = (exit_p - entry) / entry * 100
                exits[label] = {
                    'exit_p':  exit_p,
                    'reason':  reason,
                    'hold':    hold,
                    'pnl_pct': pnl_pct,
                    'win':     exit_p >= entry,
                }

            results.append({
                'date':     sig_date,
                'symbol':   sym,
                'entry':    entry,
                'risk_pct': risk_pct,
                'exits':    exits,
            })

        return results
    except Exception:
        return []


# ─── Симуляция счёта ──────────────────────────────────────────────────────────
def simulate_account(signals, scenario_label, max_positions=4, use_leverage=False):
    account  = START_BALANCE
    open_pos = []
    rows     = []

    for sig in signals:
        e_date = pd.to_datetime(sig['date'])
        ex     = sig['exits'][scenario_label]
        x_date = e_date + pd.Timedelta(days=int(ex['hold']) + 1)

        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= max_positions:
            continue
        open_pos.append(x_date)

        pos     = (account * RISK_PCT) / (sig['risk_pct'] / 100)
        margin  = 0.0
        if use_leverage:
            year   = e_date.year
            rate   = IBKR_RATES.get(year, 0.06)
            margin = (pos * 0.5) * rate * (ex['hold'] / 365)
        pnl_usd = pos * (ex['pnl_pct'] / 100) - COMMISSION - margin
        account += pnl_usd

        rows.append({
            'date':    sig['date'],
            'symbol':  sig['symbol'],
            'win':     ex['win'],
            'hold':    ex['hold'],
            'reason':  ex['reason'],
            'pnl_usd': pnl_usd,
            'account': account,
        })

    return pd.DataFrame(rows), account


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=== Сравнение TP-множителей (R:R) — Type B 2010–2026 ===\n")

    # SPY
    spx_raw   = yf.download('SPY', start=DATA_START, progress=False, auto_adjust=True)
    spx_close = spx_raw['Close'].squeeze()

    # S&P500
    print("Загружаем список S&P500...")
    symbols, _ = get_sp500()
    earnings   = load_earnings()
    print(f"  Акций: {len(symbols)}")

    # Котировки
    print(f"Загружаем котировки {len(symbols)} акций (2008–2026)...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw = yf.download(batch, start=DATA_START, progress=False,
                          auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except:
                pass
        print(f"  [{min(i+50, len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    # Сигналы
    print("\nСканируем сигналы...")
    all_signals = []
    for sym, df in all_data.items():
        all_signals.extend(run_symbol(sym, df, spx_close, earnings))
    all_signals.sort(key=lambda x: x['date'])
    print(f"  Всего сигналов (до слотов): {len(all_signals)}")

    years = (TEST_END - TEST_START).days / 365.25

    def print_table(title, slots, leverage):
        print()
        print('=' * 78)
        print(f'  {title}')
        print('=' * 78)
        print(f"  {'TP':<10} {'Сделок':>7} {'WR':>5} {'Avg дн':>7} {'→Tgt':>6} {'→Stop':>6} {'CAGR':>7}  {'Итог':>12}")
        print('  ' + '─' * 73)
        summary = []
        sims = {}
        for label, _ in TP_SCENARIOS:
            sim, final = simulate_account(all_signals, label,
                                          max_positions=slots,
                                          use_leverage=leverage)
            sims[label] = sim
            if len(sim) == 0:
                continue
            cagr     = (final / START_BALANCE) ** (1 / years) - 1
            wr       = sim['win'].mean() * 100
            avg_h    = sim['hold'].mean()
            tgt_pct  = (sim['reason'] == 'target').mean() * 100
            stop_pct = (sim['reason'] == 'stop').mean() * 100
            summary.append((label, len(sim), wr, avg_h, tgt_pct, stop_pct, cagr, final))
            marker = ' ←' if label == 'dynamic' else ''
            print(f"  {label:<10} {len(sim):>7} {wr:>4.0f}% {avg_h:>6.1f}d "
                  f"{tgt_pct:>5.0f}% {stop_pct:>5.0f}% {cagr*100:>6.1f}%  "
                  f"${final:>11,.0f}{marker}")
        best = max(summary, key=lambda x: x[6])
        print('  ' + '─' * 73)
        print(f"  Лучший CAGR: {best[0]}  ({best[6]*100:.1f}%  итог ${best[7]:,.0f})")
        print('=' * 78)

        # По годам — лучший сценарий
        sim_best = sims[best[0]].copy()
        sim_best['year'] = pd.to_datetime(sim_best['date']).dt.year
        print(f"\n  По годам — {best[0]}:")
        print(f"  {'Год':<5} {'Сделок':>7} {'WR':>5} {'Avg дн':>7} {'→Tgt':>6} {'→Stop':>6} {'Счёт':>12}")
        print('  ' + '─' * 57)
        for yr, g in sim_best.groupby('year'):
            print(f"  {yr:<5} {len(g):>7} {g['win'].mean()*100:>4.0f}% "
                  f"{g['hold'].mean():>6.1f}d "
                  f"{(g['reason']=='target').mean()*100:>5.0f}% "
                  f"{(g['reason']=='stop').mean()*100:>5.0f}% "
                  f"${g['account'].iloc[-1]:>11,.0f}")

    print_table('4 СЛОТА — БЕЗ ПЛЕЧА  ($10k старт, 2010–2026)', slots=4, leverage=False)
    print_table('8 СЛОТОВ — 2x ПЛЕЧО  ($10k старт, 2010–2026)', slots=8, leverage=True)
    print()


if __name__ == '__main__':
    main()
