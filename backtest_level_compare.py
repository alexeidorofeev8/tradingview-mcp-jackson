"""
Сравнение трёх методов расчёта уровня для Type B (ложный пробой).

Вопрос: влияет ли окно расчёта prior_high на результаты стратегии?

Три конфигурации:
  1. Current  — shift 20, window 60 (текущий)
  2. Closer   — shift 5,  window 55 (более свежие уровни)
  3. Wider    — shift 10, window 110 (шире, стабильнее)
"""
import sys, io, json, os, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Настройки ────────────────────────────────────────────────────
DATA_START    = '2018-01-01'
TEST_START    = '2020-01-01'
TEST_END      = '2026-04-11'
START_BALANCE = 4000.0
RISK_PCT      = 0.01
COMMISSION    = 2.0
MAX_POSITIONS = 4

SMA_FAST, SMA_SLOW       = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
MAX_ATR_PCT = 5.0
MIN_PRICE   = 20
MIN_VOL     = 500_000
EXCLUDE_SECTORS = {'Real Estate', 'Utilities'}
TOP_SECTORS = 5
EARNINGS_BUFFER = 7
HOURLY_DISC = 0.01

B_BREAKDOWN_BARS = 3
B_MAX_BREACH_PCT = 3.0
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

# Три конфигурации уровня
CONFIGS = [
    ('Current  (shift 20–80)',   20,  80),
    ('Closer   (shift 5–60)',     5,  60),
    ('Wider    (shift 10–120)',  10, 120),
]


# ─── Вспомогательные ─────────────────────────────────────────────
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
    raw = yf.download(etfs, start=DATA_START, end='2027-01-01',
                      progress=False, auto_adjust=True, group_by='ticker')
    sec_prices = {}
    for sector, etf in SECTOR_ETF.items():
        try:
            sec_prices[sector] = raw[etf]['Close'].squeeze()
        except: pass
    idx = spx_close.index
    leaders = {}
    for i in range(RS_LEN, len(idx)):
        date = idx[i]
        spx_ret = spx_close.iloc[i] / spx_close.iloc[i - RS_LEN]
        scores = {}
        for sec, prices in sec_prices.items():
            p = prices.reindex(idx, method='ffill')
            p_now, p_ago = p.iloc[i], p.iloc[i - RS_LEN]
            if pd.isna(p_now) or pd.isna(p_ago) or p_ago == 0: continue
            scores[sec] = (p_now / p_ago) / spx_ret
        if len(scores) >= TOP_SECTORS:
            leaders[date] = set(sorted(scores, key=scores.get, reverse=True)[:TOP_SECTORS])
    return leaders


# ─── Сигналы Type B с переменным уровнем ─────────────────────────
def signals_type_b(close, high, low, open_, volume, atr, vol_ma,
                   sma50, sma200, spx, rs, sr_min, sr_max):
    prior_high = high.shift(sr_min).rolling(sr_max - sr_min).max()
    local_min  = low.rolling(15).min().shift(1)

    recent_low = low.shift(1).rolling(B_BREAKDOWN_BARS).min()

    false_break_ph = (
        (recent_low < prior_high) &
        (recent_low >= prior_high * (1 - B_MAX_BREACH_PCT / 100))
    )
    false_break_lm = (
        (recent_low < local_min) &
        (recent_low >= local_min * (1 - B_MAX_BREACH_PCT / 100))
    )
    false_break = false_break_ph | false_break_lm

    recovery_ph = (close > prior_high) & (low.shift(1) < prior_high)
    recovery_lm = (close > local_min)  & (low.shift(1) < local_min)

    strong_recovery = (
        (close > open_) &
        ((close - open_) / atr > B_RECOVERY_BODY) &
        (volume > vol_ma * B_RECOVERY_VOL)
    )

    recovered = ((recovery_ph | recovery_lm) & false_break) | (
        (close > prior_high * 0.999) & (close.shift(1) < prior_high) & false_break_ph |
        (close > local_min  * 0.999) & (close.shift(1) < local_min)  & false_break_lm
    )

    trend_ok   = (close > sma200) & (sma50 > sma200)
    quality_ok = ((atr / close * 100) < MAX_ATR_PCT) & (rs > 1.0) & (vol_ma > MIN_VOL) & (close > MIN_PRICE)
    market_ok  = spx > spx.rolling(200).mean()

    return trend_ok & quality_ok & false_break & recovered & strong_recovery & market_ok


# ─── Бэктест одной акции ─────────────────────────────────────────
def run_symbol(sym, df_stock, spx_close, earnings, sec_leaders, stock_sec, sr_min, sr_max):
    try:
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)
        close  = df_stock['Close'].squeeze()
        high   = df_stock['High'].squeeze()
        low    = df_stock['Low'].squeeze()
        open_  = df_stock['Open'].squeeze()
        volume = df_stock['Volume'].squeeze()
        if len(close) < 220: return []

        sma50  = close.rolling(SMA_FAST).mean()
        sma200 = close.rolling(SMA_SLOW).mean()
        atr    = calc_atr(high, low, close, ATR_LEN)
        vol_ma = volume.rolling(VOL_LEN).mean()
        spx    = spx_close.reindex(close.index, method='ffill')
        rs     = (close / close.shift(RS_LEN)) / (spx / spx.shift(RS_LEN))

        earn_set = earnings.get(sym, set())
        mask = (close.index >= TEST_START) & (close.index <= TEST_END)
        sig_b = signals_type_b(close, high, low, open_, volume, atr, vol_ma,
                               sma50, sma200, spx, rs, sr_min, sr_max)
        bars = list(close.index)
        results = []

        for sig_date in sig_b[mask & sig_b].index:
            sig_d = sig_date.date()
            if any(0 <= (pd.to_datetime(ed).date() - sig_d).days <= EARNINGS_BUFFER for ed in earn_set):
                continue
            if stock_sec and sig_date in sec_leaders:
                if stock_sec not in sec_leaders[sig_date]:
                    continue

            idx_i = bars.index(sig_date)
            entry_price = close.iloc[idx_i] * (1 - HOURLY_DISC)

            lookback = min(B_BREAKDOWN_BARS + 1, idx_i)
            recent_lows = low.iloc[idx_i - lookback: idx_i]
            breakdown_low = recent_lows.min() if len(recent_lows) > 0 else low.iloc[idx_i]
            sl_init = breakdown_low - atr.iloc[idx_i] * B_STOP_BUFFER

            risk = entry_price - sl_init
            risk_pct = risk / entry_price * 100
            if risk <= 0 or risk_pct > 15: continue

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
            if actual_rr < 1.0: continue

            exit_p, reason, hold = None, 'timeout', 0
            for j in range(1, 21):
                if idx_i + j >= len(bars): break
                bh = high.iloc[idx_i + j]; bl = low.iloc[idx_i + j]
                bc = close.iloc[idx_i + j]; bs = sma50.iloc[idx_i + j]; ba = atr.iloc[idx_i + j]
                hold = j
                if bl <= sl_init:    exit_p = sl_init; reason = 'stop';      break
                if bh >= tp:         exit_p = tp;      reason = 'target';    break
                if bc < bs - ba*0.5: exit_p = bc;      reason = 'below_sma'; break
            if exit_p is None:
                exit_p = close.iloc[min(idx_i + hold, len(close) - 1)]

            results.append({
                'symbol':  sym,
                'date':    sig_date,
                'entry':   entry_price,
                'exit':    exit_p,
                'sl':      sl_init,
                'tp':      tp,
                'rr':      actual_rr,
                'risk_pct': risk_pct,
                'pnl_pct': (exit_p - entry_price) / entry_price * 100,
                'win':     exit_p >= entry_price,
                'reason':  reason,
                'days':    hold,
                'year':    sig_date.year,
            })
        return results
    except:
        return []


# ─── Симуляция счёта ─────────────────────────────────────────────
def simulate_account(trades_df):
    df = trades_df.sort_values('date').copy()
    account, open_pos, rows = START_BALANCE, [], []
    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)
        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= MAX_POSITIONS: continue
        open_pos.append(x_date)
        pos     = (account * RISK_PCT) / (t['risk_pct'] / 100)
        pnl_usd = pos * (t['pnl_pct'] / 100) - COMMISSION
        account += pnl_usd
        rows.append({'date': t['date'], 'pnl_$': pnl_usd,
                     'account': account, 'win': t['win'], 'year': t['year']})
    return pd.DataFrame(rows), account


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=== Сравнение методов расчёта уровня — Type B ===\n")

    print("Загружаем S&P 500...")
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
    df_wiki = pd.read_html(io.StringIO(resp.text))[0]
    df_wiki = df_wiki[~df_wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols = df_wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map = dict(zip(df_wiki['Symbol'].str.replace('.', '-', regex=False),
                          df_wiki['GICS Sector']))
    print(f"  Акций: {len(symbols)}")

    print("Загружаем SPY + секторы...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01',
                           progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()
    sec_leaders = build_sector_leaders(spx_close)

    print("Загружаем кэш отчётов...")
    earnings = load_earnings()

    print(f"Загружаем котировки {len(symbols)} акций...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw = yf.download(batch, start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except: pass
        print(f"  [{min(i+50,len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    # Запускаем три конфигурации
    results_per_config = {}
    for label, sr_min, sr_max in CONFIGS:
        print(f"\nБэктест: {label}...")
        trades = []
        for sym, df in all_data.items():
            sec = sector_map.get(sym)
            t = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec, sr_min, sr_max)
            trades.extend(t)
        df_t = pd.DataFrame(trades)
        if len(df_t) == 0:
            print(f"  Нет сделок!")
            continue
        df_t['date'] = pd.to_datetime(df_t['date'])
        sim, final = simulate_account(df_t)
        results_per_config[label] = (df_t, sim, final)
        print(f"  Сделок: {len(sim)}, Финал: ${final:,.0f}")

    # Итоговая таблица
    print(f"\n{'='*72}")
    print(f"  ИТОГОВОЕ СРАВНЕНИЕ МЕТОДОВ УРОВНЯ")
    print(f"{'='*72}")
    print(f"  {'Метод':<28} {'Сд':>5} {'Win%':>6} {'$4k→':>10} {'CAGR':>7}")
    print(f"  {'-'*60}")
    for label, (df_t, sim, final) in results_per_config.items():
        wr   = sim['win'].mean() * 100
        cagr = (final / START_BALANCE) ** (1/6.3) - 1
        print(f"  {label:<28} {len(sim):>5}  {wr:>5.1f}%  ${final:>8,.0f}  {cagr*100:>6.1f}%")

    # По годам для каждой конфигурации
    for label, (df_t, sim, final) in results_per_config.items():
        print(f"\n{'='*72}")
        print(f"  {label}")
        print(f"{'='*72}")
        print(f"  {'Год':<5} {'Сд':>4} {'Win%':>6} {'P&L $':>8} {'P&L %':>7} {'Счёт':>9}")
        print(f"  {'-'*55}")
        account = START_BALANCE
        for yr, g in sim.groupby(sim['date'].dt.year):
            end = g['account'].iloc[-1]
            wr  = g['win'].mean() * 100
            pnl = g['pnl_$'].sum()
            pct = (end - account) / account * 100
            print(f"  {yr:<5} {len(g):>4}  {wr:>5.1f}%  ${pnl:>+7,.0f}  {pct:>+6.1f}%  ${end:>8,.0f}")
            account = end

    print("\nГотово.")


if __name__ == '__main__':
    import io
    main()
