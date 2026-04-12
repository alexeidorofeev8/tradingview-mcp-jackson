"""
Sniper Pullback Type B — тестирование 5 улучшений по одному

Каждый фильтр тестируется изолированно от базовой стратегии (bars=1, breach=1.5%).

Конфиги:
  0. Baseline  — текущие параметры
  1. low_vol   — объём на день пробоя < среднего (шейкаут)
  2. top_half  — закрытие свечи восстановления в верхней половине диапазона
  3. vix_25    — VIX < 25 (нет хаоса)
  4. rs_trend  — RS растёт (сейчас > 10 дней назад)
  5. intraday  — вчера закрылись ВЫШЕ уровня (пробой только внутри дня)
"""

import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Настройки ────────────────────────────────────────────────────
DATA_START  = '2018-01-01'
TEST_START  = '2020-01-01'
TEST_END    = '2026-04-11'

START_BALANCE = 4000.0
RISK_PCT      = 0.01
COMMISSION    = 2.0
MAX_POSITIONS = 4

EARNINGS_BUFFER = 7
TOP_SECTORS     = 5
HOURLY_DISC     = 0.01

SMA_FAST, SMA_SLOW      = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
SR_LOOKBACK_MIN          = 20
SR_LOOKBACK_MAX          = 80
MAX_ATR_PCT              = 5.0
MIN_PRICE                = 20
MIN_VOL                  = 500_000
EXCLUDE_SECTORS          = {'Real Estate', 'Utilities'}

# Type B baseline параметры
B_BREAKDOWN_BARS  = 1
B_MAX_BREACH_PCT  = 1.5
B_RECOVERY_BODY   = 0.5
B_RECOVERY_VOL    = 1.2
B_STOP_BUFFER     = 0.5

SECTOR_ETF = {
    'Information Technology': 'XLK', 'Financials': 'XLF',
    'Energy': 'XLE',                  'Health Care': 'XLV',
    'Industrials': 'XLI',             'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',        'Materials': 'XLB',
    'Communication Services': 'XLC',
}
EARNINGS_CACHE = 'd:/projects/trading/earnings_cache.json'


# ─── Вспомогательные ──────────────────────────────────────────────
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_earnings(symbols):
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


# ─── Сигналы Type B с опциональными фильтрами ─────────────────────
def signals_type_b(close, high, low, open_, volume, atr, vol_ma, sma50, sma200,
                   spx, rs, vix=None, extra=''):
    """
    extra: '' | 'low_vol' | 'top_half' | 'vix_25' | 'rs_trend' | 'intraday'
    """
    prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
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

    base = trend_ok & quality_ok & false_break & recovered & strong_recovery & market_ok

    # ── Дополнительные фильтры ──
    if extra == 'low_vol':
        # Объём вчера (день пробоя) ниже среднего — шейкаут, не настоящее давление
        base = base & (volume.shift(1) < vol_ma)

    elif extra == 'top_half':
        # Свеча восстановления закрывается в верхней половине диапазона
        base = base & (close > (high + low) / 2)

    elif extra == 'vix_25':
        # VIX ниже 25 — не хаос
        if vix is not None:
            vix_aligned = vix.reindex(close.index, method='ffill')
            base = base & (vix_aligned < 25)

    elif extra == 'rs_trend':
        # RS растёт (сейчас > 10 дней назад)
        base = base & (rs > rs.shift(10))

    elif extra == 'intraday':
        # Вчера ЗАКРЫЛИСЬ выше уровня — пробой был только внутри дня (чистый шейкаут)
        intraday_ph = close.shift(1) >= prior_high * 0.998
        intraday_lm = close.shift(1) >= local_min  * 0.998
        # Применяем только к соответствующим уровням
        intraday_ok = (
            (false_break_ph & intraday_ph) |
            (false_break_lm & intraday_lm)
        )
        base = base & intraday_ok

    return base


# ─── Бэктест одной акции ──────────────────────────────────────────
def run_symbol(sym, df_stock, spx_close, earnings, sec_leaders, stock_sec, vix_close=None, extra=''):
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
        vix    = vix_close.reindex(close.index, method='ffill') if vix_close is not None else None

        mask = (close.index >= TEST_START) & (close.index <= TEST_END)
        sig  = signals_type_b(close, high, low, open_, volume, atr, vol_ma,
                               sma50, sma200, spx, rs, vix=vix, extra=extra)

        results = []
        bars = list(close.index)
        earn_set = earnings.get(sym, set())

        for sig_date in sig[mask & sig].index:
            sig_d = sig_date.date()
            if any(abs((pd.to_datetime(ed).date() - sig_d).days) <= EARNINGS_BUFFER for ed in earn_set):
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
                'symbol':   sym,
                'date':     sig_date,
                'entry':    entry_price,
                'exit':     exit_p,
                'sl':       sl_init,
                'tp':       tp,
                'rr':       actual_rr,
                'risk_pct': risk_pct,
                'pnl_pct':  (exit_p - entry_price) / entry_price * 100,
                'win':      exit_p >= entry_price,
                'reason':   reason,
                'days':     hold,
                'year':     sig_date.year,
            })
        return results
    except:
        return []


# ─── Симуляция счёта ──────────────────────────────────────────────
def simulate_account(trades_df):
    df = trades_df.sort_values('date').copy()
    account  = START_BALANCE
    open_pos = []
    rows = []
    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)
        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= MAX_POSITIONS: continue
        open_pos.append(x_date)
        pos     = (account * RISK_PCT) / (t['risk_pct'] / 100)
        pnl_usd = pos * (t['pnl_pct'] / 100) - COMMISSION
        account += pnl_usd
        rows.append({
            'date':    t['date'],
            'pnl_$':   round(pnl_usd, 2),
            'account': round(account, 2),
            'symbol':  t['symbol'],
            'win':     t['win'],
        })
    return pd.DataFrame(rows), account


# ─── Main ─────────────────────────────────────────────────────────
def main():
    print("=== Type B: тестирование 5 улучшений ===\n")

    print("Шаг 1: S&P 500 список...")
    import requests
    import io as _io
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers, timeout=15)
    df_wiki = pd.read_html(_io.StringIO(resp.text))[0]
    df_wiki = df_wiki[~df_wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols = df_wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map = dict(zip(
        df_wiki['Symbol'].str.replace('.', '-', regex=False),
        df_wiki['GICS Sector']
    ))
    print(f"  Акций: {len(symbols)}")

    print("\nШаг 2: Загружаем SPY + VIX + секторы...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()

    vix_raw = yf.download('^VIX', start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix_close = vix_raw['Close'].squeeze()

    sec_leaders = build_sector_leaders(spx_close)

    print("\nШаг 3: Загружаем даты отчётов...")
    earnings = load_earnings(symbols)

    print(f"\nШаг 4: Загружаем котировки {len(symbols)} акций...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw = yf.download(batch, start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except: pass
        print(f"  [{min(i+50, len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    configs = [
        ('',         'Baseline (текущий)              '),
        ('low_vol',  'Фильтр 1: низкий объём пробоя   '),
        ('top_half', 'Фильтр 2: закрытие в верх.пол.  '),
        ('vix_25',   'Фильтр 3: VIX < 25              '),
        ('rs_trend', 'Фильтр 4: RS растёт (>10 дней)  '),
        ('intraday', 'Фильтр 5: пробой только внутридня'),
    ]

    results = {}
    for extra, label in configs:
        print(f"\nШаг 5: {label.strip()}...")
        all_trades = []
        for sym, df in all_data.items():
            sec = sector_map.get(sym)
            trades = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec,
                                vix_close=vix_close, extra=extra)
            all_trades.extend(trades)

        if not all_trades:
            print("  Нет сделок!")
            continue

        df_trades = pd.DataFrame(all_trades)
        df_trades['date'] = pd.to_datetime(df_trades['date'])
        sim, final = simulate_account(df_trades)
        results[extra] = (label, df_trades, sim, final)
        wr   = sim['win'].mean() * 100 if len(sim) > 0 else 0
        cagr = (final / START_BALANCE) ** (1/6.3) - 1
        print(f"  → {len(sim)} сделок, WR {wr:.1f}%, итог ${final:,.0f}, CAGR {cagr*100:.1f}%")

    # ── Итоговая таблица ──
    print(f"\n\n{'='*72}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{'='*72}")
    print(f"  {'Конфиг':<36} {'Сделок':>7} {'Win%':>6} {'Итог':>10} {'CAGR':>8} {'vs Base':>8}")
    print(f"  {'-'*68}")

    base_final = None
    for extra, label in configs:
        if extra not in results: continue
        lbl, df_t, sim, final = results[extra]
        wr   = sim['win'].mean() * 100 if len(sim) > 0 else 0
        cagr = (final / START_BALANCE) ** (1/6.3) - 1
        if base_final is None:
            base_final = final
            vs = '  —'
        else:
            diff = final - base_final
            vs = f'${diff:+,.0f}'
        print(f"  {lbl} {len(sim):>7}  {wr:>5.1f}%  ${final:>8,.0f}  {cagr*100:>7.1f}%  {vs:>8}")

    # ── По годам для каждого конфига ──
    print(f"\n\n{'='*72}")
    print("  ПО ГОДАМ")
    print(f"{'='*72}")

    for extra, label in configs:
        if extra not in results: continue
        lbl, df_t, sim, final = results[extra]
        if len(sim) == 0: continue
        print(f"\n  {lbl.strip()}")
        print(f"  {'Год':<5} {'Сд':>4} {'Win%':>6} {'P&L $':>8} {'CAGR yr':>8}")
        account = START_BALANCE
        for yr, g in sim.groupby(sim['date'].dt.year):
            start = account
            end   = g['account'].iloc[-1]
            account = end
            wr  = g['win'].mean() * 100
            pnl = g['pnl_$'].sum()
            cagr_yr = (end / start) - 1
            print(f"  {yr:<5} {len(g):>4}  {wr:>5.1f}%  ${pnl:>+7,.0f}  {cagr_yr*100:>+7.1f}%")
        total_cagr = (final / START_BALANCE) ** (1/6.3) - 1
        print(f"  {'ИТОГО':<5} {len(sim):>4}  {'':>6}  {'':>8}  CAGR {total_cagr*100:.1f}%")


if __name__ == '__main__':
    main()
