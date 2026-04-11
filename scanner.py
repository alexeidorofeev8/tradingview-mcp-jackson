"""
Sniper Pullback Scanner — находит акции близко к точке входа прямо сейчас
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_START = '2023-01-01'
SMA_FAST, SMA_SLOW = 50, 200
ATR_LEN = 14
RS_LEN  = 63
VOL_LEN = 20
SR_LOOKBACK_MIN = 20
SR_LOOKBACK_MAX = 80
SR_TOUCH_PCT    = 2.5
CONSOL_BARS     = 5
CONSOL_RANGE    = 1.8
CONSOL_VOL_MIN  = 0.8
EXCLUDE_SECTORS = {'Real Estate', 'Utilities'}

def get_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=15)
    df = pd.read_html(io.StringIO(resp.text))[0]
    filtered = df[~df['GICS Sector'].isin(EXCLUDE_SECTORS)]
    return filtered['Symbol'].str.replace('.', '-', regex=False).tolist()

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def scan_symbol(symbol, df, spx_close):
    try:
        if len(df) < 220:
            return None
        close  = df['Close'].squeeze()
        high   = df['High'].squeeze()
        low    = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        atr    = calc_atr(high, low, close, ATR_LEN)
        sma50  = close.rolling(SMA_FAST).mean()
        sma200 = close.rolling(SMA_SLOW).mean()
        vol_ma = volume.rolling(VOL_LEN).mean()

        # Последние значения
        c   = close.iloc[-1]
        a   = atr.iloc[-1]
        s50 = sma50.iloc[-1]
        s200= sma200.iloc[-1]
        vm  = vol_ma.iloc[-1]
        v   = volume.iloc[-1]

        # Базовые фильтры
        if c < 20: return None
        if vm < 500_000: return None
        if a/c*100 > 5.0: return None
        if c < s200: return None
        if s50 < s200: return None

        # SPY market filter
        spx_aligned = spx_close.reindex(close.index, method='ffill')
        spx_sma200  = spx_aligned.rolling(SMA_SLOW).mean()
        if spx_aligned.iloc[-1] < spx_sma200.iloc[-1]:
            return None

        # RS vs SPX
        rs_stock = close / close.shift(RS_LEN)
        rs_spx   = spx_aligned / spx_aligned.shift(RS_LEN)
        rs_ratio = (rs_stock / rs_spx).iloc[-1]
        if rs_ratio < 1.0: return None

        # --- Уровни ---
        # S/R flip: бывшее сопротивление стало поддержкой
        prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        dist_sr = (c - prior_high.iloc[-1]) / prior_high.iloc[-1] * 100 if not pd.isna(prior_high.iloc[-1]) else 99
        near_sr = abs(dist_sr) < SR_TOUCH_PCT

        # Pivot Low: локальный минимум 15 баров
        pivot_low = low.rolling(15).min().iloc[-1]
        dist_pivot = (c - pivot_low) / pivot_low * 100
        near_pivot = abs(dist_pivot) < 2.0

        # SMA50
        dist_sma50 = (c - s50) / s50 * 100
        near_sma50 = abs(dist_sma50) < 2.0

        at_level = near_sr or near_pivot or near_sma50
        if not at_level: return None

        # --- Поглощение ---
        recent_high = high.iloc[-CONSOL_BARS:-1]
        recent_low  = low.iloc[-CONSOL_BARS:-1]
        consol_range = (recent_high.max() - recent_low.min()) / a
        tight = consol_range < CONSOL_RANGE

        consol_vol = volume.iloc[-CONSOL_BARS:-1].mean()
        vol_ok = consol_vol > vm * CONSOL_VOL_MIN

        # Уровень (текущая цена vs S/R тип)
        if near_sr:
            level_type = "S/R flip"
            level_dist = dist_sr
        elif near_pivot:
            level_type = "Pivot Low"
            level_dist = dist_pivot
        else:
            level_type = "SMA50"
            level_dist = dist_sma50

        # Найти ближайшее сопротивление (тейк)
        res_candidates = []
        for lb in [10, 20, 40, 60]:
            start = max(0, len(high) - lb)
            res = high.iloc[start:-1].max()
            if res > c * 1.01:
                res_candidates.append(res)
        if res_candidates:
            tp = min(res_candidates)
            sl = s50 - 1.5 * a
            risk = c - sl
            reward = tp - c
            rr = reward / risk if risk > 0 else 0
        else:
            tp = 0
            rr = 0

        # Скор
        score = 0
        if near_sr:     score += 30
        elif near_pivot: score += 20
        elif near_sma50: score += 10
        if tight:        score += 20
        if vol_ok:       score += 15
        score += min(int((rs_ratio - 1.0) * 100), 20)
        if rr >= 2.0:    score += 15

        tight_str = "<<TIGHT" if tight else ""
        return {
            'symbol':     symbol,
            'price':      round(c, 2),
            'level':      level_type,
            'dist_%':     round(level_dist, 1),
            'consol_atr': round(consol_range, 1),
            'tight':      tight_str,
            'vol_ok':     "Y" if vol_ok else "N",
            'RS':         round(rs_ratio, 2),
            'SMA50':      round(s50, 2),
            'TP':         round(tp, 2),
            'SL':         round(s50 - 1.5 * a, 2),
            'R:R':        round(rr, 1),
            'score':      score,
        }
    except Exception as e:
        return None

def main():
    print("=== Sniper Pullback Scanner — сегодня ===\n")

    symbols = get_sp500()
    print(f"Загружаем данные для {len(symbols)} акций...\n")

    # SPY для фильтра рынка и RS
    spx_raw = yf.download('SPY', start=DATA_START, progress=False, auto_adjust=True)
    spx_close = spx_raw['Close'].squeeze()

    # Скачиваем батчами
    batch_size = 50
    all_data = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        raw = yf.download(batch, start=DATA_START, progress=False, auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                if len(batch) == 1:
                    all_data[sym] = raw
                else:
                    all_data[sym] = raw[sym].dropna()
            except:
                pass
        print(f"  Загружено {min(i+batch_size, len(symbols))}/{len(symbols)}...", flush=True)

    print("\nСканируем...\n")
    results = []
    for sym, df in all_data.items():
        r = scan_symbol(sym, df, spx_close)
        if r:
            results.append(r)

    if not results:
        print("Нет сетапов сегодня.")
        return

    df_res = pd.DataFrame(results).sort_values('score', ascending=False)

    print(f"Найдено сетапов: {len(df_res)}\n")
    print("=" * 90)
    print(f"{'Sym':<6} {'Price':>7} {'Level':<10} {'Dist%':>6} {'Consol':>6} {'Tight':<7} {'VolOK':>5} {'RS':>5} {'TP':>7} {'SL':>7} {'R:R':>5} {'Score':>6}")
    print("-" * 90)
    for _, r in df_res.head(25).iterrows():
        print(f"{r['symbol']:<6} {r['price']:>7.2f} {r['level']:<10} {r['dist_%']:>+6.1f}% {r['consol_atr']:>6.1f} {r['tight']:<7} {r['vol_ok']:>5} {r['RS']:>5.2f} {r['TP']:>7.2f} {r['SL']:>7.2f} {r['R:R']:>5.1f} {r['score']:>6}")
    print("=" * 90)
    print("\nLegend: Dist% = насколько далеко от уровня | Consol = ширина проторговки в ATR (< 1.8 = TIGHT) | Score = суммарный балл")

if __name__ == '__main__':
    main()
