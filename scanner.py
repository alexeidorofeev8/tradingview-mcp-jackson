"""
Sniper Pullback Scanner — Type B
Ищет акции где ВЧЕРА тень нырнула ниже уровня на 0-1.5%, СЕГОДНЯ закрылись выше.
Запускать вечером после закрытия рынка.
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_START = '2023-01-01'
SMA_FAST, SMA_SLOW   = 50, 200
ATR_LEN              = 14
RS_LEN               = 63
VOL_LEN              = 20
SR_LOOKBACK_MIN      = 20
SR_LOOKBACK_MAX      = 80
B_MAX_BREACH_PCT     = 1.5   # макс пробой ниже уровня (%)
B_RECOVERY_BODY      = 0.5   # тело свечи восстановления в ATR
B_RECOVERY_VOL       = 1.2   # объём на восстановлении
B_STOP_BUFFER        = 0.5   # ATR буфер ниже минимума пробоя
EXCLUDE_SECTORS      = {'Real Estate', 'Utilities'}
EARNINGS_CACHE       = 'd:/projects/trading/earnings_cache.json'
EARNINGS_BUFFER      = 7


def get_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=15)
    df = pd.read_html(io.StringIO(resp.text))[0]
    filtered = df[~df['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols = filtered['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map = dict(zip(
        filtered['Symbol'].str.replace('.', '-', regex=False),
        filtered['GICS Sector']
    ))
    return symbols, sector_map


def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def load_earnings():
    if os.path.exists(EARNINGS_CACHE):
        raw = json.load(open(EARNINGS_CACHE, encoding='utf-8'))
        return {s: set(v) for s, v in raw.items()}
    return {}


def scan_symbol(symbol, df, spx_close, earnings, sector=None):
    try:
        if len(df) < 220:
            return None

        close  = df['Close'].squeeze()
        high   = df['High'].squeeze()
        low    = df['Low'].squeeze()
        open_  = df['Open'].squeeze()
        volume = df['Volume'].squeeze()

        atr    = calc_atr(high, low, close, ATR_LEN)
        sma50  = close.rolling(SMA_FAST).mean()
        sma200 = close.rolling(SMA_SLOW).mean()
        vol_ma = volume.rolling(VOL_LEN).mean()

        # Последние значения (сегодня = -1, вчера = -2)
        c      = close.iloc[-1]
        o      = open_.iloc[-1]
        h      = high.iloc[-1]
        l      = low.iloc[-1]
        v      = volume.iloc[-1]
        a      = atr.iloc[-1]
        s50    = sma50.iloc[-1]
        s200   = sma200.iloc[-1]
        vm     = vol_ma.iloc[-1]
        yday_l = low.iloc[-2]   # вчерашний минимум (день пробоя)
        yday_c = close.iloc[-2]

        # ── Базовые фильтры ──────────────────────────────────────────
        if c < 20:        return None
        if vm < 500_000:  return None
        if a/c*100 > 5.0: return None
        if c < s200:      return None
        if s50 < s200:    return None

        # SPY > SMA200
        spx = spx_close.reindex(close.index, method='ffill')
        if spx.iloc[-1] < spx.rolling(SMA_SLOW).mean().iloc[-1]:
            return None

        # RS > 1.0
        rs = (close / close.shift(RS_LEN)) / (spx / spx.shift(RS_LEN))
        if rs.iloc[-1] < 1.0:
            return None

        # Отчётность
        today = close.index[-1].date()
        earn_set = earnings.get(symbol, set())
        if any(0 <= (pd.to_datetime(ed).date() - today).days <= EARNINGS_BUFFER for ed in earn_set):
            return None

        # ── Уровни ───────────────────────────────────────────────────
        prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        local_min  = low.rolling(15).min().shift(1)

        ph = prior_high.iloc[-1]
        lm = local_min.iloc[-1]

        if pd.isna(ph) and pd.isna(lm):
            return None

        # ── Type B условие ───────────────────────────────────────────
        # Вчера: тень пробила уровень на 0-1.5%, тело осталось выше (или хотя бы low нырнул)
        # Сегодня: закрылись ВЫШЕ уровня, сильная зелёная свеча, объём

        level_type = None
        level_val  = None

        # Проверяем prior_high
        if not pd.isna(ph):
            broke_ph = (yday_l < ph) and (yday_l >= ph * (1 - B_MAX_BREACH_PCT / 100))
            recovered_ph = (c > ph) and (yday_l < ph)
            if broke_ph and recovered_ph:
                level_type = 'S/R flip'
                level_val  = ph

        # Проверяем local_min (только если prior_high не сработал)
        if level_type is None and not pd.isna(lm):
            broke_lm = (yday_l < lm) and (yday_l >= lm * (1 - B_MAX_BREACH_PCT / 100))
            recovered_lm = (c > lm) and (yday_l < lm)
            if broke_lm and recovered_lm:
                level_type = 'Local min'
                level_val  = lm

        if level_type is None:
            return None

        # ── Свеча восстановления ─────────────────────────────────────
        green       = c > o
        body_ok     = (c - o) / a > B_RECOVERY_BODY
        vol_ok      = v > vm * B_RECOVERY_VOL

        if not green or not body_ok:
            return None

        # ── Стоп и цель ──────────────────────────────────────────────
        breakdown_low = min(yday_l, low.iloc[-3]) if len(low) > 2 else yday_l
        sl = breakdown_low - a * B_STOP_BUFFER

        entry = c * 0.99  # приблизительный вход (интрадей завтра)
        risk  = entry - sl
        if risk <= 0 or risk / entry > 0.15:
            return None

        # Ближайшее сопротивление
        tp_candidates = []
        for lb in [10, 20, 40, 60]:
            res = high.iloc[max(0, len(high)-lb):-1].max()
            if res > entry * 1.01:
                tp_candidates.append(res)
        tp = min(tp_candidates) if tp_candidates else entry + risk * 3.0
        rr = (tp - entry) / risk

        if rr < 1.0:
            return None

        breach_pct = (level_val - yday_l) / level_val * 100

        return {
            'symbol':    symbol,
            'price':     round(c, 2),
            'level':     level_type,
            'level_$':   round(level_val, 2),
            'breach%':   round(breach_pct, 2),
            'yday_low':  round(yday_l, 2),
            'vol_ok':    'Y' if vol_ok else 'n',
            'RS':        round(rs.iloc[-1], 2),
            'entry~':    round(entry, 2),
            'SL':        round(sl, 2),
            'TP':        round(tp, 2),
            'R:R':       round(rr, 1),
            'risk%':     round(risk / entry * 100, 1),
        }

    except Exception:
        return None


def main():
    print("=== Type B Scanner — ложный пробой + восстановление ===\n")

    symbols, sector_map = get_sp500()
    earnings = load_earnings()
    print(f"Акций в S&P500: {len(symbols)}")

    spx_raw   = yf.download('SPY', start=DATA_START, progress=False, auto_adjust=True)
    spx_close = spx_raw['Close'].squeeze()

    # Проверка рынка
    if spx_close.iloc[-1] < spx_close.rolling(SMA_SLOW).mean().iloc[-1]:
        print("⚠️  SPY ниже SMA200 — рынок не в восходящем тренде. Сигналов нет.")
        return

    print("Загружаем котировки...\n")
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
        print(f"  {min(i+50, len(symbols))}/{len(symbols)}...", end=' ', flush=True)
    print()

    print("\nСканируем...\n")
    results = []
    for sym, df in all_data.items():
        r = scan_symbol(sym, df, spx_close, earnings, sector_map.get(sym))
        if r:
            results.append(r)

    if not results:
        print("Сегодня сетапов Type B нет.")
        return

    df_res = pd.DataFrame(results).sort_values('R:R', ascending=False)

    print(f"Найдено сетапов Type B: {len(df_res)}\n")
    print("=" * 100)
    print(f"{'Sym':<6} {'Price':>7} {'Уровень':<10} {'Level$':>8} {'Пробой%':>8} {'YdayLow':>8} {'Vol':>4} {'RS':>5} {'Вход~':>7} {'SL':>7} {'TP':>7} {'R:R':>5} {'Risk%':>6}")
    print("-" * 100)
    for _, r in df_res.iterrows():
        print(f"{r['symbol']:<6} {r['price']:>7.2f} {r['level']:<10} {r['level_$']:>8.2f} {r['breach%']:>7.2f}% {r['yday_low']:>8.2f} {r['vol_ok']:>4} {r['RS']:>5.2f} {r['entry~']:>7.2f} {r['SL']:>7.2f} {r['TP']:>7.2f} {r['R:R']:>5.1f} {r['risk%']:>5.1f}%")
    print("=" * 100)
    print("\nПодсказка: Vol=Y означает объём >1.2x среднего. Вход~ = приблизительно (интрадей).")
    print("Завтра в 15:30 Берлин открой часовик этих акций и входи когда цена держится выше уровня.")


if __name__ == '__main__':
    main()
