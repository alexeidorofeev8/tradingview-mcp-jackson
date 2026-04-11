"""
Тест часового фильтра консолидации — 2024–2026
Добавляем к v5: последние 6 часовых баров перед сигналом должны быть < 1.5 ATR
Часовые данные доступны только за последние ~2 года, поэтому тест 2024–2026.
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_START  = '2022-01-01'
TEST_START  = '2024-01-01'
TEST_END    = '2026-04-11'

START_BALANCE   = 4000.0
RISK_PCT        = 0.01
COMMISSION      = 2.0
MAX_POSITIONS   = 4

# v5 параметры
EARNINGS_BUFFER = 7
TOP_SECTORS     = 5
HOURLY_DISC     = 0.01

# Новый часовой фильтр
H_CONSOL_BARS  = 6    # последних 6 часовых баров
H_CONSOL_MAX   = 1.5  # диапазон < 1.5 ATR часового

SMA_FAST, SMA_SLOW  = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
SR_LOOKBACK_MIN, SR_LOOKBACK_MAX = 20, 80
SR_TOUCH_PCT    = 2.5
CONSOL_BARS     = 5;  CONSOL_RANGE = 1.8;  CONSOL_VOL_MIN = 0.8
ENTRY_BODY_MIN  = 0.25;  ENTRY_VOL_MIN = 1.1
MAX_ATR_PCT     = 5.0;  MIN_PRICE = 20;  MIN_VOL = 500_000
EXCLUDE_SECTORS = {'Real Estate', 'Utilities'}

SECTOR_ETF = {
    'Information Technology': 'XLK', 'Financials': 'XLF',
    'Energy': 'XLE',  'Health Care': 'XLV', 'Industrials': 'XLI',
    'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
    'Materials': 'XLB', 'Communication Services': 'XLC',
}
EARNINGS_CACHE = 'd:/projects/trading/earnings_cache.json'

def get_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers, timeout=15)
    df = pd.read_html(io.StringIO(resp.text))[0]
    filtered = df[~df['GICS Sector'].isin(EXCLUDE_SECTORS)]
    syms = filtered['Symbol'].str.replace('.', '-', regex=False).tolist()
    sec_map = dict(zip(
        filtered['Symbol'].str.replace('.', '-', regex=False),
        filtered['GICS Sector']
    ))
    return syms, sec_map

def calc_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
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
        try: sec_prices[sector] = raw[etf]['Close'].squeeze()
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

def check_hourly_consol(sym, sig_date, hourly_data):
    """Проверяет узкую консолидацию на часовике перед дневным сигналом"""
    try:
        h = hourly_data.get(sym)
        if h is None or len(h) < 20:
            return None  # нет данных — не фильтруем

        close_h = h['Close'].squeeze()
        high_h  = h['High'].squeeze()
        low_h   = h['Low'].squeeze()

        if close_h.index.tz is not None:
            close_h.index = close_h.index.tz_localize(None)
            high_h.index  = high_h.index.tz_localize(None)
            low_h.index   = low_h.index.tz_localize(None)

        atr_h = calc_atr(high_h, low_h, close_h, 14)

        # Берём часовые бары ПЕРЕД дневным сигналом (до 22:00 того дня)
        sig_dt = pd.to_datetime(sig_date)
        cutoff = sig_dt.replace(hour=21, minute=59)
        before = close_h[close_h.index <= cutoff]

        if len(before) < H_CONSOL_BARS:
            return None

        last_bars_high = high_h[high_h.index <= cutoff].tail(H_CONSOL_BARS)
        last_bars_low  = low_h[low_h.index <= cutoff].tail(H_CONSOL_BARS)
        last_atr       = atr_h[atr_h.index <= cutoff].iloc[-1] if len(atr_h[atr_h.index <= cutoff]) > 0 else None

        if last_atr is None or last_atr <= 0:
            return None

        consol_range = (last_bars_high.max() - last_bars_low.min()) / last_atr
        return float(consol_range)
    except:
        return None

def run_symbol(sym, df_stock, spx_close, earnings, sec_leaders, stock_sec,
               hourly_data, use_hourly_filter):
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
        rs     = (close/close.shift(RS_LEN)) / (spx/spx.shift(RS_LEN))

        prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        at_level = (
            ((close - prior_high).abs() / prior_high * 100 < SR_TOUCH_PCT) |
            ((close - low.rolling(15).min()).abs() / low.rolling(15).min() * 100 < 2.0) |
            ((close - sma50).abs() / sma50 * 100 < 3.0)
        )
        absorption = (
            ((high.shift(1).rolling(CONSOL_BARS).max() - low.shift(1).rolling(CONSOL_BARS).min()) / atr < CONSOL_RANGE) &
            (volume.shift(1).rolling(CONSOL_BARS).mean() > vol_ma * CONSOL_VOL_MIN) &
            (close.pct_change(CONSOL_BARS).shift(1) >= spx.pct_change(CONSOL_BARS).shift(1) - 0.01)
        )
        strong_green = (
            (close > open_) &
            ((close - open_) / atr > ENTRY_BODY_MIN) &
            (volume > vol_ma * ENTRY_VOL_MIN)
        )
        trend_ok   = (close > sma200) & (sma50 > sma200)
        quality_ok = ((atr/close*100) < MAX_ATR_PCT) & (rs > 1.0) & (vol_ma > MIN_VOL) & (close > MIN_PRICE)
        market_ok  = spx > spx.rolling(200).mean()

        sig_mask  = trend_ok & quality_ok & at_level & absorption & strong_green & market_ok
        mask      = (sig_mask.index >= TEST_START) & (sig_mask.index <= TEST_END)
        sig_dates = sig_mask[mask & sig_mask].index

        results = []
        bars = list(close.index)
        earn_set = earnings.get(sym, set())

        for sig_date in sig_dates:
            # Фильтр отчётности
            sig_d = sig_date.date()
            if any(0 <= (pd.to_datetime(ed).date() - sig_d).days <= EARNINGS_BUFFER for ed in earn_set):
                continue
            # Фильтр секторов
            if stock_sec and sig_date in sec_leaders:
                if stock_sec not in sec_leaders[sig_date]:
                    continue
            # Часовой фильтр консолидации (только в режиме with_hourly)
            if use_hourly_filter:
                consol = check_hourly_consol(sym, sig_date, hourly_data)
                if consol is not None and consol >= H_CONSOL_MAX:
                    continue  # слишком широко — пропускаем

            idx_i = bars.index(sig_date)
            entry_price = close.iloc[idx_i] * (1 - HOURLY_DISC)
            sl_init     = sma50.iloc[idx_i] - atr.iloc[idx_i] * 1.5
            risk        = entry_price - sl_init
            risk_pct    = risk / entry_price * 100
            if risk <= 0 or risk_pct > 15: continue

            res_cands = [high.iloc[max(0,idx_i-lb):idx_i].max()
                         for lb in [10,20,40,60]
                         if high.iloc[max(0,idx_i-lb):idx_i].max() > entry_price * 1.01]
            tp = entry_price + risk * (max(1.5, min(5.0, (min(res_cands) - entry_price) / risk)) if res_cands else 3.0)

            exit_p, reason, hold = None, 'timeout', 0
            for j in range(1, 21):
                if idx_i + j >= len(bars): break
                bh = high.iloc[idx_i+j]; bl = low.iloc[idx_i+j]
                bc = close.iloc[idx_i+j]; bs = sma50.iloc[idx_i+j]; ba = atr.iloc[idx_i+j]
                hold = j
                if bl <= sl_init: exit_p = sl_init; reason = 'stop';      break
                if bh >= tp:      exit_p = tp;      reason = 'target';    break
                if bc < bs-ba*0.5: exit_p = bc;    reason = 'below_sma'; break
            if exit_p is None:
                exit_p = close.iloc[min(idx_i+hold, len(close)-1)]; reason = 'timeout'

            results.append({
                'symbol': sym, 'date': sig_date,
                'entry': entry_price, 'exit': exit_p,
                'sl': sl_init, 'tp': tp,
                'risk_pct': risk_pct,
                'pnl_pct': (exit_p - entry_price) / entry_price * 100,
                'win': exit_p >= entry_price,
                'reason': reason, 'days': hold,
            })
        return results
    except: return []

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
        risk_usd = account * RISK_PCT
        pos      = risk_usd / (t['risk_pct'] / 100)
        pnl_usd  = pos * (t['pnl_pct'] / 100) - COMMISSION
        account += pnl_usd
        rows.append({'date': t['date'], 'pnl_$': round(pnl_usd, 2),
                     'account': round(account, 2), 'symbol': t['symbol'], 'win': t['win']})
    return pd.DataFrame(rows), account

def print_results(label, trades, account_start=START_BALANCE):
    if not trades: print(f"  {label}: нет сделок"); return None, account_start
    df = pd.DataFrame(trades)
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    sim, final = simulate_account(df)
    sim['year'] = pd.to_datetime(sim['date']).dt.year

    print(f"\n  {'Год':<5} {'Сд':>5} {'Win%':>6} {'Avg P&L':>8} {'P&L $':>8} {'Счёт':>10}")
    print(f"  {'-'*50}")
    acc = account_start
    for yr, g in sim.groupby('year'):
        start = acc
        end   = g['account'].iloc[-1]; acc = end
        pct   = (end - start) / start * 100
        wr    = g['win'].mean()*100
        pnl   = g['pnl_$'].sum()
        # avg pnl из оригинальных сделок
        avg   = df[df['year']==yr]['pnl_pct'].mean()
        print(f"  {yr:<5} {len(g):>5}  {wr:>5.1f}%  {avg:>+7.2f}%  ${pnl:>+7,.0f}  ${end:>8,.0f}")
    total = (final - account_start) / account_start * 100
    print(f"  {'─'*50}")
    print(f"  Итого: {len(df)} сд  Win {df['win'].mean()*100:.1f}%  Avg {df['pnl_pct'].mean():+.2f}%")
    print(f"  ${account_start:,.0f} → ${final:,.0f}  ({total:+.1f}%)")
    return df, final

def main():
    print("=== Тест часового фильтра консолидации (2024–2026) ===\n")

    symbols, sector_map = get_sp500()
    earnings = load_earnings(symbols)

    print("Загружаем SPY + секторы...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01',
                           progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()
    sec_leaders = build_sector_leaders(spx_close)

    print(f"Загружаем дневные данные {len(symbols)} акций...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw = yf.download(batch, start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True, group_by='ticker')
        for sym in batch:
            try: all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except: pass
        print(f"  [{min(i+50,len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    # ── Фаза 1: v5 без часового фильтра (база) ───────────────
    print("\nФаза 1: v5 базовый (без часового фильтра)...")
    base_trades = []
    for sym, df in all_data.items():
        trades = run_symbol(sym, df.copy(), spx_close,
                           earnings, sec_leaders, sector_map.get(sym),
                           {}, False)
        base_trades.extend(trades)

    # ── Фаза 2: Скачиваем часовые данные ─────────────────────
    print(f"\nФаза 2: Скачиваем часовые данные...")
    # Определяем какие акции нам нужны (у которых были сигналы)
    if base_trades:
        active_syms = list(set(t['symbol'] for t in base_trades))
    else:
        active_syms = symbols[:100]

    hourly_data = {}
    print(f"  Загружаем часовые для {len(active_syms)} акций...")
    for i in range(0, len(active_syms), 20):
        batch = active_syms[i:i+20]
        try:
            raw_h = yf.download(batch, start='2024-05-01', end='2027-01-01',
                                interval='1h', progress=False,
                                auto_adjust=True, group_by='ticker')
            for sym in batch:
                try:
                    h = raw_h[sym].dropna() if len(batch) > 1 else raw_h.dropna()
                    if len(h) > 50:
                        hourly_data[sym] = h
                except: pass
        except: pass
        print(f"  [{min(i+20,len(active_syms))}/{len(active_syms)}]", end=' ', flush=True)
    print(f"\n  Часовые данные загружены для {len(hourly_data)} акций")

    # ── Фаза 3: v5 + часовой фильтр ──────────────────────────
    print(f"\nФаза 3: v5 + часовой фильтр консолидации ({H_CONSOL_BARS} баров < {H_CONSOL_MAX} ATR)...")
    hourly_trades = []
    for sym, df in all_data.items():
        trades = run_symbol(sym, df.copy(), spx_close,
                           earnings, sec_leaders, sector_map.get(sym),
                           hourly_data, True)
        hourly_trades.extend(trades)

    # ── Сравнение ─────────────────────────────────────────────
    months_ru = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']

    print(f"\n{'='*60}")
    print(f"  РЕЗУЛЬТАТЫ: v5 базовый (2024–2026)")
    print(f"{'='*60}")
    df_base, final_base = print_results("v5 базовый", base_trades)

    print(f"\n{'='*60}")
    print(f"  РЕЗУЛЬТАТЫ: v5 + часовой фильтр (2024–2026)")
    print(f"{'='*60}")
    df_hour, final_hour = print_results("v5 + часовой", hourly_trades)

    # ── Итоговое сравнение ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  СРАВНЕНИЕ ИТОГОВ")
    print(f"{'='*60}")
    if df_base is not None and df_hour is not None:
        b_wr  = df_base['win'].mean()*100
        h_wr  = df_hour['win'].mean()*100
        b_avg = df_base['pnl_pct'].mean()
        h_avg = df_hour['pnl_pct'].mean()
        b_n   = len(df_base)
        h_n   = len(df_hour)
        filtered = b_n - h_n

        print(f"\n  {'Метрика':<25} {'v5 база':>10} {'v5+часовик':>12} {'Разница':>10}")
        print(f"  {'-'*60}")
        print(f"  {'Сделок':<25} {b_n:>10} {h_n:>12} {h_n-b_n:>+10}")
        print(f"  {'Win rate':<25} {b_wr:>9.1f}% {h_wr:>11.1f}% {h_wr-b_wr:>+9.1f}%")
        print(f"  {'Avg P&L на сделку':<25} {b_avg:>+9.2f}% {h_avg:>+11.2f}% {h_avg-b_avg:>+9.2f}%")
        print(f"  {'Итог счёта':<25} ${final_base:>9,.0f} ${final_hour:>11,.0f} ${final_hour-final_base:>+9,.0f}")
        print(f"\n  Отфильтровано часовым фильтром: {filtered} сделок ({filtered/b_n*100:.0f}%)")

        # Какие сделки отфильтровал часовик?
        if df_base is not None and df_hour is not None:
            base_keys = set(zip(df_base['symbol'], df_base['date'].astype(str)))
            hour_keys = set(zip(df_hour['symbol'], df_hour['date'].astype(str)))
            filtered_trades = df_base[
                ~df_base.apply(lambda r: (r['symbol'], str(r['date'])) in hour_keys, axis=1)
            ]
            if len(filtered_trades) > 0:
                wr_filtered = filtered_trades['win'].mean()*100
                avg_filtered = filtered_trades['pnl_pct'].mean()
                print(f"\n  Качество отфильтрованных сделок:")
                print(f"  Win rate:  {wr_filtered:.1f}%  (у оставшихся: {h_wr:.1f}%)")
                print(f"  Avg P&L:   {avg_filtered:+.2f}%  (у оставшихся: {h_avg:+.2f}%)")

    # Сохраняем
    if hourly_trades:
        pd.DataFrame(hourly_trades).to_csv('d:/projects/trading/signals_hourly_filter.csv', index=False)
        print(f"\n  Сохранено: signals_hourly_filter.csv")

if __name__ == '__main__':
    main()
