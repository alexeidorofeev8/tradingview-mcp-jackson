"""
Sniper Pullback v6 — правильное скользящее обучение
Каждые 2 года учимся на прошлом, торгуем следующий период.
Без заглядывания в будущее.

Периоды:
  Обучение 2020–2021 → тест 2022
  Обучение 2021–2022 → тест 2023
  Обучение 2022–2023 → тест 2024
  Обучение 2023–2024 → тест 2025–2026

Плюс все улучшения v5:
  1. Фильтр отчётности (7 дней до отчёта)
  2. Фильтр секторов (топ-5 по силе 63 дня)
  3. Часовой вход (-1% к цене)
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_START = '2018-01-01'

START_BALANCE   = 4000.0
RISK_PCT        = 0.01
COMMISSION      = 2.0
MAX_POSITIONS   = 4
TRAIN_WIN_MIN   = 0.40   # минимальный win rate в обучении
TRAIN_SIG_MIN   = 2      # минимум сигналов в обучении

EARNINGS_BUFFER = 7
TOP_SECTORS     = 5
HOURLY_DISC     = 0.01

SMA_FAST, SMA_SLOW  = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
SR_LOOKBACK_MIN, SR_LOOKBACK_MAX = 20, 80
SR_TOUCH_PCT    = 2.5
CONSOL_BARS     = 5;  CONSOL_RANGE = 1.8;  CONSOL_VOL_MIN = 0.8
ENTRY_BODY_MIN  = 0.25;  ENTRY_VOL_MIN = 1.1
MAX_ATR_PCT     = 5.0;  MIN_PRICE = 20;  MIN_VOL = 500_000
EXCLUDE_SECTORS = {'Real Estate', 'Utilities'}

# Скользящие окна: (обучение_от, обучение_до, тест_от, тест_до)
WINDOWS = [
    ('2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2023-01-01', '2024-12-31', '2025-01-01', '2026-04-11'),
]

SECTOR_ETF = {
    'Information Technology': 'XLK',
    'Financials':             'XLF',
    'Energy':                 'XLE',
    'Health Care':            'XLV',
    'Industrials':            'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples':       'XLP',
    'Materials':              'XLB',
    'Communication Services': 'XLC',
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
    print("  Кэш отчётностей не найден — запусти сначала backtest_v5.py")
    return {}

def build_sector_leaders(spx_close):
    print("  Скачиваем секторные ETF...")
    etfs = list(SECTOR_ETF.values())
    raw = yf.download(etfs, start=DATA_START, end='2027-01-01',
                      progress=False, auto_adjust=True, group_by='ticker')
    sec_prices = {}
    for sector, etf in SECTOR_ETF.items():
        try: sec_prices[sector] = raw[etf]['Close'].squeeze()
        except: pass

    print("  Вычисляем лидеров секторов...")
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

def get_signals(sym, df_stock, spx_close, sig_start, sig_end, earnings, sec_leaders, stock_sec):
    """Возвращает все сигналы для акции в заданном периоде"""
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
        mask      = (sig_mask.index >= sig_start) & (sig_mask.index <= sig_end)
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

def main():
    print("=== Sniper Pullback v6 — ПРАВИЛЬНОЕ СКОЛЬЗЯЩЕЕ ОБУЧЕНИЕ ===\n")

    symbols, sector_map = get_sp500()
    print(f"Акций: {len(symbols)}\n")

    earnings = load_earnings(symbols)

    print("Загружаем SPY + секторы...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01', progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()
    sec_leaders = build_sector_leaders(spx_close)

    print(f"\nЗагружаем {len(symbols)} акций...")
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

    # ── Скользящее обучение ──────────────────────────────────
    all_test_trades = []
    window_stats = []

    print("\n" + "="*70)
    print("  СКОЛЬЗЯЩЕЕ ОБУЧЕНИЕ")
    print("="*70)

    for train_s, train_e, test_s, test_e in WINDOWS:
        print(f"\n  Обучение {train_s[:4]}–{train_e[:4]} → тест {test_s[:4]}–{test_e[:4]}")

        # Обучение: находим какие акции работали
        train_trades = {}
        for sym, df in all_data.items():
            trades = get_signals(sym, df.copy(), spx_close,
                                train_s, train_e, earnings, sec_leaders, sector_map.get(sym))
            if trades:
                train_trades[sym] = trades

        # Квалификация: выбираем только тех кто прошёл порог
        qualified = []
        for sym, trades in train_trades.items():
            if len(trades) >= TRAIN_SIG_MIN:
                wr = sum(t['win'] for t in trades) / len(trades)
                if wr >= TRAIN_WIN_MIN:
                    qualified.append(sym)

        print(f"    Квалифицировано: {len(qualified)} из {len(all_data)} акций", flush=True)

        # Тест: только на квалифицированных
        test_trades = []
        for sym in qualified:
            if sym not in all_data: continue
            trades = get_signals(sym, all_data[sym].copy(), spx_close,
                                test_s, test_e, earnings, sec_leaders, sector_map.get(sym))
            test_trades.extend(trades)

        if test_trades:
            df_t = pd.DataFrame(test_trades)
            wr = df_t['win'].mean() * 100
            avg = df_t['pnl_pct'].mean()
            print(f"    Сделок в тесте: {len(test_trades)}  Win: {wr:.1f}%  Avg P&L: {avg:+.2f}%")
            window_stats.append((test_s[:4], test_e[:4], len(test_trades), wr, avg, len(qualified)))
        else:
            print(f"    Сделок в тесте: 0")

        all_test_trades.extend(test_trades)

    if not all_test_trades:
        print("Нет сделок."); return

    df = pd.DataFrame(all_test_trades)
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # ── Таблица 1: По годам ───────────────────────────────────
    V5 = {2022:(22,45.5,+0.28), 2023:(121,39.7,+0.75),
          2024:(170,58.2,+2.25), 2025:(113,54.0,+2.03), 2026:(38,63.2,+3.49)}

    print("\n" + "="*78)
    print("  РЕЗУЛЬТАТЫ v6 vs v5")
    print("="*78)
    print(f"  {'Год':<5}  {'v6 Сд':>6} {'v6 Win%':>8} {'v6 Avg':>8}   │  {'v5 Win%':>8} {'v5 Avg':>8}  {'Дельта':>7}")
    print("-"*78)
    for yr, g in df.groupby('year'):
        wr  = g['win'].mean()*100
        avg = g['pnl_pct'].mean()
        v5  = V5.get(yr, (0, 0, 0))
        d   = avg - v5[2]
        print(f"  {yr:<5}  {len(g):>6}  {wr:>6.1f}%   {avg:>+7.2f}%   │  {v5[1]:>6.1f}%   {v5[2]:>+7.2f}%  {d:>+6.2f}%{'↑' if d>0 else '↓'}")
    print("-"*78)
    print(f"  {'ВСЕ':<5}  {len(df):>6}  {df['win'].mean()*100:>6.1f}%   {df['pnl_pct'].mean():>+7.2f}%   │  {'52.5':>6}%   {'+1.71':>7}%")

    # ── Таблица 2: Сделок по месяцам ─────────────────────────
    months = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']
    pivot = df.groupby(['year','month']).size().unstack(fill_value=0)
    for m in range(1,13):
        if m not in pivot.columns: pivot[m] = 0

    print("\n" + "="*78)
    print("  СДЕЛОК В МЕСЯЦ")
    print("="*78)
    print(f"  {'Год':<5}" + "".join(f"{months[m-1]:>5}" for m in range(1,13)) + f"{'Итого':>7}")
    print("-"*78)
    for yr, row in pivot.iterrows():
        vals = [int(row.get(m,0)) for m in range(1,13)]
        print(f"  {yr:<5}" + "".join(f"{v:>5}" for v in vals) + f"{sum(vals):>7}")

    # ── Таблица 3: Рост счёта ─────────────────────────────────
    print("\n" + "="*78)
    print(f"  РОСТ СЧЁТА: ${START_BALANCE:,.0f}, риск 1%, комиссия ${COMMISSION}")
    print("="*78)
    sim_df, final = simulate_account(df)
    sim_df['year'] = pd.to_datetime(sim_df['date']).dt.year

    V5_ACC = {2022:6253, 2023:7303, 2024:9711, 2025:11551, 2026:12174}
    account = START_BALANCE
    print(f"\n  {'Год':<5} {'Сделок':>7} {'Win%':>6} {'P&L $':>9} {'P&L %':>8} {'Счёт v6':>10}  {'Счёт v5':>10}")
    print("-"*78)
    for yr, g in sim_df.groupby('year'):
        start_bal = account
        end_bal   = g['account'].iloc[-1]
        account   = end_bal
        pct       = (end_bal - start_bal) / start_bal * 100
        wr        = g['win'].mean()*100
        pnl       = g['pnl_$'].sum()
        v5a       = V5_ACC.get(yr, 0)
        diff      = end_bal - v5a
        print(f"  {yr:<5} {len(g):>7}  {wr:>5.1f}%  ${pnl:>+8,.0f}  {pct:>+7.1f}%  ${end_bal:>8,.0f}  ${v5a:>8,.0f} {'↑' if diff>0 else '↓'}{abs(diff):,.0f}")

    total_pct = (final - START_BALANCE) / START_BALANCE * 100
    cagr = (final / START_BALANCE) ** (1/4.3) - 1  # 2022-2026 = ~4.3 года
    print("-"*78)
    print(f"\n  $4,000 → ${final:,.0f}  |  {final/START_BALANCE:.1f}x  |  CAGR {cagr*100:.1f}% в год (2022–2026)")
    print(f"  v5 было (2022–2026): $5,409 → $12,174")

    df.to_csv('d:/projects/trading/signals_v6.csv', index=False)
    print(f"\n  Сохранено: signals_v6.csv")

if __name__ == '__main__':
    main()
