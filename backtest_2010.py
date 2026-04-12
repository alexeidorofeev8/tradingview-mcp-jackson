"""
Бэктест Type B 2010–2026
Сравнение: с фильтром топ-5 секторов и без, 4 слота и без лимита
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_START  = '2008-01-01'
TEST_START  = '2010-01-01'
TEST_END    = '2026-04-11'

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

        sig_mask = trend_ok & quality_ok & false_break & recovery & strong_recovery & market_ok
        date_mask = (close.index >= TEST_START) & (close.index <= TEST_END)

        results = []
        bars = list(close.index)
        earn_set = earnings.get(sym, set())

        for sig_date in sig_mask[date_mask & sig_mask].index:
            sig_d = sig_date.date()
            if any(0 <= (pd.to_datetime(ed).date() - sig_d).days <= EARNINGS_BUFFER for ed in earn_set):
                continue

            idx_i = bars.index(sig_date)
            entry_price = close.iloc[idx_i] * (1 - HOURLY_DISC)

            lookback = min(B_BREAKDOWN_BARS + 1, idx_i)
            recent_lows = low.iloc[idx_i - lookback: idx_i]
            breakdown_low = recent_lows.min() if len(recent_lows) > 0 else low.iloc[idx_i]
            sl_init = breakdown_low - atr.iloc[idx_i] * B_STOP_BUFFER

            risk = entry_price - sl_init
            risk_pct = risk / entry_price * 100
            if risk <= 0 or risk_pct > 15:
                continue

            res_cands = [
                high.iloc[max(0, idx_i - lb):idx_i].max()
                for lb in [10, 20, 40, 60]
                if high.iloc[max(0, idx_i - lb):idx_i].max() > entry_price * 1.01
            ]
            tp = min(res_cands) + 0 if res_cands else entry_price + risk * 3.0
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
                'symbol':   sym,
                'sector':   stock_sec or '',
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
                # Сектор-фильтр: был ли в топ-5 на эту дату
                'top_sector': (
                    bool(stock_sec and sig_date in sec_leaders and stock_sec in sec_leaders[sig_date])
                ),
            })

        return results
    except Exception:
        return []


def simulate_account(trades_df, max_positions=4):
    df = trades_df.sort_values('date').copy()
    account  = START_BALANCE
    open_pos = []
    rows = []
    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)
        open_pos = [d for d in open_pos if d > e_date]
        if len(open_pos) >= max_positions:
            continue
        open_pos.append(x_date)
        pos     = (account * RISK_PCT) / (t['risk_pct'] / 100)
        pnl_usd = pos * (t['pnl_pct'] / 100) - COMMISSION
        account += pnl_usd
        rows.append({'date': t['date'], 'pnl_$': pnl_usd, 'account': account,
                     'symbol': t['symbol'], 'win': t['win']})
    return pd.DataFrame(rows), account


def print_result(label, trades_df, max_positions):
    sim, final = simulate_account(trades_df, max_positions)
    years = (pd.Timestamp(TEST_END) - pd.Timestamp(TEST_START)).days / 365.25
    cagr  = (final / START_BALANCE) ** (1 / years) - 1
    wr    = trades_df['win'].mean() * 100
    slot_label = 'без лимита' if max_positions >= 999 else f'{max_positions} слота'
    print(f"  {label:<35} {slot_label:<12} {len(sim):>5} сделок  "
          f"WR {wr:.0f}%  ${final:>10,.0f}  CAGR {cagr*100:.1f}%")


def main():
    print("=== Бэктест Type B: 2010–2026 ===\n")

    print("Загружаем S&P500...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers, timeout=15)
    wiki = pd.read_html(io.StringIO(resp.text))[0]
    wiki = wiki[~wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols   = wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map = dict(zip(wiki['Symbol'].str.replace('.', '-', regex=False), wiki['GICS Sector']))
    print(f"  Акций: {len(symbols)}")

    print("Загружаем SPY + секторные ETF...")
    spx_raw   = yf.download('SPY', start=DATA_START, progress=False, auto_adjust=True)
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
        raw = yf.download(batch, start=DATA_START, progress=False,
                          auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except:
                pass
        print(f"  [{min(i+50,len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    print("\nСканируем сигналы...")
    all_trades = []
    for sym, df in all_data.items():
        sec = sector_map.get(sym)
        trades = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec)
        all_trades.extend(trades)

    df_all = pd.DataFrame(all_trades)
    df_all['date'] = pd.to_datetime(df_all['date'])
    print(f"  Всего сигналов: {len(df_all)}")

    # Два набора: все сигналы и только топ-5 секторов
    df_top = df_all[df_all['top_sector']].copy()
    print(f"  Из них в топ-5 секторах: {len(df_top)}")

    # Сохраняем
    df_all.to_csv('d:/projects/trading/signals_type_b_2010.csv', index=False)
    print(f"  Сохранено: signals_type_b_2010.csv")

    years = (pd.Timestamp(TEST_END) - pd.Timestamp(TEST_START)).days / 365.25

    print(f"\n{'='*85}")
    print(f"  СРАВНЕНИЕ: все сектора vs топ-5, $10k стартовый баланс, 2010–2026")
    print(f"{'='*85}")
    print(f"  {'Вариант':<35} {'Слоты':<12} {'Сделок':>6}  {'WR':>5}  {'Итог':>11}  {'CAGR':>7}")
    print(f"  {'-'*83}")

    for label, df_use in [('Все сектора', df_all), ('Топ-5 секторов', df_top)]:
        for max_pos in [4, 999]:
            print_result(label, df_use, max_pos)
        print()

    print(f"{'='*85}")
    print(f"  Без лимита = гипотетически (требует плечо до 5x в пике)")


if __name__ == '__main__':
    main()
