"""
Sniper Pullback v4 — бэктест с 2020 года
Показывает: годовой P&L %, сделки по месяцам, рост $4,000 со сложным процентом
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

DATA_START = '2018-01-01'   # для прогрева SMA200
TEST_START = '2020-01-01'
TEST_END   = '2026-04-11'

START_BALANCE = 4000.0
RISK_PCT      = 0.01        # 1% риска на сделку
COMMISSION    = 2.0         # $2 round-trip (IBKR)
MAX_POSITIONS = 4           # макс одновременных позиций

SMA_FAST, SMA_SLOW = 50, 200
ATR_LEN  = 14
RS_LEN   = 63
VOL_LEN  = 20
SR_LOOKBACK_MIN, SR_LOOKBACK_MAX = 20, 80
SR_TOUCH_PCT   = 2.5
CONSOL_BARS    = 5
CONSOL_RANGE   = 1.8
CONSOL_VOL_MIN = 0.8
ENTRY_BODY_MIN = 0.25
ENTRY_VOL_MIN  = 1.1
MAX_ATR_PCT    = 5.0
MIN_PRICE      = 20
MIN_VOL        = 500_000
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

def run_symbol(symbol, df_stock, spx_close):
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

        rs      = (close / close.shift(RS_LEN)) / (spx / spx.shift(RS_LEN))
        spx_s200= spx.rolling(200).mean()

        prior_high  = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        near_sr     = (close - prior_high).abs() / prior_high * 100 < SR_TOUCH_PCT
        pivot_sup   = low.rolling(15).min()
        near_pivot  = (close - pivot_sup).abs() / pivot_sup * 100 < 2.0
        near_sma50  = (close - sma50).abs() / sma50 * 100 < 3.0
        at_level    = near_sr | near_pivot | near_sma50

        consol_high  = high.shift(1).rolling(CONSOL_BARS).max()
        consol_low   = low.shift(1).rolling(CONSOL_BARS).min()
        tight_range  = (consol_high - consol_low) / atr < CONSOL_RANGE
        vol_at_level = volume.shift(1).rolling(CONSOL_BARS).mean() > vol_ma * CONSOL_VOL_MIN
        rs_at_level  = close.pct_change(CONSOL_BARS).shift(1) >= spx.pct_change(CONSOL_BARS).shift(1) - 0.01
        absorption   = tight_range & vol_at_level & rs_at_level

        body_size    = (close - open_) / atr
        strong_green = (close > open_) & (body_size > ENTRY_BODY_MIN) & (volume > vol_ma * ENTRY_VOL_MIN)
        market_ok    = spx > spx_s200
        trend_ok     = (close > sma200) & (sma50 > sma200)
        quality_ok   = ((atr / close * 100) < MAX_ATR_PCT) & (rs > 1.0) & (vol_ma > MIN_VOL) & (close > MIN_PRICE)

        entry_sig    = trend_ok & quality_ok & at_level & absorption & strong_green & market_ok
        mask         = (entry_sig.index >= TEST_START) & (entry_sig.index <= TEST_END)
        signal_dates = entry_sig[mask & entry_sig].index

        results = []
        bars = list(close.index)
        for sig_date in signal_dates:
            idx         = bars.index(sig_date)
            ep          = close.iloc[idx]
            sl_init     = sma50.iloc[idx] - atr.iloc[idx] * 1.5
            risk        = ep - sl_init
            risk_pct    = risk / ep * 100
            if risk <= 0 or risk_pct > 15: continue

            # Тейк: ближайшее сопротивление
            res_candidates = [high.iloc[max(0,idx-lb):idx].max()
                              for lb in [10,20,40,60]
                              if high.iloc[max(0,idx-lb):idx].max() > ep * 1.01]
            if res_candidates:
                rr = max(1.5, min(5.0, (min(res_candidates) - ep) / risk))
                tp = ep + risk * rr
            else:
                tp = ep + risk * 3.0

            # Выход (fixed mode)
            sl = sl_init
            exit_p, reason, hold = None, 'timeout', 0
            for j in range(1, 21):
                if idx + j >= len(bars): break
                bh = high.iloc[idx+j]; bl = low.iloc[idx+j]
                bc = close.iloc[idx+j]; bs = sma50.iloc[idx+j]; ba = atr.iloc[idx+j]
                hold = j
                if bl <= sl:   exit_p = sl;  reason = 'stop';    break
                if bh >= tp:   exit_p = tp;  reason = 'target';  break
                if bc < bs - ba * 0.5: exit_p = bc; reason = 'below_sma'; break
            if exit_p is None:
                exit_p = close.iloc[min(idx+hold, len(close)-1)]; reason = 'timeout'

            pnl_pct = (exit_p - ep) / ep * 100
            results.append({
                'symbol':   symbol,
                'date':     sig_date,
                'entry':    ep,
                'exit':     exit_p,
                'sl':       sl_init,
                'tp':       tp,
                'risk_pct': risk_pct,
                'pnl_pct':  pnl_pct,
                'win':      exit_p >= ep,
                'reason':   reason,
                'days':     hold,
            })
        return results
    except:
        return []


def simulate_account(trades_df, start=START_BALANCE, risk_pct=RISK_PCT, commission=COMMISSION):
    """Симулирует рост счёта со сложным процентом, макс MAX_POSITIONS одновременно"""
    df = trades_df.sort_values('date').copy()
    account = start
    open_positions = 0
    rows = []
    for _, t in df.iterrows():
        if open_positions >= MAX_POSITIONS:
            continue
        risk_dollars = account * risk_pct
        pos_size     = risk_dollars / (t['risk_pct'] / 100)
        pnl_dollars  = pos_size * (t['pnl_pct'] / 100) - commission
        account     += pnl_dollars
        rows.append({'date': t['date'], 'pnl_$': round(pnl_dollars, 2),
                     'account': round(account, 2), 'symbol': t['symbol'],
                     'win': t['win']})
    return pd.DataFrame(rows), account


def main():
    print("=== Sniper Pullback v4 — БЭКТЕСТ 2020–2026 ===\n")
    symbols = get_sp500()

    print("Загружаем SPY...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01', progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()

    print(f"Загружаем {len(symbols)} акций и сканируем...")
    all_trades = []
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        raw = yf.download(batch, start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                df = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
                trades = run_symbol(sym, df.copy(), spx_close)
                all_trades.extend(trades)
            except: pass
        print(f"  [{min(i+batch_size, len(symbols))}/{len(symbols)}] сделок: {len(all_trades)}")

    if not all_trades:
        print("Нет сделок."); return

    df = pd.DataFrame(all_trades)
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # ── 1. Итоги по годам ───────────────────────────────────
    print("\n" + "="*70)
    print("  РЕЗУЛЬТАТЫ ПО ГОДАМ")
    print("="*70)
    print(f"  {'Год':<6} {'Сделок':>7} {'Win%':>6} {'Avg P&L':>8} {'Avg Win':>8} {'Avg Loss':>9}")
    print("-"*70)
    for yr, g in df.groupby('year'):
        wins   = g[g['win']]['pnl_pct']
        losses = g[~g['win']]['pnl_pct']
        wr     = g['win'].mean()*100
        avg    = g['pnl_pct'].mean()
        aw     = wins.mean()  if len(wins)  else 0
        al     = losses.mean() if len(losses) else 0
        print(f"  {yr:<6} {len(g):>7}  {wr:>5.1f}%  {avg:>+7.2f}%  {aw:>+7.2f}%  {al:>+8.2f}%")
    g_all = df
    wins_all = g_all[g_all['win']]['pnl_pct']; losses_all = g_all[~g_all['win']]['pnl_pct']
    print("-"*70)
    print(f"  {'ИТОГО':<6} {len(df):>7}  {df['win'].mean()*100:>5.1f}%  "
          f"{df['pnl_pct'].mean():>+7.2f}%  {wins_all.mean():>+7.2f}%  {losses_all.mean():>+8.2f}%")

    # ── 2. Сделки по месяцам ────────────────────────────────
    print("\n" + "="*70)
    print("  СДЕЛОК В МЕСЯЦ (количество сигналов)")
    print("="*70)
    months_ru = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек']
    pivot = df.groupby(['year','month']).size().unstack(fill_value=0)
    # добавляем пустые месяцы если нет
    for m in range(1,13):
        if m not in pivot.columns: pivot[m] = 0
    pivot = pivot[[m for m in range(1,13) if m in pivot.columns]]
    header = f"  {'Год':<6}" + "".join(f"{months_ru[m-1]:>5}" for m in range(1,13)) + f"{'Итого':>7}"
    print(header)
    print("-"*70)
    for yr, row in pivot.iterrows():
        vals = [int(row.get(m, 0)) for m in range(1,13)]
        line = f"  {yr:<6}" + "".join(f"{v:>5}" for v in vals) + f"{sum(vals):>7}"
        print(line)
    totals = [int(pivot[m].sum()) if m in pivot.columns else 0 for m in range(1,13)]
    print("-"*70)
    total_line = f"  {'Итого':<6}" + "".join(f"{v:>5}" for v in totals) + f"{sum(totals):>7}"
    print(total_line)

    # ── 3. Рост счёта со сложным процентом ──────────────────
    print("\n" + "="*70)
    print(f"  РОСТ СЧЁТА: старт ${START_BALANCE:,.0f}, риск {RISK_PCT*100:.0f}%/сделку, комиссия ${COMMISSION}")
    print("="*70)
    sim_df, final = simulate_account(df)
    sim_df['year'] = pd.to_datetime(sim_df['date']).dt.year

    account = START_BALANCE
    print(f"\n  {'Год':<6} {'Сделок':>7} {'Win%':>6} {'P&L $':>9} {'P&L %':>8} {'Счёт в конце':>14}")
    print("-"*70)
    for yr, g in sim_df.groupby('year'):
        start_bal = account
        year_pnl  = g['pnl_$'].sum()
        account  += year_pnl  # уже включено в накопленный счёт
        # пересчитываем финальный счёт для года
        account_end = g['account'].iloc[-1]
        account = account_end
        wr_y = g['win'].mean()*100
        pct_y= (account_end - start_bal) / start_bal * 100
        print(f"  {yr:<6} {len(g):>7}  {wr_y:>5.1f}%  ${year_pnl:>+8,.0f}  {pct_y:>+7.1f}%  ${account_end:>12,.0f}")
    total_pct = (final - START_BALANCE) / START_BALANCE * 100
    print("-"*70)
    print(f"  {'ИТОГО':<6} {len(sim_df):>7}             ${final-START_BALANCE:>+8,.0f}  {total_pct:>+7.1f}%  ${final:>12,.0f}")
    print(f"\n  Старт: ${START_BALANCE:,.0f}  →  Финал: ${final:,.2f}")
    print(f"  Итоговый множитель: {final/START_BALANCE:.1f}x")

    df.to_csv('d:/projects/trading/signals_2020.csv', index=False)
    print(f"\n  Сохранено: signals_2020.csv")

if __name__ == '__main__':
    main()
