"""
Sniper Pullback Strategy v4 — Wyckoff Absorption
Улучшения vs v3:
  1. Поглощение на уровне: узкие бары (красная/зелёная/красная = бой на уровне)
  2. Объём был во время проторговки но не пробило
  3. Акция держалась пока рынок падал (RS прямо на уровне)
  4. Вход на первой сильной зелёной свече с объёмом
  5. S/R flip зоны (из v3)
  6. Walk-forward: обучение 2023-2024 → тест 2025-2026
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

# ─── Настройки ───────────────────────────────────────────
TRAIN_START = '2023-01-01'
TRAIN_END   = '2024-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2026-04-11'
DATA_START  = '2021-06-01'

SMA_FAST    = 50
SMA_SLOW    = 200
ATR_LEN     = 14
MAX_ATR_PCT = 5.0
RS_LEN      = 63
RS_MIN      = 1.0
VOL_LEN     = 20
MIN_PRICE   = 20
MIN_VOL     = 500_000

# S/R flip
SR_LOOKBACK_MIN = 20
SR_LOOKBACK_MAX = 80
SR_TOUCH_PCT    = 2.5

# Поглощение (absorption)
CONSOL_BARS     = 5    # баров проторговки перед входом
CONSOL_RANGE    = 1.8  # ширина проторговки в ATR (узко = поглощение)
CONSOL_VOL_MIN  = 0.8  # объём во время проторговки >= 80% от средн.
ENTRY_BODY_MIN  = 0.25 # минимальный тел свечи входа в ATR
ENTRY_VOL_MIN   = 1.1  # объём входной свечи >= 110% средн.

# Walk-forward
TRAIN_MIN_SIGNALS = 2
TRAIN_MIN_WINRATE = 0.40

EXCLUDE_SECTORS = {'Real Estate', 'Utilities'}

# ─── Получить список S&P 500 ─────────────────────────────
def get_sp500():
    print("Загружаем список S&P 500 с Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'}
    resp = requests.get(url, headers=headers, timeout=15)
    df = pd.read_html(io.StringIO(resp.text))[0]
    filtered = df[~df['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols = filtered['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"После фильтрации секторов: {len(symbols)} акций")
    return symbols

def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

# ─── Основная логика сигналов ────────────────────────────
def run_period(symbol, df_stock, spx_close, sig_start, sig_end):
    try:
        if isinstance(df_stock.columns, pd.MultiIndex):
            df_stock.columns = df_stock.columns.get_level_values(0)

        close  = df_stock['Close'].squeeze()
        high   = df_stock['High'].squeeze()
        low    = df_stock['Low'].squeeze()
        open_  = df_stock['Open'].squeeze()
        volume = df_stock['Volume'].squeeze()

        avg_vol   = volume.rolling(20).mean()
        avg_price = close.rolling(20).mean()
        if avg_price.iloc[-1] < MIN_PRICE or avg_vol.iloc[-1] < MIN_VOL:
            return []

        sma50  = close.rolling(SMA_FAST).mean()
        sma200 = close.rolling(SMA_SLOW).mean()
        atr    = calc_atr(high, low, close, ATR_LEN)
        vol_ma = volume.rolling(VOL_LEN).mean()

        spx       = spx_close.reindex(close.index, method='ffill')
        stock_chg = close / close.shift(RS_LEN)
        spx_chg   = spx   / spx.shift(RS_LEN)
        rs        = stock_chg / spx_chg

        # ── Уровни: S/R flip + pivot low + SMA50 ──────────
        prior_high  = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        near_sr     = (close - prior_high).abs() / prior_high * 100 < SR_TOUCH_PCT
        pivot_sup   = low.rolling(15).min()
        near_pivot  = (close - pivot_sup).abs() / pivot_sup * 100 < 2.0
        near_sma50  = (close - sma50).abs() / sma50 * 100 < 3.0
        at_level    = near_sr | near_pivot | near_sma50

        # ── Поглощение на уровне (Wyckoff absorption) ─────
        # 1. Проторговка: последние N баров в узком диапазоне
        consol_high  = high.shift(1).rolling(CONSOL_BARS).max()
        consol_low   = low.shift(1).rolling(CONSOL_BARS).min()
        consol_range = (consol_high - consol_low) / atr
        tight_range  = consol_range < CONSOL_RANGE

        # 2. Объём во время проторговки был (пытались пробить — не вышло)
        consol_vol   = volume.shift(1).rolling(CONSOL_BARS).mean()
        vol_at_level = consol_vol > vol_ma * CONSOL_VOL_MIN

        # 3. Акция держалась пока рынок падал/стоял
        #    Сравниваем изменение акции vs SPX за период проторговки
        spx_period   = spx.pct_change(CONSOL_BARS).shift(1)
        stk_period   = close.pct_change(CONSOL_BARS).shift(1)
        rs_at_level  = stk_period >= spx_period - 0.01  # акция держалась не хуже рынка

        absorption   = tight_range & vol_at_level & rs_at_level

        # ── Входная свеча ──────────────────────────────────
        # Сильная зелёная свеча с объёмом (не просто маленький зелёный бар)
        body_size    = (close - open_) / atr
        strong_green = (close > open_) & (body_size > ENTRY_BODY_MIN) & (volume > vol_ma * ENTRY_VOL_MIN)

        # ── Режим рынка ────────────────────────────────────
        spx_sma200  = spx.rolling(200).mean()
        market_ok   = spx > spx_sma200

        # ── Тренд и качество акции ─────────────────────────
        trend_ok    = (close > sma200) & (sma50 > sma200)
        vol_ok      = (atr / close * 100) < MAX_ATR_PCT
        rs_ok       = rs > RS_MIN

        # ── Финальный сигнал ───────────────────────────────
        entry = trend_ok & vol_ok & rs_ok & at_level & absorption & strong_green & market_ok

        mask         = (entry.index >= sig_start) & (entry.index <= sig_end)
        signal_dates = entry[mask & entry].index

        results = []
        bars = list(close.index)

        for sig_date in signal_dates:
            idx         = bars.index(sig_date)
            entry_price = close.iloc[idx]
            sl_init     = sma50.iloc[idx] - atr.iloc[idx] * 1.5
            risk        = entry_price - sl_init
            risk_pct    = risk / entry_price * 100

            # ── Умный тейк: ближайшее сопротивление выше входа ──
            res_candidates = []
            for lookback in [10, 20, 40, 60]:
                start_i = max(0, idx - lookback)
                res = high.iloc[start_i:idx].max()
                if res > entry_price * 1.01:
                    res_candidates.append(res)
            if res_candidates:
                rr = max(1.5, min(5.0, (min(res_candidates) - entry_price) / risk))
                tp = entry_price + risk * rr
            else:
                tp = entry_price + risk * 3.0
            tp_rr = (tp - entry_price) / risk

            # ── Три варианта выхода ───────────────────────────
            exit_results = {}

            for mode in ['fixed', 'trail_tight', 'trail_wide']:
                sl      = sl_init
                ep      = None
                reason  = 'timeout'
                hdays   = 0
                run_hi  = entry_price   # бегущий максимум для трейлинга

                trail_mult = 1.5 if mode == 'trail_tight' else 2.5

                for j in range(1, 31):   # до 30 дней для трейлинга
                    if idx + j >= len(bars):
                        break
                    bar_high  = high.iloc[idx + j]
                    bar_low   = low.iloc[idx + j]
                    bar_close = close.iloc[idx + j]
                    bar_sma50 = sma50.iloc[idx + j]
                    bar_atr   = atr.iloc[idx + j]
                    hdays = j

                    # Обновляем трейлинг стоп
                    if mode != 'fixed':
                        run_hi = max(run_hi, bar_high)
                        trail_sl = run_hi - bar_atr * trail_mult
                        sl = max(sl, trail_sl)   # стоп только вверх

                    # Проверяем выходы
                    if bar_low <= sl:
                        ep = sl; reason = 'stop' if sl <= sl_init * 1.001 else 'trail'; break
                    if bar_high >= tp:
                        ep = tp; reason = 'target'; break
                    # below_sma только для fixed режима
                    if mode == 'fixed' and bar_close < bar_sma50 - bar_atr * 0.5:
                        ep = bar_close; reason = 'below_sma'; break

                if ep is None:
                    ep     = close.iloc[min(idx + hdays, len(close)-1)]
                    reason = 'timeout'

                exit_results[mode] = {
                    'exit': round(ep, 2),
                    'exit_reason': reason,
                    'hold_days': hdays,
                    'pnl_pct': round((ep - entry_price) / entry_price * 100, 2),
                    'win': ep >= entry_price,
                }

            base = exit_results['fixed']
            results.append({
                'symbol':      symbol,
                'date':        str(sig_date.date()),
                'entry':       round(entry_price, 2),
                'sl':          round(sl_init, 2),
                'tp':          round(tp, 2),
                'tp_rr':       round(tp_rr, 2),
                'risk_pct':    round(risk_pct, 2),
                # fixed
                'exit_fixed':       base['exit'],
                'reason_fixed':     base['exit_reason'],
                'days_fixed':       base['hold_days'],
                'pnl_fixed':        base['pnl_pct'],
                'win_fixed':        base['win'],
                # trail tight
                'exit_tight':       exit_results['trail_tight']['exit'],
                'reason_tight':     exit_results['trail_tight']['exit_reason'],
                'days_tight':       exit_results['trail_tight']['hold_days'],
                'pnl_tight':        exit_results['trail_tight']['pnl_pct'],
                'win_tight':        exit_results['trail_tight']['win'],
                # trail wide
                'exit_wide':        exit_results['trail_wide']['exit'],
                'reason_wide':      exit_results['trail_wide']['exit_reason'],
                'days_wide':        exit_results['trail_wide']['hold_days'],
                'pnl_wide':         exit_results['trail_wide']['pnl_pct'],
                'win_wide':         exit_results['trail_wide']['win'],
            })

        return results
    except Exception:
        return []

# ─── Вывод статистики ────────────────────────────────────
def print_stats(label, trades):
    if not trades:
        print(f"\n{label}: нет сделок"); return None
    df = pd.DataFrame(trades)
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.to_period('M')
    months = df['month'].nunique()

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Сделок: {len(df)}   Акций: {df['symbol'].nunique()}   В месяц: {len(df)/months:.1f}")
    print()

    for mode, col_pnl, col_win, col_days, col_reason in [
        ('Фикс. выход  ', 'pnl_fixed',  'win_fixed',  'days_fixed',  'reason_fixed'),
        ('Трейл узкий  ', 'pnl_tight',  'win_tight',  'days_tight',  'reason_tight'),
        ('Трейл широкий', 'pnl_wide',   'win_wide',   'days_wide',   'reason_wide'),
    ]:
        wins   = df[df[col_win] == True][col_pnl]
        losses = df[df[col_win] == False][col_pnl]
        wr     = df[col_win].mean() * 100
        print(f"  {mode}  WR {wr:4.1f}%  avg {df[col_pnl].mean():+.2f}%  "
              f"win +{wins.mean():.2f}%  loss {losses.mean():.2f}%  "
              f"hold {df[col_days].mean():.1f}д")

    print(f"\n  По годам:")
    for mode, col_pnl, col_win in [
        ('Фикс ', 'pnl_fixed', 'win_fixed'),
        ('Тайт ', 'pnl_tight', 'win_tight'),
        ('Вайд ', 'pnl_wide',  'win_wide'),
    ]:
        print(f"    {mode}", end='')
        for yr, g in df.groupby('year'):
            wr_y = g[col_win].mean() * 100
            print(f"  {yr}: WR {wr_y:2.0f}% avg {g[col_pnl].mean():+.1f}%", end='')
        print()

    print(f"\n  Выходы (трейл широкий):")
    for r, c in df['reason_wide'].value_counts().items():
        print(f"    {r:<12} {c:4d}  ({c/len(df)*100:.0f}%)")

    return df

# ─── Main ────────────────────────────────────────────────
if __name__ == '__main__':
    symbols = get_sp500()

    print(f"\nЗагружаем SPX...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()

    # ── Фаза 1: скачиваем всё и обучаем ─────────────────
    print(f"\nФаза 1: обучение {TRAIN_START}—{TRAIN_END}...")
    stock_data   = {}
    train_trades = []
    qualified    = []

    for i, sym in enumerate(symbols):
        try:
            df = yf.download(sym, start=DATA_START, end='2027-01-01',
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < 220:
                continue
            stock_data[sym] = df
            trades = run_period(sym, df.copy(), spx_close, TRAIN_START, TRAIN_END)
            if len(trades) >= TRAIN_MIN_SIGNALS:
                wr = sum(t['win_fixed'] for t in trades) / len(trades)
                if wr >= TRAIN_MIN_WINRATE:
                    qualified.append(sym)
            train_trades.extend(trades)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(symbols)}] сделок: {len(train_trades)}  квалиф: {len(qualified)}")

    print(f"\nКвалифицировано: {len(qualified)} акций из {len(symbols)}")

    # ── Фаза 2: тест на квалифицированных ───────────────
    print(f"\nФаза 2: тест {TEST_START}—{TEST_END} на {len(qualified)} акциях...")
    test_trades = []
    for sym in qualified:
        if sym not in stock_data:
            continue
        trades = run_period(sym, stock_data[sym].copy(), spx_close, TEST_START, TEST_END)
        test_trades.extend(trades)

    # ── Результаты ───────────────────────────────────────
    print_stats(f"ВСЕ АКЦИИ 2023-2026 (квалифицированные {len(qualified)})", test_trades)

    if test_trades:
        pd.DataFrame(test_trades).to_csv('d:/projects/trading/signals_v4.csv', index=False)
        print(f"\n  Сохранено: signals_v4.csv")
