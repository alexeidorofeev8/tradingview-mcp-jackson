"""
Sniper Pullback — Type B: «Ложный пробой + восстановление»

Идея: дневной сканер находит акции у уровня (как v5),
но сигнал даёт НЕ «консолидация → прорыв вверх»,
а «кратковременный пробой НИЖЕ уровня → сильный отскок обратно».

Сравнение:
  Type A = v5 (текущая стратегия)
  Type B = ложный пробой + восстановление
  A+B    = комбинация (дополнительные сделки)

Бэктест 2020–2026, $4k счёт, риск 1%, с комиссиями.
"""

import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Настройки (те же что v5) ─────────────────────────────────────
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
SR_TOUCH_PCT             = 2.5
MAX_ATR_PCT              = 5.0
MIN_PRICE                = 20
MIN_VOL                  = 500_000
EXCLUDE_SECTORS          = {'Real Estate', 'Utilities'}

# Type A (v5) параметры
CONSOL_BARS   = 5
CONSOL_RANGE  = 1.8
CONSOL_VOL    = 0.8
ENTRY_BODY    = 0.25
ENTRY_VOL     = 1.1

# Type B параметры
B_BREAKDOWN_BARS  = 1    # пробой должен быть вчера (не 3 дня назад)
B_MAX_BREACH_PCT  = 1.5  # максимальный пробой ниже уровня (%) — апрель 2026: ужесточено с 3.0 до 1.5
B_RECOVERY_BODY   = 0.5  # тело свечи восстановления (сильнее чем v5)
B_RECOVERY_VOL    = 1.2  # объём на восстановлении (сильнее чем v5)
B_STOP_BUFFER     = 0.5  # ATR буфер ниже минимума пробоя

SECTOR_ETF = {
    'Information Technology': 'XLK', 'Financials': 'XLF',
    'Energy': 'XLE',                  'Health Care': 'XLV',
    'Industrials': 'XLI',             'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',        'Materials': 'XLB',
    'Communication Services': 'XLC',
}
EARNINGS_CACHE = 'd:/projects/trading/earnings_cache.json'


# ─── Вспомогательные ─────────────────────────────────────────────
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
    print("  Кэш отчётов не найден — нужно сначала запустить backtest_v5.py")
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


# ─── Сигналы Type A (v5) ──────────────────────────────────────────
def signals_type_a(close, high, low, open_, volume, atr, vol_ma, sma50, sma200, spx, rs):
    prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
    at_level = (
        ((close - prior_high).abs() / prior_high * 100 < SR_TOUCH_PCT) |
        ((close - low.rolling(15).min()).abs() / low.rolling(15).min() * 100 < 2.0) |
        ((close - sma50).abs() / sma50 * 100 < 3.0)
    )
    absorption = (
        ((high.shift(1).rolling(CONSOL_BARS).max() - low.shift(1).rolling(CONSOL_BARS).min()) / atr < CONSOL_RANGE) &
        (volume.shift(1).rolling(CONSOL_BARS).mean() > vol_ma * CONSOL_VOL) &
        (close.pct_change(CONSOL_BARS).shift(1) >= spx.pct_change(CONSOL_BARS).shift(1) - 0.01)
    )
    strong_green = (
        (close > open_) &
        ((close - open_) / atr > ENTRY_BODY) &
        (volume > vol_ma * ENTRY_VOL)
    )
    trend_ok   = (close > sma200) & (sma50 > sma200)
    quality_ok = ((atr / close * 100) < MAX_ATR_PCT) & (rs > 1.0) & (vol_ma > MIN_VOL) & (close > MIN_PRICE)
    market_ok  = spx > spx.rolling(200).mean()
    return trend_ok & quality_ok & at_level & absorption & strong_green & market_ok


# ─── Сигналы Type B (ложный пробой + восстановление) ──────────────
def signals_type_b(close, high, low, open_, volume, atr, vol_ma, sma50, sma200, spx, rs):
    # Уровень (разворот уровня — приоритет ⭐⭐⭐)
    prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
    local_min  = low.rolling(15).min().shift(1)

    # Проверяем оба уровня
    for level in [prior_high, local_min]:
        pass  # строим маски ниже для обоих

    # Маска 1: ложный пробой уровня в последних B_BREAKDOWN_BARS барах
    # (минимум последних N баров пробил уровень, но не более чем на B_MAX_BREACH_PCT%)
    recent_low = low.shift(1).rolling(B_BREAKDOWN_BARS).min()  # минимум за последние N баров (без сегодня)

    false_break_ph = (
        (recent_low < prior_high) &
        (recent_low >= prior_high * (1 - B_MAX_BREACH_PCT / 100))
    )
    false_break_lm = (
        (recent_low < local_min) &
        (recent_low >= local_min * (1 - B_MAX_BREACH_PCT / 100))
    )
    false_break = false_break_ph | false_break_lm

    # Маска 2: восстановление сегодня
    # Сегодня цена вернулась ВЫШЕ уровня с силой
    recovery_ph = (close > prior_high) & (low.shift(1) < prior_high)  # вчера было ниже, сегодня выше
    recovery_lm = (close > local_min)  & (low.shift(1) < local_min)

    # Свеча восстановления: сильная зелёная с объёмом
    strong_recovery = (
        (close > open_) &
        ((close - open_) / atr > B_RECOVERY_BODY) &
        (volume > vol_ma * B_RECOVERY_VOL)
    )

    # Восстановление от любого уровня
    recovered = ((recovery_ph | recovery_lm) & false_break) | (
        # Или: сегодня вернулась выше уровня после того как была ниже
        (close > prior_high * 0.999) & (close.shift(1) < prior_high) & false_break_ph |
        (close > local_min  * 0.999) & (close.shift(1) < local_min)  & false_break_lm
    )

    trend_ok   = (close > sma200) & (sma50 > sma200)
    quality_ok = ((atr / close * 100) < MAX_ATR_PCT) & (rs > 1.0) & (vol_ma > MIN_VOL) & (close > MIN_PRICE)
    market_ok  = spx > spx.rolling(200).mean()

    return trend_ok & quality_ok & false_break & recovered & strong_recovery & market_ok


# ─── Бэктест одной акции ─────────────────────────────────────────
def run_symbol(sym, df_stock, spx_close, earnings, sec_leaders, stock_sec, mode='both'):
    """
    mode: 'A' = только Type A, 'B' = только Type B, 'both' = оба
    Возвращает список сделок с полем 'type' = 'A' или 'B'
    """
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

        results = []
        bars = list(close.index)
        earn_set = earnings.get(sym, set())

        # Собираем сигналы
        sigs = {}
        mask = (close.index >= TEST_START) & (close.index <= TEST_END)

        if mode in ('A', 'both'):
            sig_a = signals_type_a(close, high, low, open_, volume, atr, vol_ma, sma50, sma200, spx, rs)
            for d in sig_a[mask & sig_a].index:
                sigs[d] = 'A'

        if mode in ('B', 'both'):
            sig_b = signals_type_b(close, high, low, open_, volume, atr, vol_ma, sma50, sma200, spx, rs)
            for d in sig_b[mask & sig_b].index:
                if d not in sigs:  # A имеет приоритет
                    sigs[d] = 'B'

        for sig_date, sig_type in sorted(sigs.items()):
            # Фильтр: отчётность
            sig_d = sig_date.date()
            if any(0 <= (pd.to_datetime(ed).date() - sig_d).days <= EARNINGS_BUFFER for ed in earn_set):
                continue

            # Фильтр: сектор
            if stock_sec and sig_date in sec_leaders:
                if stock_sec not in sec_leaders[sig_date]:
                    continue

            idx_i = bars.index(sig_date)

            # Вход
            entry_price = close.iloc[idx_i] * (1 - HOURLY_DISC)

            # Стоп: для Type B — под минимумом ложного пробоя
            if sig_type == 'B':
                lookback = min(B_BREAKDOWN_BARS + 1, idx_i)
                recent_lows = low.iloc[idx_i - lookback: idx_i]
                breakdown_low = recent_lows.min() if len(recent_lows) > 0 else low.iloc[idx_i]
                sl_init = breakdown_low - atr.iloc[idx_i] * B_STOP_BUFFER
            else:
                sl_init = sma50.iloc[idx_i] - atr.iloc[idx_i] * 1.5

            risk = entry_price - sl_init
            risk_pct = risk / entry_price * 100
            if risk <= 0 or risk_pct > 15: continue

            # Тейк: ближайшее сопротивление (как v5)
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

            # Проверяем R:R — отсеиваем если меньше 1.0 (убыточные в ожидании)
            actual_rr = (tp - entry_price) / risk
            if actual_rr < 1.0: continue

            # Симуляция выхода (max 20 дней, как v5)
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
                reason = 'timeout'

            results.append({
                'symbol':   sym,
                'type':     sig_type,
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
    except Exception as e:
        return []


# ─── Симуляция счёта ─────────────────────────────────────────────
def simulate_account(trades_df, label=''):
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
            'type':    t.get('type', '?'),
            'win':     t['win'],
        })
    return pd.DataFrame(rows), account


# ─── Печать таблицы по годам ──────────────────────────────────────
def print_year_table(sim_df, label, ref_final=None):
    account = START_BALANCE
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  {'Год':<5} {'Сд':>4} {'Win%':>6} {'P&L $':>8} {'P&L %':>7} {'Счёт':>9}")
    print(f"  {'-'*55}")
    for yr, g in sim_df.groupby(sim_df['date'].dt.year):
        start = account
        end   = g['account'].iloc[-1]
        account = end
        wr  = g['win'].mean() * 100
        pnl = g['pnl_$'].sum()
        pct = (end - start) / start * 100
        print(f"  {yr:<5} {len(g):>4}  {wr:>5.1f}%  ${pnl:>+7,.0f}  {pct:>+6.1f}%  ${end:>8,.0f}")
    total_pct = (account - START_BALANCE) / START_BALANCE * 100
    cagr = (account / START_BALANCE) ** (1/6.3) - 1
    print(f"  {'-'*55}")
    print(f"  $4,000 → ${account:,.0f}  ({total_pct:+.0f}%)  CAGR {cagr*100:.1f}%/yr")


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=== Sniper Pullback: Type A vs Type B vs A+B ===\n")

    # Загружаем данные
    print("Шаг 1: S&P 500 список...")
    import requests
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers, timeout=15)
    import io as _io
    df_wiki = pd.read_html(_io.StringIO(resp.text))[0]
    df_wiki = df_wiki[~df_wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols = df_wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    sector_map = dict(zip(
        df_wiki['Symbol'].str.replace('.', '-', regex=False),
        df_wiki['GICS Sector']
    ))
    print(f"  Акций: {len(symbols)}")

    print("\nШаг 2: Загружаем SPY + секторы...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()
    sec_leaders = build_sector_leaders(spx_close)

    print("\nШаг 3: Загружаем даты отчётов из кэша...")
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

    print(f"\nШаг 5: Бэктест (три режима)...")
    all_trades_a    = []
    all_trades_b    = []
    all_trades_both = []

    for sym, df in all_data.items():
        sec = sector_map.get(sym)
        ta   = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec, mode='A')
        tb   = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec, mode='B')
        both = run_symbol(sym, df.copy(), spx_close, earnings, sec_leaders, sec, mode='both')
        all_trades_a.extend(ta)
        all_trades_b.extend(tb)
        all_trades_both.extend(both)

    print(f"\n  Type A сделок: {len(all_trades_a)}")
    print(f"  Type B сделок: {len(all_trades_b)}")
    print(f"  Комбо A+B:    {len(all_trades_both)}")

    # Конвертируем в DataFrame
    dfa = pd.DataFrame(all_trades_a); dfa['date'] = pd.to_datetime(dfa['date'])
    dfb = pd.DataFrame(all_trades_b); dfb['date'] = pd.to_datetime(dfb['date'])
    dfab = pd.DataFrame(all_trades_both); dfab['date'] = pd.to_datetime(dfab['date'])

    # Симуляция счёта
    sim_a,  final_a  = simulate_account(dfa,  'Type A')
    sim_b,  final_b  = simulate_account(dfb,  'Type B')
    sim_ab, final_ab = simulate_account(dfab, 'A+B')

    # Результаты по годам
    for sim, label, final in [
        (sim_a,  'TYPE A (v5 — текущая)',              final_a),
        (sim_b,  'TYPE B (ложный пробой + отскок)',    final_b),
        (sim_ab, 'КОМБО A+B (дополнительные сделки)',  final_ab),
    ]:
        if len(sim) > 0:
            print_year_table(sim, label)

    # Сравнительная таблица
    print(f"\n{'='*65}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{'='*65}")
    print(f"  {'Режим':<25} {'Сделок':>7} {'Win%':>6} {'Финал':>10} {'CAGR':>8}")
    print(f"  {'-'*55}")
    for label, sim, final, df_ in [
        ('Type A (v5)',          sim_a,  final_a,  dfa),
        ('Type B (new)',         sim_b,  final_b,  dfb),
        ('A+B (комбо)',          sim_ab, final_ab, dfab),
    ]:
        if len(sim) > 0:
            wr   = sim['win'].mean() * 100
            cagr = (final / START_BALANCE) ** (1/6.3) - 1
            print(f"  {label:<25} {len(sim):>7}  {wr:>5.1f}%  ${final:>8,.0f}  {cagr*100:>7.1f}%")

    # Детали Type B сделок
    if len(dfb) > 0:
        print(f"\n{'='*65}")
        print("  TYPE B — ДЕТАЛИ ПО ПРИЧИНАМ ВЫХОДА")
        print(f"{'='*65}")
        for reason, g in dfb.groupby('reason'):
            wr  = g['win'].mean() * 100
            avg = g['pnl_pct'].mean()
            print(f"  {reason:<12}: {len(g):>4} сделок, WR {wr:.0f}%, avg P&L {avg:+.2f}%")

        avg_rr = dfb['rr'].mean()
        avg_risk = dfb['risk_pct'].mean()
        print(f"\n  Средний R:R: {avg_rr:.2f}")
        print(f"  Средний риск на сделку: {avg_risk:.1f}%")

        print(f"\n  Топ-10 Type B сделок:")
        top = dfb.nlargest(10, 'pnl_pct')[['symbol','date','entry','exit','rr','pnl_pct','reason','days']]
        for _, r in top.iterrows():
            print(f"  {str(r['date'])[:10]}  {r['symbol']:<5} rr={r['rr']:.1f} pnl={r['pnl_pct']:+.1f}% {r['reason']} {r['days']}d")

    # Сохраняем
    if len(dfb) > 0:
        dfb.to_csv('d:/projects/trading/signals_type_b.csv', index=False)
        print(f"\n  Сохранено: signals_type_b.csv ({len(dfb)} сделок)")
    if len(dfab) > 0:
        dfab.to_csv('d:/projects/trading/signals_combined.csv', index=False)
        print(f"  Сохранено: signals_combined.csv ({len(dfab)} сделок)")


if __name__ == '__main__':
    main()
