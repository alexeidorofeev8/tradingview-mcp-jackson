"""
Бэктест: Динамический риск по числу открытых позиций
=====================================================
Гипотеза: если слоты свободны — рисковать 2%, если заполнены — 1%.
Плечо (маржа) всегда доступно → нет жёсткого лимита на позиции.

Сравнение трёх вариантов:
  flat_1 : всегда 1%, без лимита позиций
  flat_2 : всегда 2%, без лимита позиций
  dynamic: 2% → 1.5% → 1% в зависимости от загрузки

Старт: $10,000. Type B only. Без секторного фильтра.
"""

import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Настройки ────────────────────────────────────────────────────
DATA_START  = '2007-01-01'
TEST_START  = '2010-01-01'
TEST_END    = '2026-04-11'
TEST_YEARS  = 16.3

START_BALANCE = 10_000.0
COMMISSION    = 2.0

EARNINGS_BUFFER = 7
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

B_BREAKDOWN_BARS  = 1
B_MAX_BREACH_PCT  = 1.5
B_RECOVERY_BODY   = 0.5
B_RECOVERY_VOL    = 1.2
B_STOP_BUFFER     = 0.5

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
    print("  Кэш отчётов не найден")
    return {}


# ─── Сигналы Type B ───────────────────────────────────────────────
def signals_type_b(close, high, low, open_, volume, atr, vol_ma, sma50, sma200, spx, rs):
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

    return trend_ok & quality_ok & false_break & recovered & strong_recovery & market_ok


# ─── Сигналы одной акции ─────────────────────────────────────────
def run_symbol(sym, df_stock, spx_close, earnings):
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

        mask  = (close.index >= TEST_START) & (close.index <= TEST_END)
        sig_b = signals_type_b(close, high, low, open_, volume, atr, vol_ma, sma50, sma200, spx, rs)

        for sig_date in sig_b[mask & sig_b].index:
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
                reason = 'timeout'

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


# ─── Варианты динамического риска ────────────────────────────────
def get_risk(n_open, deployed_ratio, mode):
    """
    n_open:        число открытых позиций
    deployed_ratio: доля занятой маржи (0.0–1.0)
    mode:          вариант расчёта риска
    """
    if mode == 'flat_1':
        return 0.01
    elif mode == 'flat_2':
        return 0.02
    elif mode == 'dynamic':       # оригинал: 2→1.5→1 по слотам
        if n_open <= 1:   return 0.02
        elif n_open <= 3: return 0.015
        else:             return 0.01
    elif mode == 'varA':          # поднять пол: 2→1.5→1.5
        if n_open <= 1:   return 0.02
        elif n_open <= 3: return 0.015
        else:             return 0.015
    elif mode == 'varB':          # сдвинуть вверх: 2.5→2→1.5
        if n_open <= 1:   return 0.025
        elif n_open <= 3: return 0.02
        else:             return 0.015
    elif mode == 'varC':          # дольше держать 2%: 2→2→1.5
        if n_open <= 3:   return 0.02
        else:             return 0.015
    elif mode == 'margin':        # по остатку маржи
        if deployed_ratio < 0.30:   return 0.02
        elif deployed_ratio < 0.60: return 0.015
        else:                       return 0.01
    return 0.01


# ─── Симуляция счёта ─────────────────────────────────────────────
def simulate(trades_df, mode='dynamic', leverage=2.0):
    df = trades_df.sort_values('date').copy()
    account  = START_BALANCE
    open_pos = []   # (дата_выхода, размер_позиции_$)
    rows     = []
    n_open_log  = []
    skipped_lev = 0

    for _, t in df.iterrows():
        e_date = pd.to_datetime(t['date'])
        x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)
        open_pos = [(d, ps) for d, ps in open_pos if d > e_date]

        n_open         = len(open_pos)
        deployed       = sum(ps for _, ps in open_pos)
        max_deployed   = account * leverage
        deployed_ratio = deployed / max_deployed if max_deployed > 0 else 0

        risk_pct = get_risk(n_open, deployed_ratio, mode)
        pos_size = (account * risk_pct) / (t['risk_pct'] / 100)

        if deployed + pos_size > max_deployed:
            skipped_lev += 1
            continue

        n_open_log.append(n_open)
        pnl_usd = pos_size * (t['pnl_pct'] / 100) - COMMISSION
        account += pnl_usd
        open_pos.append((x_date, pos_size))

        rows.append({
            'date':      t['date'],
            'pnl_$':     round(pnl_usd, 2),
            'account':   round(account, 2),
            'symbol':    t['symbol'],
            'win':       t['win'],
            'n_open':    n_open,
            'risk_used': risk_pct,
        })

    df_out    = pd.DataFrame(rows)
    avg_slots = np.mean(n_open_log) if n_open_log else 0
    return df_out, account, avg_slots, skipped_lev


# ─── Просадка ────────────────────────────────────────────────────
def max_drawdown(sim_df):
    equity = sim_df['account']
    peak   = equity.cummax()
    dd     = (equity - peak) / peak
    return dd.min() * 100


# ─── Таблица по годам ─────────────────────────────────────────────
def print_year_table(sim_df, label):
    account = START_BALANCE
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    print(f"  {'Год':<5} {'Сд':>4} {'Win%':>6} {'P&L $':>8} {'P&L %':>7} {'Счёт':>10}")
    print(f"  {'-'*58}")
    for yr, g in sim_df.groupby(sim_df['date'].dt.year):
        start = account
        end   = g['account'].iloc[-1]
        account = end
        wr  = g['win'].mean() * 100
        pnl = g['pnl_$'].sum()
        pct = (end - start) / start * 100
        print(f"  {yr:<5} {len(g):>4}  {wr:>5.1f}%  ${pnl:>+7,.0f}  {pct:>+6.1f}%  ${end:>9,.0f}")
    total_pct = (account - START_BALANCE) / START_BALANCE * 100
    cagr = (account / START_BALANCE) ** (1 / TEST_YEARS) - 1
    mdd  = max_drawdown(sim_df)
    avg_slots = sim_df['n_open'].mean() if 'n_open' in sim_df.columns else 0
    print(f"  {'-'*58}")
    print(f"  $10k → ${account:,.0f}  ({total_pct:+.0f}%)  CAGR {cagr*100:.1f}%  MaxDD {mdd:.1f}%  avgSlots {avg_slots:.1f}")


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=== Динамический риск: flat 1% vs flat 2% vs dynamic ===\n")

    print("Шаг 1: S&P 500 список...")
    import requests, io as _io
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp    = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                           headers=headers, timeout=15)
    df_wiki = pd.read_html(_io.StringIO(resp.text))[0]
    df_wiki = df_wiki[~df_wiki['GICS Sector'].isin(EXCLUDE_SECTORS)]
    symbols = df_wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    print(f"  Акций: {len(symbols)}")

    print("\nШаг 2: Загружаем SPY...")
    spx_raw = yf.download('SPY', start=DATA_START, end='2027-01-01',
                          progress=False, auto_adjust=True)
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    spx_close = spx_raw['Close'].squeeze()

    print("\nШаг 3: Загружаем даты отчётов из кэша...")
    earnings = load_earnings(symbols)

    print(f"\nШаг 4: Загружаем котировки {len(symbols)} акций...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw   = yf.download(batch, start=DATA_START, end='2027-01-01',
                            progress=False, auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except: pass
        print(f"  [{min(i+50, len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    print(f"\nШаг 5: Генерируем сигналы Type B (2016–2026)...")
    all_trades = []
    for sym, df in all_data.items():
        trades = run_symbol(sym, df.copy(), spx_close, earnings)
        all_trades.extend(trades)

    print(f"  Всего сигналов: {len(all_trades)}")
    if not all_trades:
        print("Нет сигналов — проверьте данные")
        return

    dfb = pd.DataFrame(all_trades)
    dfb['date'] = pd.to_datetime(dfb['date'])
    dfb = dfb.sort_values('date').reset_index(drop=True)

    print(f"\nШаг 6: Полный перебор вариантов...")

    # ─── Grid search ──────────────────────────────────────────────
    # Уровни риска
    risk_levels = [0.01, 0.015, 0.02, 0.025, 0.03]
    # Пороги по числу позиций
    thresh_pairs = [(1,3), (1,4), (2,4), (2,5), (3,5)]

    results = []

    # Базовые
    for mode in ('flat_1', 'flat_2'):
        sim, final, slots, skip = simulate(dfb, mode=mode, leverage=2.0)
        cagr = (final / START_BALANCE) ** (1 / TEST_YEARS) - 1
        mdd  = max_drawdown(sim)
        label = 'flat 1%' if mode == 'flat_1' else 'flat 2%'
        results.append({'label': label, 'final': final, 'cagr': cagr*100,
                        'mdd': mdd, 'trades': len(sim), 'skip': skip})

    # Вариант по марже
    sim, final, slots, skip = simulate(dfb, mode='margin', leverage=2.0)
    cagr = (final / START_BALANCE) ** (1 / TEST_YEARS) - 1
    mdd  = max_drawdown(sim)
    results.append({'label': 'margin-based', 'final': final, 'cagr': cagr*100,
                    'mdd': mdd, 'trades': len(sim), 'skip': skip})

    # Перебор: 3 уровня риска + 2 порога
    def simulate_grid(trades_df, r_high, r_mid, r_low, t1, t2, leverage=2.0):
        df = trades_df.sort_values('date').copy()
        account, open_pos, rows, skipped = START_BALANCE, [], [], 0
        for _, t in df.iterrows():
            e_date = pd.to_datetime(t['date'])
            x_date = e_date + pd.Timedelta(days=int(t['days']) + 1)
            open_pos = [(d, ps) for d, ps in open_pos if d > e_date]
            n = len(open_pos)
            deployed = sum(ps for _, ps in open_pos)
            max_dep  = account * leverage
            if n <= t1:       r = r_high
            elif n <= t2:     r = r_mid
            else:             r = r_low
            pos = (account * r) / (t['risk_pct'] / 100)
            if deployed + pos > max_dep:
                skipped += 1; continue
            pnl = pos * (t['pnl_pct'] / 100) - COMMISSION
            account += pnl
            open_pos.append((x_date, pos))
            rows.append({'date': t['date'], 'pnl_$': pnl, 'account': account,
                         'win': t['win']})
        return pd.DataFrame(rows), account, skipped

    best_cagr = 0
    combos = []
    for rh in risk_levels:
        for rm in risk_levels:
            for rl in risk_levels:
                if not (rh >= rm >= rl): continue   # high ≥ mid ≥ low
                if rh == rm == rl: continue          # не flat
                for t1, t2 in thresh_pairs:
                    sim, final, skip = simulate_grid(dfb, rh, rm, rl, t1, t2)
                    if len(sim) == 0: continue
                    cagr = (final / START_BALANCE) ** (1 / TEST_YEARS) - 1
                    mdd  = max_drawdown(sim)
                    label = f'{rh*100:.0f}→{rm*100:.0f}→{rl*100:.0f}% t≤{t1}/≤{t2}'
                    combos.append({'label': label, 'final': final, 'cagr': cagr*100,
                                   'mdd': mdd, 'trades': len(sim), 'skip': skip,
                                   'rh': rh, 'rm': rm, 'rl': rl, 't1': t1, 't2': t2})

    # Топ-5 по CAGR
    combos.sort(key=lambda x: x['cagr'], reverse=True)
    # сохраняем параметры лучшего
    for c in combos:
        c['is_best'] = False
    combos[0]['is_best'] = True
    results += combos[:5]

    # ─── Итоговая таблица ─────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  ИТОГ: $10k → ?  (2010–2026, ~{TEST_YEARS:.0f} лет, плечо 2x)")
    print(f"{'='*80}")
    print(f"  {'Вариант':<32} {'Итог':>12} {'CAGR':>7} {'MaxDD':>7} {'Сделок':>7} {'Пропущено':>10}")
    print(f"  {'-'*75}")
    for r in results:
        marker = ' ◀ ЛУЧШИЙ' if r == combos[0] else ''
        print(f"  {r['label']:<32} ${r['final']:>11,.0f}  {r['cagr']:>5.1f}%  {r['mdd']:>6.1f}%  {r['trades']:>6}  {r['skip']:>9}{marker}")

    # ─── Детальный год по лучшему ─────────────────────────────────
    best = combos[0]
    sim_best, _, _ = simulate_grid(dfb, best['rh'], best['rm'], best['rl'],
                                   best['t1'], best['t2'])
    print_year_table(sim_best, f"ЛУЧШИЙ: {best['label']}")

    # Сохраняем сигналы
    dfb.to_csv('d:/projects/trading/signals_dynamic_risk.csv', index=False)
    print(f"\n  Сохранено: signals_dynamic_risk.csv")


if __name__ == '__main__':
    main()
