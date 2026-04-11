"""
Ресёрч: как выглядит часовой вход на последних сигналах v5
Задача: понять что происходит на часовике ПЕРЕД и ПОСЛЕ дневного сигнала.

Вопросы:
1. Есть ли консолидация на часовике ПЕРЕД дневным сигналом?
2. Когда появляется первая сильная зелёная свеча на часовике ПОСЛЕ сигнала?
3. Насколько лучше вход по часовику vs по дневному закрытию?
4. Торговли что НЕ получили подтверждения на часовике — они чаще убыточны?
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── Параметры ──────────────────────────────────────────────────
HOURLY_CONSOL_BARS   = 6    # баров для консолидации на часовике
HOURLY_CONSOL_RANGE  = 1.5  # узко = < 1.5 ATR
HOURLY_BODY_MIN      = 0.3  # тело входной свечи > 0.3 ATR
HOURLY_VOL_MIN       = 1.15 # объём > 115% среднего
HOURLY_LOOK_AHEAD    = 48   # часов после дневного сигнала — ищем подтверждение
DAILY_DISC           = 0.01 # текущая симуляция часового входа: 1%

def calc_atr_h(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def analyze_hourly(sym, signal_date, daily_entry, daily_sl, daily_pnl, daily_win):
    """Анализирует часовой график вокруг точки дневного входа"""
    try:
        sig_dt = pd.to_datetime(signal_date)

        # Скачиваем часовые данные: 5 дней до и 5 после сигнала
        start = (sig_dt - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
        end   = (sig_dt + pd.Timedelta(days=6)).strftime('%Y-%m-%d')
        raw = yf.download(sym, start=start, end=end,
                          interval='1h', progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 20:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        close  = raw['Close'].squeeze()
        high   = raw['High'].squeeze()
        low    = raw['Low'].squeeze()
        open_  = raw['Open'].squeeze()
        volume = raw['Volume'].squeeze()

        atr_h  = calc_atr_h(high, low, close, 14)
        vol_ma = volume.rolling(20).mean()

        # Нормализуем индекс (убираем timezone если есть)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
            high.index  = high.index.tz_localize(None)
            low.index   = low.index.tz_localize(None)
            open_.index = open_.index.tz_localize(None)
            volume.index = volume.index.tz_localize(None)
            atr_h.index = atr_h.index.tz_localize(None)
            vol_ma.index = vol_ma.index.tz_localize(None)

        # Найти часовые бары ДО дневного сигнала (в тот же день и накануне)
        before_mask = close.index < sig_dt.replace(hour=22, minute=0)
        bars_before = close[before_mask].tail(24)  # последние 24 часа до сигнала

        # ── 1. Консолидация на часовике перед сигналом ──────────
        pre_consol = None
        if len(bars_before) >= HOURLY_CONSOL_BARS:
            pre_high = high[before_mask].tail(HOURLY_CONSOL_BARS).max()
            pre_low  = low[before_mask].tail(HOURLY_CONSOL_BARS).min()
            pre_atr  = atr_h[before_mask].tail(1).values[0] if len(atr_h[before_mask]) > 0 else 1
            pre_consol = (pre_high - pre_low) / pre_atr if pre_atr > 0 else None

        # ── 2. Первая сильная зелёная свеча ПОСЛЕ сигнала ────────
        after_mask = close.index >= sig_dt.replace(hour=0)
        bars_after = close[after_mask].head(HOURLY_LOOK_AHEAD)

        hourly_entry  = None
        hourly_entry_dt = None
        hours_to_entry  = None

        if len(bars_after) > 0:
            for i in range(len(bars_after)):
                bar_idx = bars_after.index[i]
                if bar_idx not in atr_h.index: continue
                h_atr = atr_h.loc[bar_idx]
                h_vol = vol_ma.loc[bar_idx] if bar_idx in vol_ma.index else 0
                h_vol_raw = volume.loc[bar_idx] if bar_idx in volume.index else 0
                h_close = close.loc[bar_idx]
                h_open  = open_.loc[bar_idx] if bar_idx in open_.index else h_close

                if h_atr <= 0 or h_vol <= 0: continue
                body = (h_close - h_open) / h_atr
                vol_ratio = h_vol_raw / h_vol if h_vol > 0 else 0

                # Сильная зелёная свеча на часовике
                if body > HOURLY_BODY_MIN and vol_ratio > HOURLY_VOL_MIN and h_close > h_open:
                    hourly_entry    = float(h_close)
                    hourly_entry_dt = bar_idx
                    # Считаем часы от конца торгового дня сигнала (22:00)
                    sig_close_time = sig_dt.replace(hour=22)
                    hours_to_entry = max(0, (bar_idx - sig_close_time).total_seconds() / 3600)
                    break

        # ── 3. Метрики улучшения ─────────────────────────────────
        improvement_pct = None
        rr_daily  = None
        rr_hourly = None
        sl_hourly = None

        if hourly_entry is not None:
            improvement_pct = (daily_entry - hourly_entry) / daily_entry * 100
            risk_daily  = daily_entry - daily_sl
            risk_hourly = hourly_entry - daily_sl  # тот же стоп

            # Ищем минимум перед входом как часовой стоп
            before_entry = low[after_mask][:bars_after.index.get_loc(hourly_entry_dt)+1] if hourly_entry_dt in bars_after.index else pd.Series()
            if len(before_entry) >= 3:
                sl_hourly = float(before_entry.tail(6).min()) * 0.998
                risk_hourly_tight = hourly_entry - sl_hourly
                if risk_hourly_tight > 0:
                    # Тейк берём как примерно R:R от дневного
                    daily_reward = daily_entry * 0.08  # ~8% тейк условно
                    rr_hourly = daily_reward / risk_hourly_tight

        return {
            'symbol':           sym,
            'date':             str(signal_date)[:10],
            'daily_entry':      round(daily_entry, 2),
            'daily_sl':         round(daily_sl, 2),
            'daily_pnl':        round(daily_pnl, 2),
            'daily_win':        daily_win,
            'pre_consol_atr':   round(pre_consol, 2) if pre_consol else None,
            'pre_tight':        pre_consol < HOURLY_CONSOL_RANGE if pre_consol else False,
            'hourly_confirmed': hourly_entry is not None,
            'hours_to_confirm': round(hours_to_entry, 1) if hours_to_entry is not None else None,
            'hourly_entry':     round(hourly_entry, 2) if hourly_entry else None,
            'entry_improvement': round(improvement_pct, 2) if improvement_pct is not None else None,
            'sl_hourly':        round(sl_hourly, 2) if sl_hourly else None,
        }
    except Exception as e:
        return None


def main():
    print("=== Ресёрч: Часовой вход на последних сигналах v5 ===\n")

    df = pd.read_csv('signals_v5.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Берём последние 6 месяцев — достаточно для анализа
    recent = df[df['date'] >= '2025-10-01'].sort_values('date').copy()
    print(f"Сигналов для анализа: {len(recent)}\n")

    results = []
    for i, (_, row) in enumerate(recent.iterrows()):
        r = analyze_hourly(
            row['symbol'], row['date'],
            row['entry'], row['sl'],
            row['pnl_pct'], row['win']
        )
        if r:
            results.append(r)
        if (i+1) % 10 == 0:
            print(f"  Обработано {i+1}/{len(recent)}...", flush=True)

    if not results:
        print("Нет данных."); return

    res = pd.DataFrame(results)
    confirmed   = res[res['hourly_confirmed'] == True]
    unconfirmed = res[res['hourly_confirmed'] == False]

    print(f"\n{'='*70}")
    print(f"  ИТОГИ АНАЛИЗА")
    print(f"{'='*70}")
    print(f"\n  Всего проанализировано:   {len(res)}")
    print(f"  Получили подтверждение:   {len(confirmed)} ({len(confirmed)/len(res)*100:.0f}%)")
    print(f"  Не получили:              {len(unconfirmed)} ({len(unconfirmed)/len(res)*100:.0f}%)")

    # ── Win rate: подтверждённые vs неподтверждённые ──────────
    print(f"\n{'─'*70}")
    print(f"  WIN RATE: подтверждённые vs неподтверждённые")
    print(f"{'─'*70}")
    if len(confirmed) > 0:
        wr_c = confirmed['daily_win'].mean()*100
        avg_c = confirmed['daily_pnl'].mean()
        print(f"  С подтверждением часовика:   Win {wr_c:.1f}%  Avg P&L {avg_c:+.2f}%  ({len(confirmed)} сд)")
    if len(unconfirmed) > 0:
        wr_u = unconfirmed['daily_win'].mean()*100
        avg_u = unconfirmed['daily_pnl'].mean()
        print(f"  Без подтверждения:           Win {wr_u:.1f}%  Avg P&L {avg_u:+.2f}%  ({len(unconfirmed)} сд)")

    # ── Консолидация перед сигналом ───────────────────────────
    has_consol = res[res['pre_consol_atr'].notna()]
    tight      = has_consol[has_consol['pre_tight'] == True]
    not_tight  = has_consol[has_consol['pre_tight'] == False]

    print(f"\n{'─'*70}")
    print(f"  КОНСОЛИДАЦИЯ ПЕРЕД СИГНАЛОМ (последние {HOURLY_CONSOL_BARS} часовых баров)")
    print(f"{'─'*70}")
    if len(tight) > 0:
        print(f"  Узкая (<{HOURLY_CONSOL_RANGE} ATR):   Win {tight['daily_win'].mean()*100:.1f}%  Avg {tight['daily_pnl'].mean():+.2f}%  ({len(tight)} сд)")
    if len(not_tight) > 0:
        print(f"  Широкая:          Win {not_tight['daily_win'].mean()*100:.1f}%  Avg {not_tight['daily_pnl'].mean():+.2f}%  ({len(not_tight)} сд)")

    # ── Время до подтверждения ────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  КОГДА ПОЯВЛЯЕТСЯ ПОДТВЕРЖДЕНИЕ НА ЧАСОВИКЕ")
    print(f"{'─'*70}")
    conf_data = confirmed[confirmed['hours_to_confirm'].notna()]
    if len(conf_data) > 0:
        bins = [0, 4, 12, 24, 48, 999]
        labels = ['0–4ч (тот же день)', '4–12ч', '12–24ч', '24–48ч', '48ч+']
        conf_data = conf_data.copy()
        conf_data['bucket'] = pd.cut(conf_data['hours_to_confirm'], bins=bins, labels=labels)
        for label, g in conf_data.groupby('bucket', observed=True):
            wr = g['daily_win'].mean()*100
            print(f"  {label:<22}  {len(g):>3} сд  Win {wr:.0f}%  Avg {g['daily_pnl'].mean():+.2f}%")

    # ── Улучшение цены входа ──────────────────────────────────
    has_impr = confirmed[confirmed['entry_improvement'].notna()]
    if len(has_impr) > 0:
        print(f"\n{'─'*70}")
        print(f"  УЛУЧШЕНИЕ ЦЕНЫ ВХОДА (часовик vs дневное закрытие)")
        print(f"{'─'*70}")
        avg_impr = has_impr['entry_improvement'].mean()
        med_impr = has_impr['entry_improvement'].median()
        better   = (has_impr['entry_improvement'] > 0).sum()
        worse    = (has_impr['entry_improvement'] <= 0).sum()
        print(f"  Среднее улучшение:  {avg_impr:+.2f}%")
        print(f"  Медиана:            {med_impr:+.2f}%")
        print(f"  Лучше дневного:     {better} из {len(has_impr)}")
        print(f"  Хуже дневного:      {worse} из {len(has_impr)}")
        print(f"  (текущая симуляция v5 использует -1.00% дисконт)")

    # ── Детальная таблица ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  ДЕТАЛИ (первые 30 сигналов)")
    print(f"{'─'*70}")
    print(f"  {'Тикер':<6} {'Дата':<12} {'P&L%':>6} {'W':<2} {'Консол':>7} {'Подтв':>6} {'Часов':>6} {'Улучш':>7}")
    print(f"  {'-'*65}")
    for _, r in res.head(30).iterrows():
        consol_str = f"{r['pre_consol_atr']:.1f}" if r['pre_consol_atr'] else "  ?"
        tight_mark = "✓" if r['pre_tight'] else " "
        conf_str   = "да" if r['hourly_confirmed'] else "нет"
        hours_str  = f"{r['hours_to_confirm']:.0f}" if r['hours_to_confirm'] is not None else "  —"
        impr_str   = f"{r['entry_improvement']:+.1f}%" if r['entry_improvement'] is not None else "   —"
        win_str    = "✓" if r['daily_win'] else "✗"
        print(f"  {r['symbol']:<6} {r['date']:<12} {r['daily_pnl']:>+5.1f}% {win_str:<2} {consol_str:>5}{tight_mark}  {conf_str:>5}  {hours_str:>5}  {impr_str:>6}")

    # ── Вывод: что менять в стратегии ────────────────────────
    print(f"\n{'='*70}")
    print(f"  ВЫВОДЫ ДЛЯ СТРАТЕГИИ")
    print(f"{'='*70}")

    if len(confirmed) > 0 and len(unconfirmed) > 0:
        wr_c = confirmed['daily_win'].mean()*100
        wr_u = unconfirmed['daily_win'].mean()*100
        diff  = wr_c - wr_u

        print(f"\n  1. ФИЛЬТР ПОДТВЕРЖДЕНИЯ:")
        if diff > 5:
            print(f"     Подтверждённые сделки Win {wr_c:.0f}% vs неподтверждённые {wr_u:.0f}%")
            print(f"     → ДОБАВИТЬ: не входить если нет сильной свечи на часовике за 48ч")
        else:
            print(f"     Разница win rate небольшая ({diff:+.1f}%)")
            print(f"     → Часовой фильтр даёт небольшое улучшение")

    if len(tight) > 0 and len(not_tight) > 0:
        wr_t  = tight['daily_win'].mean()*100
        wr_nt = not_tight['daily_win'].mean()*100
        diff_t = wr_t - wr_nt
        print(f"\n  2. КОНСОЛИДАЦИЯ НА ЧАСОВИКЕ ПЕРЕД СИГНАЛОМ:")
        if diff_t > 5:
            print(f"     С консолидацией: {wr_t:.0f}% vs без: {wr_nt:.0f}%")
            print(f"     → ДОБАВИТЬ: требовать узкую консолидацию ({HOURLY_CONSOL_BARS} ч < {HOURLY_CONSOL_RANGE} ATR) перед входом")
        else:
            print(f"     Разница небольшая ({diff_t:+.1f}%)")

    if len(has_impr) > 0:
        avg_impr = has_impr['entry_improvement'].mean()
        print(f"\n  3. ЦЕНА ВХОДА:")
        print(f"     Реальное улучшение часового входа: {avg_impr:+.2f}%")
        print(f"     Текущий дисконт в v5: -1.00%")
        if abs(avg_impr - (-1.0)) > 0.3:
            new_disc = abs(avg_impr)
            print(f"     → ОБНОВИТЬ дисконт до -{new_disc:.1f}% в backtest_v5.py")

    res.to_csv('d:/projects/trading/hourly_research.csv', index=False)
    print(f"\n  Сохранено: hourly_research.csv")

if __name__ == '__main__':
    main()
