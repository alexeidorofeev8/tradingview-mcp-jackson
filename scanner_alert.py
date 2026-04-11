"""
Sniper Pullback — Ежедневный сканер + Telegram алерты

Запускается автоматически через Windows Task Scheduler каждый будний день в 22:05.
Находит Type A (проторговка) и Type B (ложный пробой) сигналы.
Сохраняет в journal.csv и отправляет в Telegram.
"""
import sys, io, os, csv, json, requests, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

from datetime import datetime, date
import yfinance as yf
import pandas as pd
import numpy as np

# ─── Конфиг ──────────────────────────────────────────────────────
try:
    from telegram_config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

JOURNAL_FILE  = 'd:/projects/trading/journal.csv'
DATA_START    = '2023-01-01'
EARNINGS_CACHE = 'd:/projects/trading/earnings_cache.json'

# ─── Параметры стратегии ─────────────────────────────────────────
SMA_FAST, SMA_SLOW   = 50, 200
ATR_LEN, RS_LEN, VOL_LEN = 14, 63, 20
SR_LOOKBACK_MIN, SR_LOOKBACK_MAX = 20, 80
EXCLUDE_SECTORS = {'Real Estate', 'Utilities'}

# Type A
CONSOL_BARS  = 5
CONSOL_RANGE = 1.8
CONSOL_VOL   = 0.8

# Type B
B_BARS       = 1      # пробой должен быть вчера (не несколько дней назад)
B_MAX_BREACH = 1.5    # максимальный пробой ниже уровня (%) — апрель 2026: ужесточено с 3.0 до 1.5

# ─── Вспомогательные ─────────────────────────────────────────────
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def send_telegram(text):
    """Отправить сообщение в Telegram."""
    if not TELEGRAM_TOKEN or 'ЗДЕСЬ' in TELEGRAM_TOKEN:
        print("[Telegram] Токен не настроен — сообщение не отправлено")
        print(f"[Telegram] Текст:\n{text}")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': text,
            'parse_mode': 'HTML'
        }, timeout=10)
        if resp.status_code == 200:
            print("[Telegram] Отправлено!")
            return True
        else:
            print(f"[Telegram] Ошибка: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[Telegram] Ошибка соединения: {e}")
        return False


def load_earnings():
    if os.path.exists(EARNINGS_CACHE):
        return json.load(open(EARNINGS_CACHE, encoding='utf-8'))
    return {}


def refresh_earnings_for_symbol(sym, earnings_cache):
    """Обновляет даты отчётности через yfinance, если кэш устарел (>5 дней)."""
    today = date.today()
    entry = earnings_cache.get(sym, {})
    if isinstance(entry, dict) and entry.get('fetched'):
        if (today - pd.to_datetime(entry['fetched']).date()).days <= 5:
            return  # кэш свежий
    try:
        cal = yf.Ticker(sym).get_earnings_dates(limit=8)
        dates = [str(d.date()) for d in cal.index] if cal is not None and not cal.empty else []
    except Exception:
        dates = []
    earnings_cache[sym] = {'dates': dates, 'fetched': str(today)}


def get_earn_dates(sym, earnings_cache):
    entry = earnings_cache.get(sym, {})
    if isinstance(entry, dict):
        return entry.get('dates', [])
    return list(entry)  # обратная совместимость со старым форматом


def get_sp500():
    resp = requests.get(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        headers={'User-Agent': 'Mozilla/5.0'}, timeout=15
    )
    df = pd.read_html(io.StringIO(resp.text))[0]
    df = df[~df['GICS Sector'].isin(EXCLUDE_SECTORS)]
    syms = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    secs = dict(zip(
        df['Symbol'].str.replace('.', '-', regex=False),
        df['GICS Sector']
    ))
    return syms, secs


def save_to_journal(signals):
    """Сохраняем сигналы в journal.csv."""
    fieldnames = [
        'date_scanned', 'symbol', 'type', 'sector',
        'price', 'level', 'level_type', 'breach_pct',
        'entry', 'stop', 'target', 'rr', 'rs',
        'result', 'result_date', 'pnl_pct', 'notes'
    ]
    file_exists = os.path.exists(JOURNAL_FILE)
    with open(JOURNAL_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        today = date.today().isoformat()
        for s in signals:
            writer.writerow({
                'date_scanned': today,
                'symbol':       s['symbol'],
                'type':         s['type'],
                'sector':       s.get('sector', ''),
                'price':        s['price'],
                'level':        s['level_price'],
                'level_type':   s.get('level_type', ''),
                'breach_pct':   s.get('breach_pct', ''),
                'entry':        s['entry'],
                'stop':         s['stop'],
                'target':       s['target'],
                'rr':           s['rr'],
                'rs':           s.get('rs', ''),
                'result':       '',
                'result_date':  '',
                'pnl_pct':      '',
                'notes':        '',
            })
    print(f"[Journal] Сохранено {len(signals)} сигналов в {JOURNAL_FILE}")


# ─── Сканер ──────────────────────────────────────────────────────
def scan_symbol(sym, df, spx_close, earnings, sector):
    try:
        if len(df) < 220:
            return None

        close  = df['Close'].squeeze()
        high   = df['High'].squeeze()
        low    = df['Low'].squeeze()
        open_  = df['Open'].squeeze()
        volume = df['Volume'].squeeze()

        sma50  = close.rolling(SMA_FAST).mean()
        sma200 = close.rolling(SMA_SLOW).mean()
        atr    = calc_atr(high, low, close, ATR_LEN)
        vol_ma = volume.rolling(VOL_LEN).mean()
        spx    = spx_close.reindex(close.index, method='ffill')
        rs     = (close / close.shift(RS_LEN)) / (spx / spx.shift(RS_LEN))

        c   = close.iloc[-1]
        a   = atr.iloc[-1]
        s50 = sma50.iloc[-1]
        s200= sma200.iloc[-1]
        vm  = vol_ma.iloc[-1]
        v   = volume.iloc[-1]
        o   = open_.iloc[-1]

        # Базовые фильтры
        if c < 20 or vm < 500_000 or a/c*100 > 5: return None
        if c < s200 or s50 < s200: return None

        spx_sma200 = spx.rolling(200).mean()
        if spx.iloc[-1] < spx_sma200.iloc[-1]: return None

        rs_val = rs.iloc[-1]
        if rs_val < 1.0: return None

        # Фильтр отчётности ±7 дней (до и после)
        refresh_earnings_for_symbol(sym, earnings)
        earn_dates = get_earn_dates(sym, earnings)
        today_d = date.today()
        if any(abs((pd.to_datetime(ed).date() - today_d).days) <= 7 for ed in earn_dates):
            return None

        # Уровень SR flip
        prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        ph = prior_high.iloc[-1]
        if pd.isna(ph): return None

        # Pivot low
        pivot_low = low.rolling(15).min().iloc[-1]

        results = []

        # ── TYPE A: консолидация + прорыв ────────────────────────
        dist_sr     = (c - ph) / ph * 100
        near_sr     = abs(dist_sr) < 2.5
        dist_pivot  = (c - pivot_low) / pivot_low * 100
        near_pivot  = abs(dist_pivot) < 2.0
        dist_sma50  = (c - s50) / s50 * 100
        near_sma50  = abs(dist_sma50) < 2.0

        if near_sr or near_pivot or near_sma50:
            recent_high = high.iloc[-CONSOL_BARS:-1]
            recent_low  = low.iloc[-CONSOL_BARS:-1]
            consol_range = (recent_high.max() - recent_low.min()) / a
            tight   = consol_range < CONSOL_RANGE
            vol_ok  = volume.iloc[-CONSOL_BARS:-1].mean() > vm * CONSOL_VOL
            green   = c > o and (c - o) / a > 0.25 and v > vm * 1.1

            if green:
                level_p  = ph if near_sr else (pivot_low if near_pivot else s50)
                level_t  = 'S/R flip' if near_sr else ('Pivot Low' if near_pivot else 'SMA50')
                sl       = s50 - 1.5 * a
                risk     = c - sl
                if risk > 0:
                    res = [high.iloc[max(0,len(high)-lb):-1].max()
                           for lb in [10,20,40,60]
                           if high.iloc[max(0,len(high)-lb):-1].max() > c*1.01]
                    tp = min(res) if res else c + risk*2.5
                    rr = (tp - c) / risk
                    if rr >= 1.3:
                        results.append({
                            'symbol':     sym,
                            'type':       'A',
                            'sector':     sector,
                            'price':      round(c, 2),
                            'level_price':round(level_p, 2),
                            'level_type': level_t,
                            'tight':      tight,
                            'vol_consol': vol_ok,
                            'breach_pct': '',
                            'entry':      round(c * 0.99, 2),
                            'stop':       round(sl, 2),
                            'target':     round(tp, 2),
                            'rr':         round(rr, 2),
                            'rs':         round(rs_val, 2),
                        })

        # ── TYPE B: ложный пробой + восстановление ───────────────
        recent_lows = low.iloc[-(B_BARS+1):-1]
        min_low     = recent_lows.min()
        false_break = (min_low < ph) and (min_low >= ph * (1 - B_MAX_BREACH/100))

        if false_break:
            recovered = c > ph * 0.995
            strong    = c > o and (c - o) / a > 0.5 and v > vm * 1.2

            if strong and recovered:
                breach_pct = (ph - min_low) / ph * 100
                sl   = min_low - a * 0.5
                risk = c - sl
                if risk > 0:
                    res = [high.iloc[max(0,len(high)-lb):-1].max()
                           for lb in [10,20,40,60]
                           if high.iloc[max(0,len(high)-lb):-1].max() > c*1.01]
                    tp = min(res) if res else c + risk*2.5
                    rr = (tp - c) / risk
                    if rr >= 1.3:
                        # Не дублируем если уже есть Type A для этого символа
                        already_a = any(r['symbol'] == sym and r['type'] == 'A' for r in results)
                        if not already_a:
                            results.append({
                                'symbol':     sym,
                                'type':       'B',
                                'sector':     sector,
                                'price':      round(c, 2),
                                'level_price':round(ph, 2),
                                'level_type': 'S/R flip',
                                'tight':      False,
                                'vol_consol': False,
                                'breach_pct': round(breach_pct, 2),
                                'entry':      round(c * 0.99, 2),
                                'stop':       round(sl, 2),
                                'target':     round(tp, 2),
                                'rr':         round(rr, 2),
                                'rs':         round(rs_val, 2),
                            })

        return results if results else None

    except Exception as e:
        return None


# ─── Форматирование Telegram сообщения ───────────────────────────
def format_message(signals_a, signals_b, spy_price, spy_ma200, today_str):
    market_ok = spy_price > spy_ma200
    market_line = (
        f"SPY ${spy_price:.0f} > SMA200 ${spy_ma200:.0f} — рынок OK"
        if market_ok else
        f"⛔ SPY ${spy_price:.0f} < SMA200 ${spy_ma200:.0f} — не торгуем"
    )

    lines = [
        f"<b>Sniper Pullback — {today_str}</b>",
        f"{market_line}",
        "",
    ]

    def fmt_signal(s):
        rr_str  = f"R:R {s['rr']:.1f}"
        tight   = " <<TIGHT" if s.get('tight') else ""
        breach  = f" | пробой {s['breach_pct']}%" if s.get('breach_pct') else ""
        return (
            f"  <b>{s['symbol']}</b> ${s['price']} | уровень ${s['level_price']}"
            f"{breach} | {rr_str} | RS {s['rs']:.2f}{tight}\n"
            f"   вход ~${s['entry']} | стоп ${s['stop']} | тейк ${s['target']}"
        )

    if signals_b:
        lines.append(f"<b>TYPE B — ложный пробой ({len(signals_b)}):</b>")
        for s in signals_b[:6]:
            lines.append(fmt_signal(s))
    else:
        lines.append("TYPE B: нет сигналов сегодня")

    lines.append("")

    if signals_a:
        lines.append(f"<b>TYPE A — проторговка ({len(signals_a)}):</b>")
        for s in signals_a[:6]:
            lines.append(fmt_signal(s))
    else:
        lines.append("TYPE A: нет сигналов сегодня")

    if not signals_a and not signals_b:
        lines.append("")
        lines.append("Жди следующего дня.")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────
def main():
    today_str = datetime.now().strftime('%d %b %Y')
    print(f"=== Sniper Alert Scanner — {today_str} ===\n")

    # SPY
    spx_raw = yf.download('SPY', start=DATA_START, progress=False, auto_adjust=True)
    spx_close = spx_raw['Close'].squeeze()
    spy_price  = float(spx_close.iloc[-1])
    spy_ma200  = float(spx_close.rolling(200).mean().iloc[-1])
    market_ok  = spy_price > spy_ma200
    print(f"SPY: ${spy_price:.2f} | SMA200: ${spy_ma200:.2f} | Market OK: {market_ok}")

    if not market_ok:
        msg = format_message([], [], spy_price, spy_ma200, today_str)
        print("\nРынок ниже SMA200 — сигналы не ищем.")
        send_telegram(msg)
        return

    # Список акций
    print("Загружаем список S&P 500...")
    symbols, sectors = get_sp500()

    # Данные отчётности
    earnings = load_earnings()

    # Котировки
    print(f"Скачиваем котировки {len(symbols)} акций...")
    all_data = {}
    for i in range(0, len(symbols), 50):
        batch = symbols[i:i+50]
        raw = yf.download(batch, start=DATA_START, progress=False,
                          auto_adjust=True, group_by='ticker')
        for sym in batch:
            try:
                all_data[sym] = raw[sym].dropna() if len(batch) > 1 else raw.dropna()
            except: pass
        print(f"  [{min(i+50,len(symbols))}/{len(symbols)}]", end=' ', flush=True)
    print()

    # Сканирование
    print("\nСканируем...")
    all_signals = []
    for sym, df in all_data.items():
        res = scan_symbol(sym, df, spx_close, earnings, sectors.get(sym, ''))
        if res:
            all_signals.extend(res)

    signals_a = sorted([s for s in all_signals if s['type'] == 'A'],
                       key=lambda x: -x['rr'])
    signals_b = sorted([s for s in all_signals if s['type'] == 'B'],
                       key=lambda x: -x['rr'])

    print(f"\nType A: {len(signals_a)} сигналов")
    print(f"Type B: {len(signals_b)} сигналов")

    # Печать в консоль
    for s in signals_b[:5]:
        print(f"  B | {s['symbol']:<6} ${s['price']} | уровень ${s['level_price']} "
              f"| пробой {s['breach_pct']}% | R:R {s['rr']} | RS {s['rs']}")
    for s in signals_a[:5]:
        print(f"  A | {s['symbol']:<6} ${s['price']} | {s['level_type']:<10} "
              f"| R:R {s['rr']} | RS {s['rs']}{' <<TIGHT' if s['tight'] else ''}")

    # Telegram
    msg = format_message(signals_a, signals_b, spy_price, spy_ma200, today_str)
    send_telegram(msg)

    # Журнал
    all_to_save = signals_b[:8] + signals_a[:8]
    if all_to_save:
        save_to_journal(all_to_save)

    # Сохраняем обновлённый кэш отчётности
    with open(EARNINGS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(earnings, f, indent=2)
    print(f"[Earnings] Кэш обновлён — {len(earnings)} символов.")

    print("\nГотово.")


if __name__ == '__main__':
    main()
