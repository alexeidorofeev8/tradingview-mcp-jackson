"""
Sniper Pullback — Ежедневный сканер + Telegram алерты

Запускается автоматически через Windows Task Scheduler каждый будний день в 22:05.
Находит Type B (ложный пробой + восстановление) сигналы.
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

# Type B
B_MAX_BREACH = 1.5    # максимальный пробой ниже уровня (%)
B_STOP_BUF   = 0.5    # ATR буфер ниже минимума пробоя

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

        c      = close.iloc[-1]
        a      = atr.iloc[-1]
        s50    = sma50.iloc[-1]
        s200   = sma200.iloc[-1]
        vm     = vol_ma.iloc[-1]
        v      = volume.iloc[-1]
        o      = open_.iloc[-1]
        yday_l = low.iloc[-2]   # вчерашний минимум (день пробоя)

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

        # Уровни
        prior_high = high.shift(SR_LOOKBACK_MIN).rolling(SR_LOOKBACK_MAX - SR_LOOKBACK_MIN).max()
        local_min  = low.rolling(15).min().shift(1)

        ph = prior_high.iloc[-1]
        lm = local_min.iloc[-1]

        if pd.isna(ph) and pd.isna(lm):
            return None

        # Сила восстановления сегодня (обязательный фильтр)
        strong = c > o and (c - o) / a > 0.5 and v > vm * 1.2
        if not strong:
            return None

        # ── TYPE B: ложный пробой + восстановление ───────────────
        level_type = None
        level_val  = None

        # Ветка 1: prior_high (S/R flip)
        if not pd.isna(ph):
            broke_ph     = (yday_l < ph) and (yday_l >= ph * (1 - B_MAX_BREACH / 100))
            recovered_ph = (c > ph) and (yday_l < ph)
            if broke_ph and recovered_ph:
                level_type = 'S/R flip'
                level_val  = ph

        # Ветка 2: local_min (только если prior_high не сработал)
        if level_type is None and not pd.isna(lm):
            broke_lm     = (yday_l < lm) and (yday_l >= lm * (1 - B_MAX_BREACH / 100))
            recovered_lm = (c > lm) and (yday_l < lm)
            if broke_lm and recovered_lm:
                level_type = 'Local min'
                level_val  = lm

        if level_type is None:
            return None

        # Стоп и цель
        breakdown_low = min(yday_l, low.iloc[-3]) if len(low) > 2 else yday_l
        sl    = breakdown_low - a * B_STOP_BUF
        entry = c * 0.99
        risk  = entry - sl

        if risk <= 0:
            return None

        risk_pct = risk / entry * 100
        if risk_pct > 15:
            return None

        tp = entry + risk * 1.5
        rr = (tp - entry) / risk

        if rr < 1.0:
            return None

        breach_pct = (level_val - yday_l) / level_val * 100

        return {
            'symbol':     sym,
            'type':       'B',
            'sector':     sector,
            'price':      round(c, 2),
            'level_price':round(level_val, 2),
            'level_type': level_type,
            'breach_pct': round(breach_pct, 2),
            'entry':      round(entry, 2),
            'stop':       round(sl, 2),
            'target':     round(tp, 2),
            'rr':         round(rr, 2),
            'rs':         round(rs_val, 2),
        }

    except Exception:
        return None


# ─── Форматирование Telegram сообщения ───────────────────────────
def format_message(signals_b, spy_price, spy_ma200, today_str):
    market_ok = spy_price > spy_ma200
    market_line = (
        f"SPY ${spy_price:.0f} > SMA200 ${spy_ma200:.0f} — рынок OK"
        if market_ok else
        f"⛔ SPY ${spy_price:.0f} < SMA200 ${spy_ma200:.0f} — не торгуем"
    )

    lines = [
        f"<b>Sniper Pullback — {today_str}</b>",
        market_line,
        "",
    ]

    def fmt_signal(s):
        breach = f" | пробой {s['breach_pct']}%" if s.get('breach_pct') else ""
        return (
            f"  <b>{s['symbol']}</b> ${s['price']} | {s['level_type']} ${s['level_price']}"
            f"{breach} | R:R {s['rr']:.1f} | RS {s['rs']:.2f}\n"
            f"   вход ~${s['entry']} | стоп ${s['stop']} | тейк ${s['target']}"
        )

    if signals_b:
        lines.append(f"<b>TYPE B — ложный пробой ({len(signals_b)}):</b>")
        for s in signals_b[:6]:
            lines.append(fmt_signal(s))
    else:
        lines.append("Сегодня сигналов нет.")
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
        msg = format_message([], spy_price, spy_ma200, today_str)
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
    signals_b = []
    for sym, df in all_data.items():
        res = scan_symbol(sym, df, spx_close, earnings, sectors.get(sym, ''))
        if res:
            signals_b.append(res)

    signals_b.sort(key=lambda x: -x['rr'])

    print(f"\nType B: {len(signals_b)} сигналов")
    for s in signals_b[:5]:
        print(f"  B | {s['symbol']:<6} ${s['price']} | {s['level_type']:<10} ${s['level_price']} "
              f"| пробой {s['breach_pct']}% | R:R {s['rr']} | RS {s['rs']}")

    # Telegram
    msg = format_message(signals_b, spy_price, spy_ma200, today_str)
    send_telegram(msg)

    # Журнал
    if signals_b:
        save_to_journal(signals_b[:8])

    # Сохраняем обновлённый кэш отчётности
    with open(EARNINGS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(earnings, f, indent=2)
    print(f"[Earnings] Кэш обновлён — {len(earnings)} символов.")

    print("\nГотово.")


if __name__ == '__main__':
    main()
