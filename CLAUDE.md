# TradingView MCP — Claude Instructions

68 tools for reading and controlling a live TradingView Desktop chart via CDP (port 9222).

## Decision Tree — Which Tool When

### "What's on my chart right now?"
1. `chart_get_state` → symbol, timeframe, chart type, list of all indicators with entity IDs
2. `data_get_study_values` → current numeric values from all visible indicators (RSI, MACD, BBands, EMAs, etc.)
3. `quote_get` → real-time price, OHLC, volume for current symbol

### "What levels/lines/labels are showing?"
Custom Pine indicators draw with `line.new()`, `label.new()`, `table.new()`, `box.new()`. These are invisible to normal data tools. Use:

1. `data_get_pine_lines` → horizontal price levels drawn by indicators (deduplicated, sorted high→low)
2. `data_get_pine_labels` → text annotations with prices (e.g., "PDH 24550", "Bias Long ✓")
3. `data_get_pine_tables` → table data formatted as rows (e.g., session stats, analytics dashboards)
4. `data_get_pine_boxes` → price zones / ranges as {high, low} pairs

Use `study_filter` parameter to target a specific indicator by name substring (e.g., `study_filter: "Profiler"`).

### "Give me price data"
- `data_get_ohlcv` with `summary: true` → compact stats (high, low, range, change%, avg volume, last 5 bars)
- `data_get_ohlcv` without summary → all bars (use `count` to limit, default 100)
- `quote_get` → single latest price snapshot

### "Analyze my chart" (full report workflow)
1. `quote_get` → current price
2. `data_get_study_values` → all indicator readings
3. `data_get_pine_lines` → key price levels from custom indicators
4. `data_get_pine_labels` → labeled levels with context (e.g., "Settlement", "ASN O/U")
5. `data_get_pine_tables` → session stats, analytics tables
6. `data_get_ohlcv` with `summary: true` → price action summary
7. `capture_screenshot` → visual confirmation

### "Change the chart"
- `chart_set_symbol` → switch ticker (e.g., "AAPL", "ES1!", "NYMEX:CL1!")
- `chart_set_timeframe` → switch resolution (e.g., "1", "5", "15", "60", "D", "W")
- `chart_set_type` → switch chart style (Candles, HeikinAshi, Line, Area, Renko, etc.)
- `chart_manage_indicator` → add or remove studies (use full name: "Relative Strength Index", not "RSI")
- `chart_scroll_to_date` → jump to a date (ISO format: "2025-01-15")
- `chart_set_visible_range` → zoom to exact date range (unix timestamps)

### "Work on Pine Script"
1. `pine_set_source` → inject code into editor
2. `pine_smart_compile` → compile with auto-detection + error check
3. `pine_get_errors` → read compilation errors
4. `pine_get_console` → read log.info() output
5. `pine_get_source` → read current code back (WARNING: can be very large for complex scripts)
6. `pine_save` → save to TradingView cloud
7. `pine_new` → create blank indicator/strategy/library
8. `pine_open` → load a saved script by name

### "Practice trading with replay"
1. `replay_start` with `date: "2025-03-01"` → enter replay mode
2. `replay_step` → advance one bar
3. `replay_autoplay` → auto-advance (set speed with `speed` param in ms)
4. `replay_trade` with `action: "buy"/"sell"/"close"` → execute trades
5. `replay_status` → check position, P&L, current date
6. `replay_stop` → return to realtime

### "Screen multiple symbols"
- `batch_run` with `symbols: ["ES1!", "NQ1!", "YM1!"]` and `action: "screenshot"` or `"get_ohlcv"`

### "Draw on the chart"
- `draw_shape` → horizontal_line, trend_line, rectangle, text (pass point + optional point2)
- `draw_list` → see what's drawn
- `draw_remove_one` → remove by ID
- `draw_clear` → remove all

### "Manage alerts"
- `alert_create` → set price alert (condition: "crossing", "greater_than", "less_than")
- `alert_list` → view active alerts
- `alert_delete` → remove alerts

### "Navigate the UI"
- `ui_open_panel` → open/close pine-editor, strategy-tester, watchlist, alerts, trading
- `ui_click` → click buttons by aria-label, text, or data-name
- `layout_switch` → load a saved layout by name
- `ui_fullscreen` → toggle fullscreen
- `capture_screenshot` → take a screenshot (regions: "full", "chart", "strategy_tester")

### "TradingView isn't running"
- `tv_launch` → auto-detect and launch TradingView with CDP on Mac/Win/Linux
- `tv_health_check` → verify connection is working

## Context Management Rules

These tools can return large payloads. Follow these rules to avoid context bloat:

1. **Always use `summary: true` on `data_get_ohlcv`** unless you specifically need individual bars
2. **Always use `study_filter`** on pine tools when you know which indicator you want — don't scan all studies unnecessarily
3. **Never use `verbose: true`** on pine tools unless the user specifically asks for raw drawing data with IDs/colors
4. **Avoid calling `pine_get_source`** on complex scripts — it can return 200KB+. Only read if you need to edit the code.
5. **Avoid calling `data_get_indicator`** on protected/encrypted indicators — their inputs are encoded blobs. Use `data_get_study_values` instead for current values.
6. **Use `capture_screenshot`** for visual context instead of pulling large datasets — a screenshot is ~300KB but gives you the full visual picture
7. **Call `chart_get_state` once** at the start to get entity IDs, then reference them — don't re-call repeatedly
8. **Cap your OHLCV requests** — `count: 20` for quick analysis, `count: 100` for deeper work, `count: 500` only when specifically needed

### Output Size Estimates (compact mode)
| Tool | Typical Output |
|------|---------------|
| `quote_get` | ~200 bytes |
| `data_get_study_values` | ~500 bytes (all indicators) |
| `data_get_pine_lines` | ~1-3 KB per study (deduplicated levels) |
| `data_get_pine_labels` | ~2-5 KB per study (capped at 50) |
| `data_get_pine_tables` | ~1-4 KB per study (formatted rows) |
| `data_get_pine_boxes` | ~1-2 KB per study (deduplicated zones) |
| `data_get_ohlcv` (summary) | ~500 bytes |
| `data_get_ohlcv` (100 bars) | ~8 KB |
| `capture_screenshot` | ~300 bytes (returns file path, not image data) |

## Tool Conventions

- All tools return `{ success: true/false, ... }`
- Entity IDs (from `chart_get_state`) are session-specific — don't cache across sessions
- Pine indicators must be **visible** on chart for pine graphics tools to read their data
- `chart_manage_indicator` requires **full indicator names**: "Relative Strength Index" not "RSI", "Moving Average Exponential" not "EMA", "Bollinger Bands" not "BB"
- Screenshots save to `screenshots/` directory with timestamps
- OHLCV capped at 500 bars, trades at 20 per request
- Pine labels capped at 50 per study by default (pass `max_labels` to override)

## Architecture

```
Claude Code ←→ MCP Server (stdio) ←→ CDP (localhost:9222) ←→ TradingView Desktop (Electron)
```

Pine graphics path: `study._graphics._primitivesCollection.dwglines.get('lines').get(false)._primitivesDataById`

---

# Пользователь и проект

## Профиль

- **Имя**: Алексей (Alexei Dorofeev), Берлин
- **Брокер**: Interactive Brokers (IBKR)
- **Счёт**: ~$10,000 (апрель 2026), реинвестирует всю прибыль. Всегда используется маржа.
- **Цель**: swing trading NYSE/S&P 500, 2-3 сделки/неделю, не более 2 часов/день
- **Стиль**: лимитные ордера, чёткие правила, не дейтрейдер
- **Язык**: общается на русском. Стратегические документы — на русском.
- **Инструменты**: TradingView (чарты), IBKR (исполнение), Python (сканер/бэктест)

---

# Стратегия: Sniper Pullback Type B (ТОЛЬКО Type B)

Type A архивирован. Использовать ТОЛЬКО Type B.

## Суть Type B
Акция пробивает уровень S/R ВНИЗ на 0–1.5% вчера → сегодня сильная зелёная свеча восстановления выше уровня.
- Пробой должен быть ровно вчера (bars=1), медленный слайд 2+ дня — пропустить
- Бэктест 2016–2026 без ограничений: $10k → $631k, CAGR 49.7%, WR 58.6%, 807 сделок
- Бэктест 2016–2026 макс. 4 позиции: $4k → $49k, CAGR 27.8%, WR 58.6%, 536 сделок
- Стоп: под минимумом ложного пробоя − 0.5×ATR (avg ~4.2% от входа)
- Средний R:R: 2.53

## Вход — ОБЯЗАТЕЛЬНО внутри дня
- Бэктест предполагает вход по close×0.99 (−1% от закрытия дня сигнала)
- Вход следующим утром на открытии → CAGR падает с 50% до 7% за 10 лет
- Причина: тот же стоп + выше вход = больший риск в % = меньшая позиция = компаундинг разрушается
- Пример: вход $99 vs $101, SL $96 → риск 3% vs 5% → позиция на 40% меньше
- NYSE открывается в 15:30 по берлинскому времени — нужно смотреть на экран в этот момент

## Никаких дополнительных фильтров
- Объёмный фильтр (≥1.5× avg): WR растёт 65%→72%, но суммарный PnL падает 130→97 (данные 2025)
- За 5 лет: без фильтра $51k vs с фильтром $21k
- Вывод: брать ВСЕ сигналы сканера, без дополнительных intraday-фильтров

## Ключевые фильтры (встроены в сканер)
1. SPY > SMA200 дневной (рыночный режим — если нет, не торговать)
2. Акция: цена > SMA200, SMA50 > SMA200, объём > 500k, цена > $20, RS > SPY 63d
3. Уровень: S/R флип (приоритет), local min
4. Нет отчётов в течение 7 дней
5. **Динамический риск** (оптимальный по перебору): 0–3 позиции открыто → 2%, 4+ → 1%. Бэктест 2010–2026: CAGR 60.7%, $10k → $22.9M (vs flat 1% → $2.0M). Плечо IBKR: интрадей 4x, овернайт 2x.
6. НЕТ секторного фильтра — проверено: CAGR 28.3% (все сектора) vs 25.3% (топ-5 секторов)

## Файлы проекта
- [backtest_type_b.py](backtest_type_b.py) — бэктест Type B
- [scanner.py](scanner.py) — живой сканер
- [STRATEGY.md](STRATEGY.md) — полная документация на русском

---

# Результаты исследований

## Часовой вход (research_hourly.py, 82 сигнала)
- Среднее улучшение входа: −1.11% ниже дневного закрытия (стратегия использует −1.00% — валидно)
- Tight hourly consolidation (< 1.5 ATR, 6 баров перед входом): WR **66.7%**
- Wide consolidation: WR **57.5%**
- Решение: часовая консолидация — ВИЗУАЛЬНОЕ подтверждение только, НЕ автоматический фильтр
- Причина: обязательный фильтр выбрасывал 75% сделок, итог по счёту хуже

## Walk-Forward (backtest_v6.py) — отклонён
- 3 года обучение → 6 месяцев тест: $4k → $4.2k (+5%)
- Причина: медвежий рынок 2022 в обучающем окне → почти ноль квалифицированных акций в тесте
- Решение: отклонён, оставляем v5

## Часовой паттерн поддержки (Вайкофф)
- Консолидация на дневном уровне на часовике = абсорбция по Вайкоффу
- Практика: ждать первую сильную зелёную 1H свечу с объёмом, вход на закрытии
- Часовой стоп → лучше R:R чем дневной

---

# Правила поведения Claude

1. **Всегда обновлять STRATEGY.md на русском** при изменении стратегии
2. **Не предлагать скользящий стоп** — бэктест показал худший результат; фиксированный TP лучше
3. **Не делать часовую консолидацию обязательным фильтром** — только визуальное подтверждение
4. **Не использовать английские термины** без объяснения на русском (walk-forward = объяснить что это)
5. **Не двигать стоп дальше от цены** — только в безубыток или ближе
6. **Intraday вход обязателен** — никогда не предлагать вход следующим утром на открытии
7. **Никаких секторных фильтров** — проверено, ухудшает результат
8. **Только Type B** — Type A не упоминать и не предлагать
9. **Никаких дополнительных intraday-фильтров** — брать все сигналы сканера

---

# Целевые расчёты

Плечо всегда используется → реалистичный сценарий = без лимита позиций (CAGR ~50%).

| Цель | CAGR | Нужно сейчас |
|------|------|-------------|
| $1,000,000 через 10 лет | 50% (реалистичный, плечо есть) | ~$17,000 |
| $1,000,000 через 10 лет | 28% (консервативный, без плеча) | ~$85,000 |

При текущем счёте $10k → через 10 лет: ~$576k (50% CAGR). Бэктест именно с $10k стартовал ($10k → $631k 2016–2026).
