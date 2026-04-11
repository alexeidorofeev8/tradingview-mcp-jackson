/**
 * swing-backtest.js — Historical backtest of NYSE Swing Strategy
 *
 * Gets up to 500 daily bars from TradingView (~2 years of data).
 * Simulates bar-by-bar: signal at close of bar N → entry at open of bar N+1.
 * Exit: TP1 (2R), stop loss (1R), or time stop (15 bars).
 * Note: Earnings blackout skipped (no historical earnings calendar).
 *
 * Usage:
 *   node swing-backtest.js                     # backtest all watchlist
 *   node swing-backtest.js --symbol NYSE:JPM   # single ticker
 *   node swing-backtest.js --no-market-filter  # ignore SPY/VIX filter
 */

import { setSymbol, setTimeframe, getState } from './src/core/chart.js';
import { getOhlcv } from './src/core/data.js';
import { disconnect } from './src/connection.js';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dir = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH  = join(__dir, 'rules-swing.json');
const RESULTS_PATH = join(__dir, 'swing-backtest-results.json');

const argv = process.argv.slice(2);
const args = {};
for (let i = 0; i < argv.length; i++) {
  if (argv[i].startsWith('--')) {
    const key = argv[i].slice(2);
    const val  = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : true;
    args[key]  = val;
    if (val !== true) i++;
  }
}

const SYMBOL_FILTER        = args.symbol || null;
const SKIP_MARKET_FILTER   = !!args['no-market-filter'];
const CFG  = JSON.parse(readFileSync(CONFIG_PATH, 'utf8'));
const P    = CFG.parameters;
const RISK = CFG.risk;
const WATCHLIST      = SYMBOL_FILTER ? [SYMBOL_FILTER] : CFG.watchlist;
const TIME_STOP_BARS = 15;   // close trade if no exit after N bars
const BAR_COUNT      = 500;  // max daily bars from TradingView (~2 years)

// ── Indicator math (full-series versions) ─────────────────────────────────────

function emaFull(values, period) {
  if (values.length < period) return values.map(() => NaN);
  const k   = 2 / (period + 1);
  const out = new Array(period - 1).fill(NaN);
  let e     = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  out.push(e);
  for (let i = period; i < values.length; i++) {
    e = values[i] * k + e * (1 - k);
    out.push(e);
  }
  return out;
}

function smaLast(values, period) {
  if (values.length < period) return NaN;
  return values.slice(-period).reduce((a, b) => a + b, 0) / period;
}

function williamsRAt(bars, i, period) {
  if (i < period - 1) return NaN;
  const sl = bars.slice(i - period + 1, i + 1);
  const hh = Math.max(...sl.map(b => b.high));
  const ll  = Math.min(...sl.map(b => b.low));
  if (hh === ll) return -50;
  return ((hh - bars[i].close) / (hh - ll)) * -100;
}

function rsiAt(closes, i, period) {
  // Wilder RSI using last 40 bars (enough warmup for period 2)
  const window = closes.slice(Math.max(0, i - 40), i + 1);
  if (window.length < period + 2) return NaN;
  let gains = 0, losses = 0;
  for (let j = 1; j <= period; j++) {
    const d = window[j] - window[j - 1];
    if (d >= 0) gains += d; else losses -= d;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  for (let j = period + 1; j < window.length; j++) {
    const d = window[j] - window[j - 1];
    avgGain = (avgGain * (period - 1) + Math.max(d,  0)) / period;
    avgLoss = (avgLoss * (period - 1) + Math.max(-d, 0)) / period;
  }
  if (avgLoss === 0) return 100;
  return 100 - (100 / (1 + avgGain / avgLoss));
}

function atrAt(bars, i, period) {
  if (i < period) return NaN;
  const sl  = bars.slice(Math.max(0, i - period), i + 1);
  const trs = [];
  for (let j = 1; j < sl.length; j++) {
    trs.push(Math.max(
      sl[j].high - sl[j].low,
      Math.abs(sl[j].high - sl[j - 1].close),
      Math.abs(sl[j].low  - sl[j - 1].close)
    ));
  }
  return smaLast(trs, period);
}

// ── Precompute all indicator series for a symbol ──────────────────────────────

function precompute(bars) {
  const n      = bars.length;
  const closes = bars.map(b => b.close);
  const vols   = bars.map(b => b.volume);

  const ema200 = emaFull(closes, P.ema_trend);
  const ema50  = emaFull(closes, P.ema_pullback_slow);
  const ema20  = emaFull(closes, P.ema_pullback_fast);

  const wr   = bars.map((_, i) => williamsRAt(bars, i, P.wr_period));
  const rsi2 = closes.map((_, i) => rsiAt(closes, i, P.rsi_period));
  const atr  = bars.map((_, i) => atrAt(bars, i, P.atr_period));

  const volSma = vols.map((_, i) =>
    i < P.volume_sma_period - 1 ? NaN
    : smaLast(vols.slice(i - P.volume_sma_period + 1, i + 1), P.volume_sma_period)
  );

  return { closes, vols, ema200, ema50, ema20, wr, rsi2, atr, volSma };
}

function tsToDate(ts) {
  return new Date(ts * 1000).toISOString().slice(0, 10);
}

// ── Bar-by-bar backtest for one symbol ───────────────────────────────────────

function backtestSymbol(bars, spyEma200Series, vixByTs) {
  const ind     = precompute(bars);
  const trades  = [];
  let   inTrade = null;

  for (let i = 210; i < bars.length - 1; i++) {
    const bar     = bars[i];
    const nextBar = bars[i + 1];

    // ── Check open trade exit ────────────────────────────────────────────
    if (inTrade) {
      const t       = inTrade;
      const barsIn  = i - t.startBar + 1;
      let   exitP   = null;
      let   reason  = null;

      if (t.signal === 'LONG') {
        if (bar.low <= t.stop)       { exitP = t.stop;  reason = 'stop';      }
        else if (bar.high >= t.tp1)  { exitP = t.tp1;   reason = 'tp1';       }
        else if (barsIn >= TIME_STOP_BARS) { exitP = bar.close; reason = 'time'; }
      } else {
        if (bar.high >= t.stop)      { exitP = t.stop;  reason = 'stop';      }
        else if (bar.low <= t.tp1)   { exitP = t.tp1;   reason = 'tp1';       }
        else if (barsIn >= TIME_STOP_BARS) { exitP = bar.close; reason = 'time'; }
      }

      if (exitP !== null) {
        const pnl = t.signal === 'LONG'
          ? (exitP - t.entry) / t.risk
          : (t.entry - exitP) / t.risk;
        trades.push({ ...t, exitDate: tsToDate(bar.time), exitPrice: +exitP.toFixed(2), exitReason: reason, rMultiple: +pnl.toFixed(2) });
        inTrade = null;
      }
    }

    if (inTrade) continue; // one trade at a time per symbol

    // ── Market context ───────────────────────────────────────────────────
    const spyEma = spyEma200Series ? spyEma200Series[i] : NaN;
    const spyClose = spyEma200Series ? bars[i].close : NaN; // will be overridden below
    // We'll use the SPY series indexed by ts alignment - see below in main backtest
    const spyAbove = SKIP_MARKET_FILTER ? true : (spyEma200Series?.spyAboveAt?.(bar.time) ?? true);
    const vix      = SKIP_MARKET_FILTER ? 0    : (vixByTs?.get(bar.time) ?? 0);
    const vixOk    = vix === 0 || vix < CFG.market_filters.vix_max;

    // ── Indicator values at bar i ────────────────────────────────────────
    const e200     = ind.ema200[i];
    const e50      = ind.ema50[i];
    const e20      = ind.ema20[i];
    const wrVal    = ind.wr[i];
    const rsi2Val  = ind.rsi2[i];
    const atrVal   = ind.atr[i];
    const volRatio = ind.volSma[i] > 0 ? bar.volume / ind.volSma[i] : 0;

    if ([e200, e50, wrVal, rsi2Val, atrVal].some(v => isNaN(v) || v === 0)) continue;

    const close          = bar.close;
    const distToEma50    = Math.abs(close - e50);
    const pullbackOk     = distToEma50 <= atrVal * P.pullback_zone_atr_mult;

    // ── LONG ─────────────────────────────────────────────────────────────
    const isLong = (
      spyAbove && vixOk &&
      close > e200 && pullbackOk &&
      wrVal  < P.wr_oversold  &&
      rsi2Val < P.rsi_oversold &&
      volRatio >= P.volume_min_ratio
    );

    // ── SHORT ─────────────────────────────────────────────────────────────
    const isShort = (
      !SKIP_MARKET_FILTER && !spyAbove &&
      close < e200 && pullbackOk &&
      wrVal  > P.wr_overbought &&
      rsi2Val > P.rsi_overbought &&
      volRatio >= P.volume_min_ratio
    );

    if (!isLong && !isShort) continue;

    const sig = isLong ? 'LONG' : 'SHORT';

    // ── Stop and entry ────────────────────────────────────────────────────
    let stop;
    if (sig === 'LONG') {
      const swingLow   = Math.min(...bars.slice(Math.max(0, i - 10), i).map(b => b.low));
      const swingStop  = swingLow * 0.999;
      const atrStop    = close - atrVal * P.atr_stop_mult;
      stop             = Math.min(swingStop, atrStop);
    } else {
      const swingHigh  = Math.max(...bars.slice(Math.max(0, i - 10), i).map(b => b.high));
      const swingStop  = swingHigh * 1.001;
      const atrStop    = close + atrVal * P.atr_stop_mult;
      stop             = Math.max(swingStop, atrStop);
    }

    const entry = nextBar.open;  // real entry = next bar's open
    const risk  = sig === 'LONG' ? entry - stop : stop - entry;
    if (risk <= 0 || risk > entry * 0.10) continue; // skip if stop is nonsensical (>10% away)

    const tp1 = sig === 'LONG' ? entry + risk * RISK.tp1_rr : entry - risk * RISK.tp1_rr;
    const tp2 = sig === 'LONG' ? entry + risk * RISK.tp2_rr : entry - risk * RISK.tp2_rr;

    inTrade = {
      signal:     sig,
      entryDate:  tsToDate(nextBar.time),
      entry:      +entry.toFixed(2),
      stop:       +stop.toFixed(2),
      tp1:        +tp1.toFixed(2),
      tp2:        +tp2.toFixed(2),
      risk:       +risk.toFixed(2),
      startBar:   i + 1,
      indicators: { e200: +e200.toFixed(2), e50: +e50.toFixed(2), wr: +wrVal.toFixed(1), rsi2: +rsi2Val.toFixed(1) },
    };
  }

  // Force-close any open trade at end of data
  if (inTrade) {
    const last = bars[bars.length - 1];
    const pnl  = inTrade.signal === 'LONG'
      ? (last.close - inTrade.entry) / inTrade.risk
      : (inTrade.entry - last.close) / inTrade.risk;
    trades.push({ ...inTrade, exitDate: tsToDate(last.time), exitPrice: +last.close.toFixed(2), exitReason: 'end_of_data', rMultiple: +pnl.toFixed(2) });
  }

  return trades;
}

// ── Stats ─────────────────────────────────────────────────────────────────────

function calcStats(trades) {
  if (trades.length === 0) return null;
  const wins    = trades.filter(t => t.rMultiple >= 1.8);    // ≥TP1 hit
  const losses  = trades.filter(t => t.rMultiple <= -0.8);   // stop hit
  const timeouts = trades.filter(t => t.exitReason === 'time' || t.exitReason === 'end_of_data');

  const totalR  = trades.reduce((s, t) => s + t.rMultiple, 0);
  const avgR    = totalR / trades.length;

  let maxWin = 0, maxLoss = 0, streak = 0, maxStreak = 0, loseStreak = 0, maxLoseStreak = 0;
  for (const t of trades) {
    if (t.rMultiple > 0) { streak++; loseStreak = 0; maxStreak = Math.max(maxStreak, streak); }
    else                  { loseStreak++; streak = 0; maxLoseStreak = Math.max(maxLoseStreak, loseStreak); }
    maxWin  = Math.max(maxWin, t.rMultiple);
    maxLoss = Math.min(maxLoss, t.rMultiple);
  }

  return {
    total:        trades.length,
    wins:         wins.length,
    losses:       losses.length,
    timeouts:     timeouts.length,
    winRate:      +((wins.length / trades.length) * 100).toFixed(1),
    totalR:       +totalR.toFixed(2),
    avgR:         +avgR.toFixed(2),
    maxWin:       +maxWin.toFixed(2),
    maxLoss:      +maxLoss.toFixed(2),
    maxWinStreak: maxStreak,
    maxLossStreak: maxLoseStreak,
  };
}

// ── Console output ────────────────────────────────────────────────────────────

const DLINE = '═'.repeat(100);
const LINE  = '─'.repeat(100);

function printTrades(symbol, trades, stats) {
  if (trades.length === 0) {
    console.log(`  ${symbol}: 0 trades`);
    return;
  }
  console.log(`\n${DLINE}`);
  console.log(`  ${symbol}  │  ${trades.length} trades  │  Win rate: ${stats.winRate}%  │  Total R: ${stats.totalR >= 0 ? '+' : ''}${stats.totalR}R  │  Avg: ${stats.avgR >= 0 ? '+' : ''}${stats.avgR}R`);
  console.log(DLINE);
  console.log(' Entry       | Exit        | Dir   | Entry $   | Exit $    | Stop      | TP1       |  R    | Reason');
  console.log(LINE);
  for (const t of trades) {
    const r = t.rMultiple >= 0 ? `+${t.rMultiple}R` : `${t.rMultiple}R`;
    const rPad = r.padStart(6);
    console.log(
      ` ${t.entryDate} | ${t.exitDate} | ${t.signal.padEnd(5)} | ` +
      `${String(t.entry).padStart(9)} | ${String(t.exitPrice).padStart(9)} | ` +
      `${String(t.stop).padStart(9)} | ${String(t.tp1).padStart(9)} | ${rPad} | ${t.exitReason}`
    );
  }
  console.log(LINE);
  console.log(`  Wins: ${stats.wins}  Losses: ${stats.losses}  Timeouts: ${stats.timeouts}  Max win streak: ${stats.maxWinStreak}  Max loss streak: ${stats.maxLossStreak}`);
}

function printSummary(allResults) {
  const allTrades = allResults.flatMap(r => r.trades);
  const combined  = calcStats(allTrades);
  if (!combined) { console.log('\nNo trades generated.'); return; }

  console.log(`\n${'═'.repeat(60)}`);
  console.log('  BACKTEST SUMMARY (all symbols combined)');
  console.log('═'.repeat(60));
  console.log(`  Total trades  : ${combined.total}`);
  console.log(`  Win rate      : ${combined.winRate}%`);
  console.log(`  Total R       : ${combined.totalR >= 0 ? '+' : ''}${combined.totalR}R`);
  console.log(`  Avg R/trade   : ${combined.avgR >= 0 ? '+' : ''}${combined.avgR}R`);
  console.log(`  Best trade    : ${combined.maxWin >= 0 ? '+' : ''}${combined.maxWin}R`);
  console.log(`  Worst trade   : ${combined.maxLoss}R`);
  console.log(`  Max win streak: ${combined.maxWinStreak}`);
  console.log(`  Max loss streak: ${combined.maxLossStreak}`);
  console.log('═'.repeat(60));

  // Per-symbol summary
  console.log('\n  Per-symbol:');
  for (const r of allResults) {
    if (!r.stats) { console.log(`  ${r.symbol.padEnd(14)}: no trades`); continue; }
    const s = r.stats;
    const rStr = `${s.totalR >= 0 ? '+' : ''}${s.totalR}R`;
    console.log(`  ${r.symbol.padEnd(14)}: ${String(r.trades.length).padStart(3)} trades  win=${s.winRate}%  total=${rStr.padStart(7)}  avg=${(s.avgR >= 0 ? '+' : '') + s.avgR}R`);
  }
  console.log('');
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function run() {
  let savedState = null;
  try {
    savedState = await getState();
    await setTimeframe({ timeframe: 'D' });

    // ── Fetch SPY data ────────────────────────────────────────────────────
    console.log(`\n[Backtest] Fetching market data...`);

    let spyBars = [];
    let spyEma200Map = new Map(); // ts -> { close, ema200 }
    if (!SKIP_MARKET_FILTER) {
      process.stdout.write('  SPY ... ');
      await setSymbol({ symbol: CFG.market_filters.spy_symbol });
      const spyOhlcv = await getOhlcv({ count: BAR_COUNT });
      spyBars = spyOhlcv.bars;
      const spyCloses  = spyBars.map(b => b.close);
      const spyEma200s = emaFull(spyCloses, P.ema_trend);
      spyBars.forEach((b, i) => spyEma200Map.set(b.time, { close: b.close, ema200: spyEma200s[i] }));
      console.log(`${spyBars.length} bars`);
    }

    let vixByTs = new Map();
    if (!SKIP_MARKET_FILTER) {
      process.stdout.write('  VIX ... ');
      try {
        await setSymbol({ symbol: CFG.market_filters.vix_symbol });
        const vixOhlcv = await getOhlcv({ count: BAR_COUNT });
        vixOhlcv.bars.forEach(b => vixByTs.set(b.time, b.close));
        console.log(`${vixOhlcv.bars.length} bars`);
      } catch {
        console.log('unavailable (skipping VIX filter)');
      }
    }

    // ── Backtest each symbol ──────────────────────────────────────────────
    console.log(`\n[Backtest] Running strategy on ${WATCHLIST.length} symbol(s)...`);

    const allResults = [];

    for (const symbol of WATCHLIST) {
      process.stdout.write(`  ${symbol.padEnd(14)} ... `);
      try {
        await setSymbol({ symbol });
        const ohlcv = await getOhlcv({ count: BAR_COUNT });
        if (!ohlcv.bars || ohlcv.bars.length < 215) {
          console.log(`insufficient data (${ohlcv.bars?.length ?? 0} bars)`);
          allResults.push({ symbol, trades: [], stats: null });
          continue;
        }
        const bars = ohlcv.bars;

        // Build per-bar market context using date alignment with SPY/VIX
        // Wrap into helper closures so backtestSymbol doesn't need to know about alignment
        const spyCtxObj = {
          spyAboveAt: (ts) => {
            const ctx = spyEma200Map.get(ts);
            if (!ctx || isNaN(ctx.ema200)) return true; // default: bull
            return ctx.close > ctx.ema200;
          }
        };

        // Re-run backtestSymbol with ts-based market context
        const trades = backtestSymbolFull(bars, spyCtxObj, vixByTs);
        const stats  = calcStats(trades);

        allResults.push({ symbol, trades, stats });
        console.log(`${trades.length} trades  win=${stats?.winRate ?? 0}%  totalR=${stats?.totalR ?? 0}R`);
      } catch (e) {
        console.log(`ERROR: ${e.message}`);
        allResults.push({ symbol, trades: [], stats: null, error: e.message });
      }
    }

    // ── Print results ─────────────────────────────────────────────────────
    for (const r of allResults) {
      printTrades(r.symbol, r.trades, r.stats);
    }
    printSummary(allResults);

    // ── Save ──────────────────────────────────────────────────────────────
    writeFileSync(RESULTS_PATH, JSON.stringify({
      backtest_date: new Date().toISOString(),
      bar_count: BAR_COUNT,
      timeframe: 'D',
      time_stop_bars: TIME_STOP_BARS,
      market_filter_applied: !SKIP_MARKET_FILTER,
      results: allResults,
    }, null, 2));
    console.log(`[Saved] ${RESULTS_PATH}`);

  } catch (e) {
    console.error(`\n[ERROR] ${e.message}`);
    if (e.message.includes('CDP') || e.message.includes('fetch')) {
      console.error('       Is TradingView Desktop open?');
    }
  } finally {
    if (savedState) {
      try {
        await setSymbol({ symbol: savedState.symbol });
        await setTimeframe({ timeframe: savedState.resolution });
      } catch {}
    }
    await disconnect();
  }
}

// ── backtestSymbolFull — ts-aligned market context ────────────────────────────

function backtestSymbolFull(bars, spyCtx, vixByTs) {
  const ind     = precompute(bars);
  const trades  = [];
  let   inTrade = null;

  for (let i = 210; i < bars.length - 1; i++) {
    const bar     = bars[i];
    const nextBar = bars[i + 1];

    // ── Exit check ───────────────────────────────────────────────────────
    if (inTrade) {
      const t      = inTrade;
      const barsIn = i - t.startBar + 1;
      let exitP = null, reason = null;

      if (t.signal === 'LONG') {
        if (bar.low  <= t.stop)             { exitP = t.stop;   reason = 'stop'; }
        else if (bar.high >= t.tp1)         { exitP = t.tp1;    reason = 'tp1';  }
        else if (barsIn >= TIME_STOP_BARS)  { exitP = bar.close; reason = 'time'; }
      } else {
        if (bar.high >= t.stop)             { exitP = t.stop;   reason = 'stop'; }
        else if (bar.low  <= t.tp1)         { exitP = t.tp1;    reason = 'tp1';  }
        else if (barsIn >= TIME_STOP_BARS)  { exitP = bar.close; reason = 'time'; }
      }

      if (exitP !== null) {
        const pnl = t.signal === 'LONG'
          ? (exitP - t.entry) / t.risk
          : (t.entry - exitP) / t.risk;
        trades.push({ ...t, exitDate: tsToDate(bar.time), exitPrice: +exitP.toFixed(2), exitReason: reason, rMultiple: +pnl.toFixed(2) });
        inTrade = null;
      }
    }

    if (inTrade) continue;

    // ── Market context ────────────────────────────────────────────────────
    const spyAbove = SKIP_MARKET_FILTER ? true : (spyCtx?.spyAboveAt(bar.time) ?? true);
    const vix      = SKIP_MARKET_FILTER ? 0    : (vixByTs?.get(bar.time) ?? 0);
    const vixOk    = vix === 0 || vix < CFG.market_filters.vix_max;

    // ── Indicator values ─────────────────────────────────────────────────
    const e200     = ind.ema200[i];
    const e50      = ind.ema50[i];
    const wrVal    = ind.wr[i];
    const rsi2Val  = ind.rsi2[i];
    const atrVal   = ind.atr[i];
    const volRatio = ind.volSma[i] > 0 ? bar.volume / ind.volSma[i] : 0;
    const close    = bar.close;

    if ([e200, e50, wrVal, rsi2Val, atrVal].some(v => isNaN(v))) continue;

    const pullbackOk = Math.abs(close - e50) <= atrVal * P.pullback_zone_atr_mult;

    const isLong  = spyAbove && vixOk && close > e200 && pullbackOk && wrVal < P.wr_oversold  && rsi2Val < P.rsi_oversold  && volRatio >= P.volume_min_ratio;
    const isShort = !spyAbove           && close < e200 && pullbackOk && wrVal > P.wr_overbought && rsi2Val > P.rsi_overbought && volRatio >= P.volume_min_ratio;

    if (!isLong && !isShort) continue;

    const sig = isLong ? 'LONG' : 'SHORT';

    let stop;
    if (sig === 'LONG') {
      const sl  = Math.min(...bars.slice(Math.max(0, i - 10), i).map(b => b.low));
      stop      = Math.min(sl * 0.999, close - atrVal * P.atr_stop_mult);
    } else {
      const sh  = Math.max(...bars.slice(Math.max(0, i - 10), i).map(b => b.high));
      stop      = Math.max(sh * 1.001, close + atrVal * P.atr_stop_mult);
    }

    const entry = nextBar.open;
    const risk  = sig === 'LONG' ? entry - stop : stop - entry;
    if (risk <= 0 || risk > entry * 0.10) continue;

    const tp1 = sig === 'LONG' ? entry + risk * RISK.tp1_rr : entry - risk * RISK.tp1_rr;
    const tp2 = sig === 'LONG' ? entry + risk * RISK.tp2_rr : entry - risk * RISK.tp2_rr;

    inTrade = {
      signal:     sig,
      entryDate:  tsToDate(nextBar.time),
      entry:      +entry.toFixed(2),
      stop:       +stop.toFixed(2),
      tp1:        +tp1.toFixed(2),
      tp2:        +tp2.toFixed(2),
      risk:       +risk.toFixed(2),
      startBar:   i + 1,
      indicators: { e200: +e200.toFixed(2), e50: +e50.toFixed(2), wr: +wrVal.toFixed(1), rsi2: +rsi2Val.toFixed(1) },
    };
  }

  if (inTrade) {
    const last = bars[bars.length - 1];
    const pnl  = inTrade.signal === 'LONG'
      ? (last.close - inTrade.entry) / inTrade.risk
      : (inTrade.entry - last.close) / inTrade.risk;
    trades.push({ ...inTrade, exitDate: tsToDate(last.time), exitPrice: +last.close.toFixed(2), exitReason: 'end_of_data', rMultiple: +pnl.toFixed(2) });
  }

  return trades;
}

run().catch(err => { console.error(err.message); process.exit(1); });
