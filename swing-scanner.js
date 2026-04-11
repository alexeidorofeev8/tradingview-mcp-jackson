/**
 * swing-scanner.js — NYSE Swing Strategy Signal Scanner
 *
 * Strategy: EMA(200) trend filter + Williams %R(14) pullback + RSI(2) + ATR(14) stops
 * Filters: SPY market regime + VIX level + Earnings blackout
 *
 * Usage:
 *   node swing-scanner.js                     # scan full watchlist
 *   node swing-scanner.js --symbol NYSE:JPM   # single ticker
 *   node swing-scanner.js --watch 300         # refresh every 5 minutes
 *   node swing-scanner.js --verbose           # show all indicator values
 */

import { setSymbol, setTimeframe, getState } from './src/core/chart.js';
import { getOhlcv } from './src/core/data.js';
import { disconnect } from './src/connection.js';
import { readFileSync, writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dir = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dir, 'rules-swing.json');
const SIGNALS_PATH = join(__dir, 'swing-signals.json');

// ── CLI args ──────────────────────────────────────────────────────────────────
const argv = process.argv.slice(2);
const args = {};
for (let i = 0; i < argv.length; i++) {
  if (argv[i].startsWith('--')) {
    const key = argv[i].slice(2);
    const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[i + 1] : true;
    args[key] = val;
    if (val !== true) i++;
  }
}

const SYMBOL_FILTER  = args.symbol    || null;
const WATCH_INTERVAL = args.watch     ? parseInt(args.watch) * 1000 : null;
const TIMEFRAME      = args.timeframe || 'D';
const VERBOSE        = !!args.verbose;

// ── Config ────────────────────────────────────────────────────────────────────
const CFG  = JSON.parse(readFileSync(CONFIG_PATH, 'utf8'));
const P    = CFG.parameters;
const RISK = CFG.risk;
const WATCHLIST = SYMBOL_FILTER ? [SYMBOL_FILTER] : CFG.watchlist;

// ── Indicator math ────────────────────────────────────────────────────────────

/**
 * EMA seeded by SMA of the first `period` bars, then smoothed.
 * Returns the last (most recent) value.
 */
function emaLast(values, period) {
  if (values.length < period) return NaN;
  const k = 2 / (period + 1);
  // Seed: SMA of first `period` bars
  let e = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = period; i < values.length; i++) {
    e = values[i] * k + e * (1 - k);
  }
  return e;
}

/**
 * SMA of last `period` values.
 */
function smaLast(values, period) {
  if (values.length < period) return NaN;
  return values.slice(-period).reduce((a, b) => a + b, 0) / period;
}

/**
 * Williams %R over last `period` bars.
 * Range: -100 (oversold) to 0 (overbought).
 */
function williamsR(bars, period) {
  const slice = bars.slice(-period);
  const hh = Math.max(...slice.map(b => b.high));
  const ll  = Math.min(...slice.map(b => b.low));
  if (hh === ll) return -50;
  return ((hh - bars[bars.length - 1].close) / (hh - ll)) * -100;
}

/**
 * Wilder's RSI for last value. Pass the last 50+ bars for stability.
 * Period 2 converges very fast — 30 bars is more than enough.
 */
function rsiLast(closes, period) {
  if (closes.length < period + 2) return 50;
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i - 1];
    if (d >= 0) gains += d; else losses -= d;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  for (let i = period + 1; i < closes.length; i++) {
    const d = closes[i] - closes[i - 1];
    avgGain = (avgGain * (period - 1) + Math.max(d,  0)) / period;
    avgLoss = (avgLoss * (period - 1) + Math.max(-d, 0)) / period;
  }
  if (avgLoss === 0) return 100;
  return 100 - (100 / (1 + avgGain / avgLoss));
}

/**
 * ATR(period): SMA of true range over last `period` bars.
 */
function atrLast(bars, period) {
  const trs = [];
  for (let i = 1; i < bars.length; i++) {
    trs.push(Math.max(
      bars[i].high - bars[i].low,
      Math.abs(bars[i].high - bars[i - 1].close),
      Math.abs(bars[i].low  - bars[i - 1].close)
    ));
  }
  return smaLast(trs, period);
}

/**
 * Lowest low of the last `lookback` bars, excluding the current bar.
 */
function recentSwingLow(bars, lookback = 10) {
  return Math.min(...bars.slice(-lookback - 1, -1).map(b => b.low));
}

/**
 * Highest high of the last `lookback` bars, excluding the current bar.
 */
function recentSwingHigh(bars, lookback = 10) {
  return Math.max(...bars.slice(-lookback - 1, -1).map(b => b.high));
}

// ── Earnings blackout check ───────────────────────────────────────────────────
function isInEarningsBlackout(symbol) {
  const dateStr = CFG.earnings_blackout?.[symbol];
  if (!dateStr || dateStr.startsWith('_')) return false;
  return new Date(dateStr) > new Date();
}

// ── Signal evaluation ─────────────────────────────────────────────────────────
function evalSignal(bars, symbol, mkt) {
  const closes = bars.map(b => b.close);
  const vols   = bars.map(b => b.volume);
  const last   = bars[bars.length - 1];
  const close  = last.close;

  // Indicators
  const ema200   = emaLast(closes, P.ema_trend);
  const ema50    = emaLast(closes, P.ema_pullback_slow);
  const ema20    = emaLast(closes, P.ema_pullback_fast);
  const wr       = williamsR(bars, P.wr_period);
  const rsi2     = rsiLast(closes.slice(-40), P.rsi_period);
  const atr      = atrLast(bars, P.atr_period);
  const volSma   = smaLast(vols, P.volume_sma_period);
  const volRatio = volSma > 0 ? last.volume / volSma : 0;

  const indic = {
    ema200:   +ema200.toFixed(2),
    ema50:    +ema50.toFixed(2),
    ema20:    +ema20.toFixed(2),
    wr:       +wr.toFixed(1),
    rsi2:     +rsi2.toFixed(1),
    atr:      +atr.toFixed(2),
    volRatio: +volRatio.toFixed(2),
  };

  // ── Earnings blackout ─────────────────────────────────────────────────────
  if (isInEarningsBlackout(symbol)) {
    const until = CFG.earnings_blackout[symbol];
    return { signal: 'BLOCKED', reason: `Earnings until ${until}`, indicators: indic };
  }

  const distToEma50 = Math.abs(close - ema50);
  const pullbackOk  = distToEma50 <= atr * P.pullback_zone_atr_mult;

  // ── LONG ──────────────────────────────────────────────────────────────────
  const longOk = (
    mkt.spyAboveEma200 === true &&
    mkt.vix !== null && mkt.vix < CFG.market_filters.vix_max &&
    close > ema200 &&
    pullbackOk &&
    wr < P.wr_oversold &&
    rsi2 < P.rsi_oversold &&
    volRatio >= P.volume_min_ratio
  );

  if (longOk) {
    const swingStop = recentSwingLow(bars, 10) * 0.999;
    const atrStop   = close - atr * P.atr_stop_mult;
    const stop      = Math.min(swingStop, atrStop); // wider = more conservative
    const risk      = close - stop;
    if (risk <= 0) return { signal: 'SKIP', reason: 'Invalid risk (stop >= entry)', indicators: indic };

    const tp1  = +(close + risk * RISK.tp1_rr).toFixed(2);
    const tp2  = +(close + risk * RISK.tp2_rr).toFixed(2);
    const size = Math.max(1, Math.floor((RISK.capital * RISK.risk_per_trade_pct / 100) / risk));

    return {
      signal:       'LONG',
      entry:        +close.toFixed(2),
      stop:         +stop.toFixed(2),
      tp1,
      tp2,
      risk:         +risk.toFixed(2),
      riskDollars:  +(risk * size).toFixed(0),
      positionSize: size,
      indicators:   indic,
    };
  }

  // ── SHORT ─────────────────────────────────────────────────────────────────
  const shortOk = (
    mkt.spyAboveEma200 === false &&
    close < ema200 &&
    pullbackOk &&
    wr > P.wr_overbought &&
    rsi2 > P.rsi_overbought &&
    volRatio >= P.volume_min_ratio
  );

  if (shortOk) {
    const swingStop = recentSwingHigh(bars, 10) * 1.001;
    const atrStop   = close + atr * P.atr_stop_mult;
    const stop      = Math.max(swingStop, atrStop); // wider = more conservative
    const risk      = stop - close;
    if (risk <= 0) return { signal: 'SKIP', reason: 'Invalid risk (stop <= entry)', indicators: indic };

    const tp1  = +(close - risk * RISK.tp1_rr).toFixed(2);
    const tp2  = +(close - risk * RISK.tp2_rr).toFixed(2);
    const size = Math.max(1, Math.floor((RISK.capital * RISK.risk_per_trade_pct / 100) / risk));

    return {
      signal:       'SHORT',
      entry:        +close.toFixed(2),
      stop:         +stop.toFixed(2),
      tp1,
      tp2,
      risk:         +risk.toFixed(2),
      riskDollars:  +(risk * size).toFixed(0),
      positionSize: size,
      indicators:   indic,
    };
  }

  // ── No signal — collect reasons ───────────────────────────────────────────
  const failReasons = [];
  if (mkt.spyAboveEma200 === false)                             failReasons.push('SPY<EMA200 (no longs)');
  if (mkt.vix !== null && mkt.vix >= CFG.market_filters.vix_max) failReasons.push(`VIX=${mkt.vix}`);
  if (close <= ema200)                                          failReasons.push(`below EMA200(${ema200.toFixed(2)})`);
  if (!pullbackOk)   failReasons.push(`dist EMA50=${distToEma50.toFixed(2)} > ${(atr * P.pullback_zone_atr_mult).toFixed(2)}`);
  if (wr >= P.wr_oversold)  failReasons.push(`W%R=${wr.toFixed(1)}`);
  if (rsi2 >= P.rsi_oversold) failReasons.push(`RSI2=${rsi2.toFixed(1)}`);
  if (volRatio < P.volume_min_ratio) failReasons.push(`vol=${volRatio.toFixed(2)}x`);

  return {
    signal:     'WAIT',
    reason:     failReasons.join(' | ') || 'conditions not met',
    indicators: indic,
  };
}

// ── Console output ────────────────────────────────────────────────────────────
const LINE = '─'.repeat(96);
const DLINE = '═'.repeat(96);

function printTable(results, mkt) {
  const ts = new Date().toLocaleString('ru-RU', { timeZone: 'Europe/Moscow' });
  const mktStr = mkt.vixOk === false
    ? `SPY:${mkt.spyAboveEma200 ? 'BULL' : 'BEAR'}  VIX:${mkt.vix} [HIGH - longs blocked]`
    : `SPY:${mkt.spyAboveEma200 ? 'BULL' : 'BEAR'}  VIX:${mkt.vix ?? 'n/a'}`;

  console.log('\n' + DLINE);
  console.log(`  NYSE Swing Scanner  |  ${ts} МСК  |  ${mktStr}`);
  console.log(DLINE);
  console.log(' Symbol          | Signal  | Entry    | Stop     | TP1      | TP2      | Size | $Risk | Reason');
  console.log(LINE);

  for (const { symbol, result: r } of results) {
    const sigPad = r.signal.padEnd(6);
    const e  = r.entry        ? String(r.entry.toFixed(2)).padStart(8)  : '       -';
    const s  = r.stop         ? String(r.stop.toFixed(2)).padStart(8)   : '       -';
    const t1 = r.tp1          ? String(r.tp1.toFixed(2)).padStart(8)    : '       -';
    const t2 = r.tp2          ? String(r.tp2.toFixed(2)).padStart(8)    : '       -';
    const sz = r.positionSize ? String(r.positionSize).padStart(4)      : '   -';
    const dr = r.riskDollars  ? `$${String(r.riskDollars).padStart(4)}` : '    -';
    const reason = (r.signal === 'LONG' || r.signal === 'SHORT')
      ? `W%R:${r.indicators?.wr} RSI2:${r.indicators?.rsi2} vol:${r.indicators?.volRatio}x`
      : (r.reason || '').slice(0, 40);

    console.log(` ${symbol.padEnd(16)}| ${sigPad} | ${e} | ${s} | ${t1} | ${t2} | ${sz} | ${dr} | ${reason}`);

    if (VERBOSE && r.indicators) {
      const ind = r.indicators;
      console.log(`   ${''.padEnd(15)}  EMA200:${ind.ema200}  EMA50:${ind.ema50}  EMA20:${ind.ema20}  ATR:${ind.atr}`);
    }
  }

  console.log(DLINE);

  const active = results.filter(r => r.result.signal === 'LONG' || r.result.signal === 'SHORT');
  if (active.length === 0) {
    console.log('  No actionable signals at this time.\n');
  } else {
    console.log(`  ${active.length} signal(s):`);
    for (const { symbol, result: r } of active) {
      console.log(`  >> ${symbol} ${r.signal}: entry=${r.entry}  stop=${r.stop}  TP1=${r.tp1}  TP2=${r.tp2}  size=${r.positionSize}sh  risk=$${r.riskDollars}`);
    }
    console.log('');
  }
}

// ── Main scan ─────────────────────────────────────────────────────────────────
async function scan() {
  let savedSymbol     = null;
  let savedResolution = null;

  try {
    // Save current chart state so we can restore it at the end
    try {
      const state = await getState();
      savedSymbol     = state.symbol;
      savedResolution = state.resolution;
    } catch { /* TradingView might not be running yet */ }

    console.log(`\n[Swing Scanner] ${new Date().toLocaleTimeString('ru-RU')} — scanning ${WATCHLIST.length} symbol(s) on ${TIMEFRAME}...`);

    // Switch to daily timeframe if needed
    if (savedResolution && savedResolution !== TIMEFRAME) {
      await setTimeframe({ timeframe: TIMEFRAME });
    }

    // ── Market context: VIX ───────────────────────────────────────────────
    let vixLevel = null;
    try {
      await setSymbol({ symbol: CFG.market_filters.vix_symbol });
      const vixOhlcv = await getOhlcv({ count: 3 });
      vixLevel = vixOhlcv.bars[vixOhlcv.bars.length - 1].close;
      process.stdout.write(`[Market] VIX=${vixLevel.toFixed(2)}`);
    } catch (e) {
      process.stdout.write('[Market] VIX=n/a (unavailable)');
    }

    // ── Market context: SPY EMA(200) ──────────────────────────────────────
    let spyAboveEma200 = true; // optimistic default
    try {
      await setSymbol({ symbol: CFG.market_filters.spy_symbol });
      const spyOhlcv = await getOhlcv({ count: 220 });
      const spyCloses    = spyOhlcv.bars.map(b => b.close);
      const spyEma200    = emaLast(spyCloses, P.ema_trend);
      const spyClose     = spyCloses[spyCloses.length - 1];
      spyAboveEma200     = spyClose > spyEma200;
      process.stdout.write(`  SPY=${spyClose.toFixed(2)} EMA200=${spyEma200.toFixed(2)} → ${spyAboveEma200 ? 'BULL' : 'BEAR'}\n`);
    } catch (e) {
      process.stdout.write('  SPY=n/a (assuming bull)\n');
    }

    const mkt = {
      vix:           vixLevel !== null ? +vixLevel.toFixed(2) : null,
      vixOk:         vixLevel === null ? null : vixLevel < CFG.market_filters.vix_max,
      spyAboveEma200,
      timestamp:     new Date().toISOString(),
    };

    if (vixLevel !== null && vixLevel >= CFG.market_filters.vix_max) {
      console.log(`[Market] VIX >= ${CFG.market_filters.vix_max} — long signals suppressed`);
    }

    // ── Scan each symbol ──────────────────────────────────────────────────
    const results = [];

    for (const symbol of WATCHLIST) {
      process.stdout.write(`  ${symbol.padEnd(12)} ... `);
      try {
        await setSymbol({ symbol });

        // Retry once if chart didn't finish loading in time
        let ohlcv;
        for (let attempt = 0; attempt < 2; attempt++) {
          try {
            ohlcv = await getOhlcv({ count: 220 });
            if (ohlcv.bars && ohlcv.bars.length >= 210) break;
          } catch {
            if (attempt === 0) await new Promise(r => setTimeout(r, 2000));
            else throw new Error(`Could not load data after retry`);
          }
        }

        if (!ohlcv.bars || ohlcv.bars.length < 210) {
          console.log(`insufficient data (${ohlcv.bars?.length ?? 0} bars)`);
          results.push({ symbol, result: { signal: 'ERROR', reason: `only ${ohlcv.bars?.length ?? 0} bars` } });
          continue;
        }

        const result = evalSignal(ohlcv.bars, symbol, mkt);
        results.push({ symbol, result });

        if (result.signal === 'LONG' || result.signal === 'SHORT') {
          console.log(`*** ${result.signal} *** entry=${result.entry} stop=${result.stop} TP1=${result.tp1}`);
        } else {
          const shortReason = result.reason ? result.reason.split('|')[0].trim().slice(0, 45) : '';
          console.log(`${result.signal}  ${shortReason}`);
        }
      } catch (e) {
        console.log(`ERROR: ${e.message}`);
        results.push({ symbol, result: { signal: 'ERROR', reason: e.message } });
      }
    }

    // ── Print table ───────────────────────────────────────────────────────
    printTable(results, mkt);

    // ── Save signals ──────────────────────────────────────────────────────
    const output = {
      scanned_at: new Date().toISOString(),
      market:     mkt,
      signals:    results.filter(r => ['LONG', 'SHORT'].includes(r.result.signal)),
      all:        results,
    };
    writeFileSync(SIGNALS_PATH, JSON.stringify(output, null, 2));
    console.log(`[Saved] ${SIGNALS_PATH}`);

  } catch (e) {
    console.error(`\n[ERROR] Scan failed: ${e.message}`);
    if (e.message.includes('CDP') || e.message.includes('connect') || e.message.includes('fetch')) {
      console.error('       Is TradingView Desktop open? Run: npm run tv health');
    }
  } finally {
    // Always restore original chart state
    if (savedSymbol) {
      try {
        await setSymbol({ symbol: savedSymbol });
        if (savedResolution && savedResolution !== TIMEFRAME) {
          await setTimeframe({ timeframe: savedResolution });
        }
      } catch { /* ignore restore errors */ }
    }
    if (!WATCH_INTERVAL) {
      await disconnect();
    }
  }
}

// ── Entry point ───────────────────────────────────────────────────────────────
if (WATCH_INTERVAL) {
  console.log(`[Swing Scanner] Watch mode: refresh every ${WATCH_INTERVAL / 1000}s. Ctrl+C to stop.`);
  scan();
  setInterval(scan, WATCH_INTERVAL);
} else {
  scan().catch(err => {
    console.error(err.message);
    process.exit(1);
  });
}
