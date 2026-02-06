# ⚡ Zenkai Edge — TradingView Indicator Spec
## Pine Script v6 | Smart Money Concepts
### Zenkai Corporation — February 2026

---

## Overview

**Zenkai Edge** is a free TradingView indicator that detects smart money footprints
in the market. Built for traders who want institutional-grade analysis without
the complexity.

**What it does:**
- Detects Fair Value Gaps (FVGs) — bullish and bearish
- Tracks FVG mitigation (price returning to fill gaps)
- Identifies Market Structure shifts (Break of Structure / Change of Character)
- Shows swing points (highs and lows)
- Dashboard overlay with current bias and active FVG counts

---

## Technical Specification

### Module 1: Fair Value Gaps (FVGs)

**Detection Logic:**
- Bullish FVG: `candle[2].high < candle[0].low` (gap between wick of candle before and after)
- Bearish FVG: `candle[2].low > candle[0].high` (inverse gap)
- Minimum gap size configurable (default: 0 — show all)

**Visualization:**
- Bullish FVGs: semi-transparent green boxes (#00ff88, 80% transparency)
- Bearish FVGs: semi-transparent red boxes (#ff4466, 80% transparency)
- Boxes extend right until mitigation or max bars

**Mitigation Tracking:**
- Option 1: Remove box when price enters the FVG zone
- Option 2: Grey out / reduce opacity on mitigation
- Option 3: Keep all FVGs visible regardless
- User-configurable via dropdown input

**Performance:**
- Max 500 boxes at any time (Pine Script limit management)
- Auto-cleanup of oldest FVGs when limit reached

### Module 2: Market Structure (BOS / CHoCH)

**Swing Point Detection:**
- Uses `ta.pivothigh()` and `ta.pivotlow()` with configurable lookback
- Default lookback: 5 bars left, 5 bars right
- Swing highs marked with small labels above
- Swing lows marked with small labels below

**Break of Structure (BOS):**
- Bullish BOS: Price breaks above the most recent swing high in a bullish trend
- Bearish BOS: Price breaks below the most recent swing low in a bearish trend
- Confirms continuation of existing trend
- Drawn as dashed line from broken swing point

**Change of Character (CHoCH):**
- Bullish CHoCH: Price breaks above swing high during a BEARISH trend (reversal signal)
- Bearish CHoCH: Price breaks below swing low during a BULLISH trend (reversal signal)
- Signals potential trend reversal
- Drawn as solid line from broken swing point, different color

**Bias Tracking:**
- Internal state machine tracks current market bias (Bullish / Bearish / Neutral)
- Updated on each BOS/CHoCH event
- Displayed in dashboard

### Module 3: Dashboard

**Position:** Top-right corner of chart (configurable)
**Contents:**
- Current Structure Bias: Bullish / Bearish / Neutral
- Active Bullish FVGs: count
- Active Bearish FVGs: count
- Last Structure Event: "BOS ▲" or "CHoCH ▼" etc.

**Style:**
- Semi-transparent dark background
- Zenkai brand colors
- Monospace font

---

## User Inputs (Grouped)

### FVG Settings
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| Show FVGs | bool | true | Toggle FVG detection |
| FVG Mitigation Mode | dropdown | "Remove" | Remove / Grey Out / Keep All |
| Min FVG Size (ticks) | int | 0 | Minimum gap size to display |
| Max FVG Age (bars) | int | 500 | Auto-remove FVGs older than X bars |
| Bullish FVG Color | color | #00ff88 | Customizable |
| Bearish FVG Color | color | #ff4466 | Customizable |

### Structure Settings
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| Show Market Structure | bool | true | Toggle BOS/CHoCH detection |
| Swing Lookback | int | 5 | Pivot point lookback period |
| Show Swing Points | bool | true | Toggle swing high/low labels |
| BOS Color | color | #00d2ff | Break of Structure line color |
| CHoCH Color | color | #b366ff | Change of Character line color |

### Dashboard Settings
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| Show Dashboard | bool | true | Toggle dashboard visibility |
| Dashboard Position | dropdown | "Top Right" | Corner selection |
| Dashboard Size | dropdown | "Normal" | Small / Normal / Large |

---

## TradingView Publishing

**Title:** Zenkai Edge — Smart Money Structure & FVGs
**Category:** Trend Analysis
**Tags:** smartmoney, fairvaluegap, fvg, marketstructure, bos, choch, institutional, priceaction, zenkai

**Description (pre-written):**
```
⚡ Zenkai Edge — Smart Money Concepts Made Simple

Detect institutional footprints with Fair Value Gaps and Market Structure analysis.

FEATURES:
• Fair Value Gap Detection — Automatically identifies bullish and bearish FVGs
• Mitigation Tracking — See when gaps get filled by returning price action
• Market Structure — Break of Structure (BOS) and Change of Character (CHoCH)
• Swing Point Detection — Key highs and lows that define market structure
• Real-time Dashboard — Current bias, active FVG count, last structure event
• Fully Customizable — Colors, sizes, and display options

HOW TO USE:
1. Add to your chart
2. Look for FVGs forming after impulsive moves
3. Watch for BOS (trend continuation) and CHoCH (trend reversal) signals
4. Use the dashboard to track overall market bias
5. Combine with your own analysis for confluence

Built by traders, for traders.
⚡ Zenkai Corporation — Evolve or Die

Free to use. Premium version with Order Blocks, Liquidity Sweeps,
and Smart Alerts coming soon.

Website: zenkaicorp.com
```

---

## Quality Standards

- [ ] No repainting — all signals based on confirmed (closed) candles
- [ ] Performance optimized — stays under Pine Script object limits
- [ ] Clean code — well-commented, organized by module
- [ ] Visual clarity — not cluttered, colors distinguish clearly
- [ ] Error-free — no runtime errors on any timeframe or symbol
- [ ] Tested on: BTC/USDT 4H, EUR/USD 1H, SPY Daily (minimum)

---

*Generated by Bulma — February 6, 2026*
*⚡ Zenkai Corporation*
