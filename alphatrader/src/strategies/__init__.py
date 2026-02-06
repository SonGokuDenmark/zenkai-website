"""
Trading strategy signal generators - 72 strategies across all regimes.

MASSIVE STRATEGY LIBRARY:
- GENERAL/MIXED (7): Work across multiple regimes
- TRENDING UP (10): Bullish momentum, breakouts, uptrends
- TRENDING DOWN (10): Bearish momentum, breakdowns, downtrends
- RANGING TRADES (8): Mean reversion, channel trading
- VOLUME-BASED (9): OBV, CMF, Force Index, etc.
- VOLATILITY-BASED (9): ATR, BB expansion, HV rank, etc.
- MOMENTUM (9): ROC, TSI, Williams %R, Aroon, etc.
- PATTERN-BASED (10): Double top/bottom, candlestick patterns

Total: 72 strategies for comprehensive market coverage.

The AI will learn:
- Which strategies work in which regime
- When to go LONG (trending UP)
- When to go SHORT (trending DOWN)
- When to trade mean reversion (ranging)
- Signal confluence from multiple strategies
"""

# Base classes
from .base import SignalGenerator, Signal, SignalDirection

# General/Mixed strategies (work in multiple regimes) - 7
from .macd import MACDStrategy
from .ma_crossover import MACrossoverStrategy
from .stochastic import StochasticMRStrategy
from .turtle import TurtleStrategy
from .support_resistance import SupportResistanceStrategy
from .candlestick import CandlestickStrategy
from .rsi_divergence import RSIDivergenceStrategy

# Trending UP strategies - 10
from .trending_up import (
    EMAStackStrategy,
    GoldenCrossStrategy,
    ParabolicSARUpStrategy,
    ADXTrendUpStrategy,
    HigherHighsStrategy,
    MomentumBreakoutStrategy,
    VolumeSurgeUpStrategy,
    SupertrendLongStrategy,
    VWAPBreakoutStrategy,
    IchimokuBullishStrategy,
    get_trending_up_strategies,
)

# Trending DOWN strategies - 10
from .trending_down import (
    EMAStackDownStrategy,
    DeathCrossStrategy,
    ParabolicSARDownStrategy,
    ADXTrendDownStrategy,
    LowerLowsStrategy,
    MomentumBreakdownStrategy,
    VolumeSurgeDownStrategy,
    SupertrendShortStrategy,
    VWAPBreakdownStrategy,
    IchimokuBearishStrategy,
    get_trending_down_strategies,
)

# Ranging market TRADING strategies - 8
from .ranging_trades import (
    BBReversionStrategy,
    RSIReversionStrategy,
    StochReversionStrategy,
    RangeBreakFadeStrategy,
    KeltnerBounceStrategy,
    PivotBounceStrategy,
    MeanReversionStrategy,
    ChannelTradeStrategy,
    get_ranging_trade_strategies,
)

# Volume-based strategies - 9
from .volume_strategies import (
    OBVTrendStrategy,
    VolumeBreakoutStrategy,
    VolumeClimaxStrategy,
    AccumulationDistributionStrategy,
    ChaikinMoneyFlowStrategy,
    ForceIndexStrategy,
    VWMAStrategy,
    RelativeVolumeStrategy,
    MFIStrategy,
    get_volume_strategies,
)

# Volatility-based strategies - 9
from .volatility_strategies import (
    ATRBreakoutStrategy,
    BollingerExpansionStrategy,
    VolatilityBreakoutStrategy,
    HistoricalVolatilityStrategy,
    RangeExpansionStrategy,
    ATRChannelStrategy,
    VolatilityContractStrategy,
    StandardDeviationStrategy,
    TrueRangeStrategy,
    get_volatility_strategies,
)

# Momentum strategies - 9
from .momentum_strategies import (
    ROCStrategy,
    MomentumOscillatorStrategy,
    TSIStrategy,
    WilliamsRStrategy,
    UltimateOscillatorStrategy,
    AwesomeOscillatorStrategy,
    PPOStrategy,
    AroonStrategy,
    CCIStrategy,
    get_momentum_strategies,
)

# Pattern strategies - 10
from .pattern_strategies import (
    DoubleBottomStrategy,
    DoubleTopStrategy,
    ThreeWhiteSoldiersStrategy,
    ThreeBlackCrowsStrategy,
    MorningStarStrategy,
    EveningStarStrategy,
    EngulfingStrategy,
    HammerStrategy,
    DojiStrategy,
    InsideBarStrategy,
    get_pattern_strategies,
)


def get_general_strategies():
    """Get general/mixed regime strategies (7)."""
    return [
        MACDStrategy(),
        MACrossoverStrategy(),
        StochasticMRStrategy(),
        TurtleStrategy(),
        SupportResistanceStrategy(),
        CandlestickStrategy(),
        RSIDivergenceStrategy(),
    ]


def get_all_strategies():
    """
    Get ALL 72 trading strategy instances.

    Returns strategies organized by category:
    - 7 General (work across regimes)
    - 10 Trending UP (bullish markets)
    - 10 Trending DOWN (bearish markets)
    - 8 Ranging trades (mean reversion)
    - 9 Volume-based
    - 9 Volatility-based
    - 9 Momentum
    - 10 Pattern-based
    """
    strategies = []
    strategies.extend(get_general_strategies())       # 7
    strategies.extend(get_trending_up_strategies())   # 10
    strategies.extend(get_trending_down_strategies()) # 10
    strategies.extend(get_ranging_trade_strategies()) # 8
    strategies.extend(get_volume_strategies())        # 9
    strategies.extend(get_volatility_strategies())    # 9
    strategies.extend(get_momentum_strategies())      # 9
    strategies.extend(get_pattern_strategies())       # 10
    return strategies  # Total: 72


def count_strategies():
    """Count total strategies by category."""
    return {
        "general": len(get_general_strategies()),
        "trending_up": len(get_trending_up_strategies()),
        "trending_down": len(get_trending_down_strategies()),
        "ranging": len(get_ranging_trade_strategies()),
        "volume": len(get_volume_strategies()),
        "volatility": len(get_volatility_strategies()),
        "momentum": len(get_momentum_strategies()),
        "pattern": len(get_pattern_strategies()),
        "total": len(get_all_strategies()),
    }


__all__ = [
    # Base
    "SignalGenerator",
    "Signal",
    "SignalDirection",

    # General strategies
    "MACDStrategy",
    "MACrossoverStrategy",
    "StochasticMRStrategy",
    "TurtleStrategy",
    "SupportResistanceStrategy",
    "CandlestickStrategy",
    "RSIDivergenceStrategy",

    # Trending UP
    "EMAStackStrategy",
    "GoldenCrossStrategy",
    "ParabolicSARUpStrategy",
    "ADXTrendUpStrategy",
    "HigherHighsStrategy",
    "MomentumBreakoutStrategy",
    "VolumeSurgeUpStrategy",
    "SupertrendLongStrategy",
    "VWAPBreakoutStrategy",
    "IchimokuBullishStrategy",

    # Trending DOWN
    "EMAStackDownStrategy",
    "DeathCrossStrategy",
    "ParabolicSARDownStrategy",
    "ADXTrendDownStrategy",
    "LowerLowsStrategy",
    "MomentumBreakdownStrategy",
    "VolumeSurgeDownStrategy",
    "SupertrendShortStrategy",
    "VWAPBreakdownStrategy",
    "IchimokuBearishStrategy",

    # Ranging TRADES
    "BBReversionStrategy",
    "RSIReversionStrategy",
    "StochReversionStrategy",
    "RangeBreakFadeStrategy",
    "KeltnerBounceStrategy",
    "PivotBounceStrategy",
    "MeanReversionStrategy",
    "ChannelTradeStrategy",

    # Volume
    "OBVTrendStrategy",
    "VolumeBreakoutStrategy",
    "VolumeClimaxStrategy",
    "AccumulationDistributionStrategy",
    "ChaikinMoneyFlowStrategy",
    "ForceIndexStrategy",
    "VWMAStrategy",
    "RelativeVolumeStrategy",
    "MFIStrategy",

    # Volatility
    "ATRBreakoutStrategy",
    "BollingerExpansionStrategy",
    "VolatilityBreakoutStrategy",
    "HistoricalVolatilityStrategy",
    "RangeExpansionStrategy",
    "ATRChannelStrategy",
    "VolatilityContractStrategy",
    "StandardDeviationStrategy",
    "TrueRangeStrategy",

    # Momentum
    "ROCStrategy",
    "MomentumOscillatorStrategy",
    "TSIStrategy",
    "WilliamsRStrategy",
    "UltimateOscillatorStrategy",
    "AwesomeOscillatorStrategy",
    "PPOStrategy",
    "AroonStrategy",
    "CCIStrategy",

    # Pattern
    "DoubleBottomStrategy",
    "DoubleTopStrategy",
    "ThreeWhiteSoldiersStrategy",
    "ThreeBlackCrowsStrategy",
    "MorningStarStrategy",
    "EveningStarStrategy",
    "EngulfingStrategy",
    "HammerStrategy",
    "DojiStrategy",
    "InsideBarStrategy",

    # Getters
    "get_general_strategies",
    "get_trending_up_strategies",
    "get_trending_down_strategies",
    "get_ranging_trade_strategies",
    "get_volume_strategies",
    "get_volatility_strategies",
    "get_momentum_strategies",
    "get_pattern_strategies",
    "get_all_strategies",
    "count_strategies",
]
