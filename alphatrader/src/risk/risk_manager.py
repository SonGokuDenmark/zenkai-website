"""
Risk management module for AlphaTrader.

Provides pre-trade checks and position monitoring to prevent catastrophic losses.
All limits are configurable and can be adjusted per-symbol or globally.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Actions that can be taken when a risk limit is breached."""
    ALLOW = "allow"           # Trade is allowed
    BLOCK = "block"           # Trade is blocked
    REDUCE = "reduce"         # Trade size should be reduced
    CLOSE_ALL = "close_all"   # Close all positions immediately
    SHUTDOWN = "shutdown"     # Emergency shutdown - stop all trading


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    action: RiskAction
    reason: str
    details: Dict = field(default_factory=dict)

    @property
    def is_allowed(self) -> bool:
        return self.action == RiskAction.ALLOW


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    pnl: float  # Realized P&L
    is_winner: bool


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    # Price deviation limits
    max_price_deviation_pct: float = 5.0  # Fat-finger: max % from market price

    # Drawdown limits
    daily_drawdown_limit_pct: float = 5.0    # Max daily loss %
    weekly_drawdown_limit_pct: float = 10.0  # Max weekly loss %
    total_drawdown_limit_pct: float = 15.0   # Kill switch threshold %

    # Position limits
    max_position_size_pct: float = 20.0      # Max % of portfolio per position
    max_positions: int = 10                   # Max concurrent positions

    # Loss streak limits
    consecutive_loss_pause: int = 5          # Pause after N consecutive losses
    loss_pause_duration_minutes: int = 60    # How long to pause

    # Rate limits
    max_trades_per_minute: int = 10          # Prevent runaway trading
    max_trades_per_hour: int = 100           # Hourly trade limit


class RiskManager:
    """
    Risk management system for trading.

    Implements multiple layers of protection:
    1. Pre-trade checks (price validation, position limits)
    2. Drawdown monitoring (daily, weekly, total)
    3. Loss streak detection
    4. Rate limiting
    5. Emergency shutdown capability

    Usage:
        risk_mgr = RiskManager(initial_balance=10000)

        # Before each trade
        result = risk_mgr.check_trade(
            symbol="BTCUSDT",
            side="buy",
            price=50000,
            quantity=0.1,
            market_price=50100
        )

        if result.is_allowed:
            # Execute trade
            pass
        else:
            print(f"Trade blocked: {result.reason}")

        # After trade completes
        risk_mgr.record_trade(...)

        # Periodic check
        if risk_mgr.should_shutdown():
            # Emergency shutdown
            pass
    """

    def __init__(
        self,
        initial_balance: float,
        limits: Optional[RiskLimits] = None,
        on_shutdown: Optional[Callable] = None,
    ):
        """
        Initialize risk manager.

        Args:
            initial_balance: Starting portfolio value
            limits: Risk limits configuration
            on_shutdown: Callback function for emergency shutdown
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.limits = limits or RiskLimits()
        self.on_shutdown = on_shutdown

        # Tracking state
        self.trade_history: List[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.peak_balance: float = initial_balance

        # Loss streak tracking
        self.consecutive_losses: int = 0
        self.paused_until: Optional[datetime] = None

        # Rate limiting
        self.recent_trades: List[datetime] = []

        # State
        self.is_shutdown: bool = False
        self.shutdown_reason: Optional[str] = None

        # Daily/weekly reset tracking
        self.last_daily_reset: datetime = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self.last_weekly_reset: datetime = self._get_week_start()

        logger.info(f"RiskManager initialized with balance=${initial_balance:,.2f}")

    def _get_week_start(self) -> datetime:
        """Get the start of the current week (Monday 00:00)."""
        now = datetime.now()
        days_since_monday = now.weekday()
        return (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def _maybe_reset_periods(self):
        """Reset daily/weekly P&L if new period started."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = self._get_week_start()

        if today_start > self.last_daily_reset:
            logger.info(f"Daily reset: previous day P&L was ${self.daily_pnl:,.2f}")
            self.daily_pnl = 0.0
            self.last_daily_reset = today_start

        if week_start > self.last_weekly_reset:
            logger.info(f"Weekly reset: previous week P&L was ${self.weekly_pnl:,.2f}")
            self.weekly_pnl = 0.0
            self.last_weekly_reset = week_start

    def check_trade(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        market_price: float,
        position_value: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Run all pre-trade risk checks.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            price: Order price
            quantity: Order quantity
            market_price: Current market price
            position_value: Current position value (for position limit check)

        Returns:
            RiskCheckResult with action and reason
        """
        self._maybe_reset_periods()

        # Check 1: Emergency shutdown
        if self.is_shutdown:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Trading shutdown: {self.shutdown_reason}",
            )

        # Check 2: Pause after loss streak
        if self.paused_until and datetime.now() < self.paused_until:
            remaining = (self.paused_until - datetime.now()).total_seconds() / 60
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Trading paused for {remaining:.0f} more minutes after {self.limits.consecutive_loss_pause} consecutive losses",
            )

        # Check 3: Fat-finger price check
        price_check = self._check_price_deviation(price, market_price, side)
        if not price_check.is_allowed:
            return price_check

        # Check 4: Drawdown limits
        drawdown_check = self._check_drawdown_limits()
        if not drawdown_check.is_allowed:
            return drawdown_check

        # Check 5: Position size limit
        if position_value is not None:
            position_check = self._check_position_size(position_value)
            if not position_check.is_allowed:
                return position_check

        # Check 6: Rate limiting
        rate_check = self._check_rate_limits()
        if not rate_check.is_allowed:
            return rate_check

        # All checks passed
        return RiskCheckResult(
            action=RiskAction.ALLOW,
            reason="All risk checks passed",
            details={
                "daily_pnl": self.daily_pnl,
                "weekly_pnl": self.weekly_pnl,
                "total_pnl": self.total_pnl,
                "consecutive_losses": self.consecutive_losses,
            }
        )

    def _check_price_deviation(
        self,
        order_price: float,
        market_price: float,
        side: str,
    ) -> RiskCheckResult:
        """Check if order price deviates too much from market (fat-finger check)."""
        if market_price <= 0:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason="Invalid market price",
            )

        deviation_pct = abs(order_price - market_price) / market_price * 100

        if deviation_pct > self.limits.max_price_deviation_pct:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Price deviation {deviation_pct:.1f}% exceeds limit of {self.limits.max_price_deviation_pct}%",
                details={
                    "order_price": order_price,
                    "market_price": market_price,
                    "deviation_pct": deviation_pct,
                }
            )

        return RiskCheckResult(action=RiskAction.ALLOW, reason="Price OK")

    def _check_drawdown_limits(self) -> RiskCheckResult:
        """Check if any drawdown limit is breached."""
        # Calculate current drawdown from peak
        if self.peak_balance > 0:
            total_drawdown_pct = (self.peak_balance - self.current_balance) / self.peak_balance * 100
        else:
            total_drawdown_pct = 0

        # Daily drawdown
        daily_drawdown_pct = abs(min(0, self.daily_pnl)) / self.initial_balance * 100
        if daily_drawdown_pct >= self.limits.daily_drawdown_limit_pct:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Daily drawdown {daily_drawdown_pct:.1f}% exceeds limit of {self.limits.daily_drawdown_limit_pct}%",
                details={"daily_pnl": self.daily_pnl, "daily_drawdown_pct": daily_drawdown_pct}
            )

        # Weekly drawdown
        weekly_drawdown_pct = abs(min(0, self.weekly_pnl)) / self.initial_balance * 100
        if weekly_drawdown_pct >= self.limits.weekly_drawdown_limit_pct:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Weekly drawdown {weekly_drawdown_pct:.1f}% exceeds limit of {self.limits.weekly_drawdown_limit_pct}%",
                details={"weekly_pnl": self.weekly_pnl, "weekly_drawdown_pct": weekly_drawdown_pct}
            )

        # Total drawdown (kill switch)
        if total_drawdown_pct >= self.limits.total_drawdown_limit_pct:
            self._trigger_shutdown(
                f"Total drawdown {total_drawdown_pct:.1f}% exceeds kill switch limit of {self.limits.total_drawdown_limit_pct}%"
            )
            return RiskCheckResult(
                action=RiskAction.SHUTDOWN,
                reason=self.shutdown_reason,
                details={"total_drawdown_pct": total_drawdown_pct}
            )

        return RiskCheckResult(action=RiskAction.ALLOW, reason="Drawdown OK")

    def _check_position_size(self, position_value: float) -> RiskCheckResult:
        """Check if position size is within limits."""
        position_pct = position_value / self.current_balance * 100

        if position_pct > self.limits.max_position_size_pct:
            return RiskCheckResult(
                action=RiskAction.REDUCE,
                reason=f"Position size {position_pct:.1f}% exceeds limit of {self.limits.max_position_size_pct}%",
                details={"position_value": position_value, "position_pct": position_pct}
            )

        return RiskCheckResult(action=RiskAction.ALLOW, reason="Position size OK")

    def _check_rate_limits(self) -> RiskCheckResult:
        """Check trading rate limits."""
        now = datetime.now()

        # Clean up old trades
        one_hour_ago = now - timedelta(hours=1)
        self.recent_trades = [t for t in self.recent_trades if t > one_hour_ago]

        # Check hourly limit
        if len(self.recent_trades) >= self.limits.max_trades_per_hour:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Hourly trade limit ({self.limits.max_trades_per_hour}) reached",
            )

        # Check per-minute limit
        one_minute_ago = now - timedelta(minutes=1)
        trades_last_minute = sum(1 for t in self.recent_trades if t > one_minute_ago)
        if trades_last_minute >= self.limits.max_trades_per_minute:
            return RiskCheckResult(
                action=RiskAction.BLOCK,
                reason=f"Per-minute trade limit ({self.limits.max_trades_per_minute}) reached",
            )

        return RiskCheckResult(action=RiskAction.ALLOW, reason="Rate OK")

    def record_trade(self, trade: TradeRecord):
        """
        Record a completed trade and update risk metrics.

        Args:
            trade: Completed trade record
        """
        self.trade_history.append(trade)
        self.recent_trades.append(trade.timestamp)

        # Update P&L
        self.daily_pnl += trade.pnl
        self.weekly_pnl += trade.pnl
        self.total_pnl += trade.pnl
        self.current_balance += trade.pnl

        # Update peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Update loss streak
        if trade.is_winner:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

            # Check for loss streak pause
            if self.consecutive_losses >= self.limits.consecutive_loss_pause:
                self.paused_until = datetime.now() + timedelta(
                    minutes=self.limits.loss_pause_duration_minutes
                )
                logger.warning(
                    f"Trading paused until {self.paused_until} after {self.consecutive_losses} consecutive losses"
                )

        logger.info(
            f"Trade recorded: {trade.symbol} {trade.side} PnL=${trade.pnl:,.2f} "
            f"(Daily: ${self.daily_pnl:,.2f}, Total: ${self.total_pnl:,.2f})"
        )

    def _trigger_shutdown(self, reason: str):
        """Trigger emergency shutdown."""
        self.is_shutdown = True
        self.shutdown_reason = reason
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")

        if self.on_shutdown:
            try:
                self.on_shutdown(reason)
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")

    def should_shutdown(self) -> bool:
        """Check if system should be shut down."""
        return self.is_shutdown

    def emergency_shutdown(self, reason: str = "Manual shutdown"):
        """Manually trigger emergency shutdown."""
        self._trigger_shutdown(reason)

    def reset_pause(self):
        """Manually reset the trading pause (use with caution)."""
        self.paused_until = None
        self.consecutive_losses = 0
        logger.info("Trading pause manually reset")

    def get_status(self) -> Dict:
        """Get current risk status summary."""
        return {
            "is_shutdown": self.is_shutdown,
            "shutdown_reason": self.shutdown_reason,
            "is_paused": self.paused_until is not None and datetime.now() < self.paused_until,
            "paused_until": self.paused_until.isoformat() if self.paused_until else None,
            "current_balance": self.current_balance,
            "initial_balance": self.initial_balance,
            "peak_balance": self.peak_balance,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "total_pnl": self.total_pnl,
            "consecutive_losses": self.consecutive_losses,
            "trades_today": sum(
                1 for t in self.trade_history
                if t.timestamp.date() == datetime.now().date()
            ),
            "total_trades": len(self.trade_history),
            "drawdown_from_peak_pct": (
                (self.peak_balance - self.current_balance) / self.peak_balance * 100
                if self.peak_balance > 0 else 0
            ),
        }

    def __repr__(self) -> str:
        status = self.get_status()
        return (
            f"RiskManager(balance=${status['current_balance']:,.2f}, "
            f"daily_pnl=${status['daily_pnl']:,.2f}, "
            f"shutdown={status['is_shutdown']}, "
            f"paused={status['is_paused']})"
        )
