#!/usr/bin/env python3
"""
Test script for RiskManager.

Runs through simulated scenarios to verify all risk checks work correctly.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.risk import RiskManager, RiskLimits, RiskAction, TradeRecord


def test_fat_finger_check():
    """Test that orders with large price deviation are blocked."""
    print("\n" + "=" * 50)
    print("Test 1: Fat-Finger Price Check")
    print("=" * 50)

    rm = RiskManager(initial_balance=10000)

    # Normal trade (0.2% deviation) - should pass
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50100,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  0.2% deviation: {result.action.value} - {result.reason}")
    assert result.is_allowed, "Normal trade should be allowed"

    # Fat-finger trade (10% deviation) - should be blocked
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=55000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  10% deviation: {result.action.value} - {result.reason}")
    assert not result.is_allowed, "Fat-finger trade should be blocked"
    assert result.action == RiskAction.BLOCK

    print("  [OK] Fat-finger check PASSED")


def test_daily_drawdown_limit():
    """Test daily drawdown limit triggers correctly."""
    print("\n" + "=" * 50)
    print("Test 2: Daily Drawdown Limit")
    print("=" * 50)

    # Disable loss pause for this test
    limits = RiskLimits(consecutive_loss_pause=100)
    rm = RiskManager(initial_balance=10000, limits=limits)

    # Record losing trades until daily limit hit
    for i in range(6):
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="sell",
            price=50000,
            quantity=0.01,
            pnl=-100,  # $100 loss each
            is_winner=False,
        )
        rm.record_trade(trade)
        print(f"  Trade {i+1}: PnL=-$100, Daily total: ${rm.daily_pnl:,.2f}")

    # Next trade should be blocked (daily loss = $600 = 6% > 5% limit)
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  After 6% daily loss: {result.action.value} - {result.reason}")
    assert not result.is_allowed, "Trade should be blocked after daily limit"

    print("  [OK] Daily drawdown limit PASSED")


def test_weekly_drawdown_limit():
    """Test weekly drawdown limit triggers correctly."""
    print("\n" + "=" * 50)
    print("Test 3: Weekly Drawdown Limit")
    print("=" * 50)

    # Use higher daily limit to test weekly, disable loss pause
    limits = RiskLimits(daily_drawdown_limit_pct=20.0, weekly_drawdown_limit_pct=10.0, consecutive_loss_pause=100)
    rm = RiskManager(initial_balance=10000, limits=limits)

    # Record losing trades until weekly limit hit
    for i in range(11):
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="sell",
            price=50000,
            quantity=0.01,
            pnl=-100,
            is_winner=False,
        )
        rm.record_trade(trade)

    print(f"  Weekly P&L: ${rm.weekly_pnl:,.2f} ({rm.weekly_pnl/rm.initial_balance*100:.1f}%)")

    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  After 11% weekly loss: {result.action.value} - {result.reason}")
    assert not result.is_allowed, "Trade should be blocked after weekly limit"

    print("  [OK] Weekly drawdown limit PASSED")


def test_total_drawdown_kill_switch():
    """Test that total drawdown triggers emergency shutdown."""
    print("\n" + "=" * 50)
    print("Test 4: Total Drawdown Kill Switch")
    print("=" * 50)

    shutdown_triggered = []

    def on_shutdown(reason):
        shutdown_triggered.append(reason)
        print(f"  SHUTDOWN CALLBACK: {reason}")

    limits = RiskLimits(
        daily_drawdown_limit_pct=50.0,
        weekly_drawdown_limit_pct=50.0,
        total_drawdown_limit_pct=15.0,
        consecutive_loss_pause=100,  # Disable for this test
    )
    rm = RiskManager(initial_balance=10000, limits=limits, on_shutdown=on_shutdown)

    # Lose 16% of balance
    for i in range(16):
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="sell",
            price=50000,
            quantity=0.01,
            pnl=-100,
            is_winner=False,
        )
        rm.record_trade(trade)

    print(f"  Current balance: ${rm.current_balance:,.2f} (lost ${-rm.total_pnl:,.2f})")

    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  After 16% total loss: {result.action.value} - {result.reason}")

    assert result.action == RiskAction.SHUTDOWN, "Should trigger shutdown"
    assert rm.is_shutdown, "System should be shut down"
    assert len(shutdown_triggered) == 1, "Shutdown callback should be called"

    print("  [OK] Kill switch PASSED")


def test_consecutive_loss_pause():
    """Test that consecutive losses trigger a trading pause."""
    print("\n" + "=" * 50)
    print("Test 5: Consecutive Loss Pause")
    print("=" * 50)

    limits = RiskLimits(
        daily_drawdown_limit_pct=50.0,
        consecutive_loss_pause=5,
        loss_pause_duration_minutes=60,
    )
    rm = RiskManager(initial_balance=10000, limits=limits)

    # Record 5 consecutive losses
    for i in range(5):
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="sell",
            price=50000,
            quantity=0.01,
            pnl=-50,
            is_winner=False,
        )
        rm.record_trade(trade)
        print(f"  Loss {i+1}: consecutive_losses={rm.consecutive_losses}")

    assert rm.paused_until is not None, "Should be paused after 5 losses"
    print(f"  Paused until: {rm.paused_until}")

    # Next trade should be blocked
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  Trade during pause: {result.action.value} - {result.reason}")
    assert not result.is_allowed, "Trade should be blocked during pause"

    # Simulate pause expiry
    rm.paused_until = datetime.now() - timedelta(minutes=1)

    # Now trade should be allowed
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  Trade after pause: {result.action.value} - {result.reason}")
    assert result.is_allowed, "Trade should be allowed after pause expires"

    # Record a win to reset streak
    win_trade = TradeRecord(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.01,
        pnl=100,
        is_winner=True,
    )
    rm.record_trade(win_trade)
    print(f"  After win: consecutive_losses={rm.consecutive_losses}")
    assert rm.consecutive_losses == 0, "Win should reset loss streak"

    print("  [OK] Consecutive loss pause PASSED")


def test_rate_limiting():
    """Test trade rate limiting."""
    print("\n" + "=" * 50)
    print("Test 6: Rate Limiting")
    print("=" * 50)

    limits = RiskLimits(
        daily_drawdown_limit_pct=50.0,
        max_trades_per_minute=3,
        max_trades_per_hour=10,
    )
    rm = RiskManager(initial_balance=10000, limits=limits)

    # Record trades up to per-minute limit
    for i in range(3):
        trade = TradeRecord(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side="buy",
            price=50000,
            quantity=0.01,
            pnl=10,
            is_winner=True,
        )
        rm.record_trade(trade)

    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  After 3 trades/minute: {result.action.value} - {result.reason}")
    assert not result.is_allowed, "Trade should be blocked by rate limit"

    print("  [OK] Rate limiting PASSED")


def test_position_size_limit():
    """Test position size limit check."""
    print("\n" + "=" * 50)
    print("Test 7: Position Size Limit")
    print("=" * 50)

    limits = RiskLimits(max_position_size_pct=20.0)
    rm = RiskManager(initial_balance=10000, limits=limits)

    # Small position - should pass
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.02,
        market_price=50000,
        position_value=1000,  # 10% of portfolio
    )
    print(f"  10% position: {result.action.value} - {result.reason}")
    assert result.is_allowed, "Small position should be allowed"

    # Large position - should suggest reduce
    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
        position_value=3000,  # 30% of portfolio
    )
    print(f"  30% position: {result.action.value} - {result.reason}")
    assert result.action == RiskAction.REDUCE, "Large position should suggest reduce"

    print("  [OK] Position size limit PASSED")


def test_status_reporting():
    """Test status reporting."""
    print("\n" + "=" * 50)
    print("Test 8: Status Reporting")
    print("=" * 50)

    rm = RiskManager(initial_balance=10000)

    # Record some trades
    rm.record_trade(TradeRecord(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.01,
        pnl=150,
        is_winner=True,
    ))
    rm.record_trade(TradeRecord(
        timestamp=datetime.now(),
        symbol="BTCUSDT",
        side="sell",
        price=50000,
        quantity=0.01,
        pnl=-50,
        is_winner=False,
    ))

    status = rm.get_status()
    print(f"  Status: {status}")
    print(f"  RiskManager repr: {rm}")

    assert status["current_balance"] == 10100, "Balance should be $10,100"
    assert status["total_trades"] == 2, "Should have 2 trades"
    assert status["is_shutdown"] == False, "Should not be shutdown"

    print("  [OK] Status reporting PASSED")


def test_manual_shutdown():
    """Test manual emergency shutdown."""
    print("\n" + "=" * 50)
    print("Test 9: Manual Emergency Shutdown")
    print("=" * 50)

    rm = RiskManager(initial_balance=10000)

    # Trigger manual shutdown
    rm.emergency_shutdown("Manual test shutdown")

    assert rm.is_shutdown, "Should be shut down"
    assert rm.shutdown_reason == "Manual test shutdown"

    result = rm.check_trade(
        symbol="BTCUSDT",
        side="buy",
        price=50000,
        quantity=0.1,
        market_price=50000,
    )
    print(f"  Trade after shutdown: {result.action.value} - {result.reason}")
    assert not result.is_allowed, "Trade should be blocked after shutdown"

    print("  [OK] Manual shutdown PASSED")


def main():
    print("=" * 50)
    print("RiskManager Test Suite")
    print("=" * 50)

    try:
        test_fat_finger_check()
        test_daily_drawdown_limit()
        test_weekly_drawdown_limit()
        test_total_drawdown_kill_switch()
        test_consecutive_loss_pause()
        test_rate_limiting()
        test_position_size_limit()
        test_status_reporting()
        test_manual_shutdown()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED [OK]")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
