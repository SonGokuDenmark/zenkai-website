"""
Performance metrics for trading strategy evaluation.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily, 365*24 for hourly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess < 1e-10:
        return 0.0

    return mean_excess / std_excess * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (penalizes only downside volatility).

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    mean_excess = np.mean(excess_returns)

    # Downside deviation - only negative returns
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) < 2:
        return float('inf') if mean_excess > 0 else 0.0

    downside_std = np.std(downside_returns, ddof=1)

    if downside_std < 1e-10:
        return float('inf') if mean_excess > 0 else 0.0

    return mean_excess / downside_std * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Array of cumulative equity values

    Returns:
        Maximum drawdown as a positive decimal (0.25 = 25% drawdown)
    """
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    return float(np.max(drawdowns))


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Array of period returns
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    # Build equity curve
    equity = np.cumprod(1 + returns)
    mdd = max_drawdown(equity)

    if mdd < 1e-10:
        return float('inf')

    # Annualized return
    total_return = equity[-1] - 1
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    return annual_return / mdd


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of trade/period returns

    Returns:
        Win rate as decimal (0.55 = 55%)
    """
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of trade returns

    Returns:
        Profit factor (>1 is profitable)
    """
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses < 1e-10:
        return float('inf') if gains > 0 else 0.0

    return gains / losses


def expectancy(returns: np.ndarray) -> float:
    """
    Calculate expectancy (average return per trade).

    Args:
        returns: Array of trade returns

    Returns:
        Expected return per trade
    """
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))


def calculate_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Calculate comprehensive trading metrics.

    Args:
        returns: Array of period/trade returns
        periods_per_year: Trading periods per year
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of metrics
    """
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "n_trades": 0,
            "volatility": 0.0,
        }

    # Build equity curve
    equity = np.cumprod(1 + returns)

    # Total and annual return
    total_return = equity[-1] - 1
    n_periods = len(returns)
    if n_periods > 0:
        annual_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    else:
        annual_return = 0.0

    # Volatility (annualized)
    volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "expectancy": expectancy(returns),
        "n_trades": len(returns),
        "volatility": float(volatility),
    }


def calculate_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate classification metrics for predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidences: Optional confidence scores

    Returns:
        Dictionary of classification metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Class names for 3-class problem
    class_names = ["DOWN", "FLAT", "UP"]

    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(precision):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]) if i < len(support) else 0,
            }

    metrics = {
        "accuracy": float(accuracy),
        "per_class": per_class,
    }

    # Confidence calibration if provided
    if confidences is not None:
        correct = y_true == y_pred
        # Correlation between confidence and correctness
        if len(confidences) > 1:
            metrics["confidence_correlation"] = float(np.corrcoef(confidences, correct)[0, 1])
        metrics["mean_confidence_correct"] = float(np.mean(confidences[correct])) if correct.any() else 0.0
        metrics["mean_confidence_wrong"] = float(np.mean(confidences[~correct])) if (~correct).any() else 0.0

    return metrics
