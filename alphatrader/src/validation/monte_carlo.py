"""
Monte Carlo simulation for risk assessment.

Simulates many random trading scenarios to get confidence intervals
on returns, drawdowns, and other risk metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .metrics import calculate_metrics, max_drawdown


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    n_trades: int

    # Return distribution
    mean_return: float
    median_return: float
    return_5th_percentile: float
    return_95th_percentile: float
    return_std: float

    # Drawdown distribution
    mean_max_drawdown: float
    median_max_drawdown: float
    worst_drawdown: float  # 95th percentile of drawdowns
    drawdown_5th_percentile: float

    # Risk metrics
    probability_of_loss: float  # P(return < 0)
    probability_of_ruin: float  # P(drawdown > 50%)
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Sharpe distribution
    mean_sharpe: float
    sharpe_5th_percentile: float
    sharpe_95th_percentile: float

    # Raw data for further analysis
    all_returns: np.ndarray
    all_drawdowns: np.ndarray
    all_sharpes: np.ndarray


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategy risk assessment.

    Performs bootstrap resampling of trade returns to simulate
    many possible trading outcomes.
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: Optional[int] = 42,
    ):
        """
        Initialize simulator.

        Args:
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate_from_trades(
        self,
        trade_returns: np.ndarray,
        n_trades_per_sim: Optional[int] = None,
        periods_per_year: int = 252,
        verbose: bool = True,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation by resampling trade returns.

        Args:
            trade_returns: Array of historical trade returns
            n_trades_per_sim: Number of trades per simulation (default: same as input)
            periods_per_year: For Sharpe calculation
            verbose: Print progress

        Returns:
            MonteCarloResult with distribution statistics
        """
        if len(trade_returns) < 10:
            raise ValueError("Need at least 10 trades for Monte Carlo simulation")

        n_trades = n_trades_per_sim or len(trade_returns)

        if verbose:
            print(f"Monte Carlo Simulation: {self.n_simulations:,} simulations")
            print(f"  Historical trades: {len(trade_returns):,}")
            print(f"  Trades per simulation: {n_trades:,}")
            print()

        all_returns = np.zeros(self.n_simulations)
        all_drawdowns = np.zeros(self.n_simulations)
        all_sharpes = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # Bootstrap sample of trade returns
            sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)

            # Calculate cumulative return
            equity_curve = np.cumprod(1 + sampled_returns)
            total_return = equity_curve[-1] - 1
            all_returns[i] = total_return

            # Calculate max drawdown
            all_drawdowns[i] = max_drawdown(equity_curve)

            # Calculate Sharpe ratio
            if np.std(sampled_returns) > 1e-10:
                all_sharpes[i] = (np.mean(sampled_returns) / np.std(sampled_returns)
                                 * np.sqrt(periods_per_year))
            else:
                all_sharpes[i] = 0

        # Calculate statistics
        result = MonteCarloResult(
            n_simulations=self.n_simulations,
            n_trades=n_trades,

            # Returns
            mean_return=float(np.mean(all_returns)),
            median_return=float(np.median(all_returns)),
            return_5th_percentile=float(np.percentile(all_returns, 5)),
            return_95th_percentile=float(np.percentile(all_returns, 95)),
            return_std=float(np.std(all_returns)),

            # Drawdowns
            mean_max_drawdown=float(np.mean(all_drawdowns)),
            median_max_drawdown=float(np.median(all_drawdowns)),
            worst_drawdown=float(np.percentile(all_drawdowns, 95)),
            drawdown_5th_percentile=float(np.percentile(all_drawdowns, 5)),

            # Risk metrics
            probability_of_loss=float(np.mean(all_returns < 0)),
            probability_of_ruin=float(np.mean(all_drawdowns > 0.5)),
            var_95=float(-np.percentile(all_returns, 5)),  # 5% worst case
            cvar_95=float(-np.mean(all_returns[all_returns <= np.percentile(all_returns, 5)])),

            # Sharpe
            mean_sharpe=float(np.mean(all_sharpes)),
            sharpe_5th_percentile=float(np.percentile(all_sharpes, 5)),
            sharpe_95th_percentile=float(np.percentile(all_sharpes, 95)),

            # Raw data
            all_returns=all_returns,
            all_drawdowns=all_drawdowns,
            all_sharpes=all_sharpes,
        )

        if verbose:
            self._print_results(result)

        return result

    def simulate_from_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        return_per_correct: float = 0.01,  # 1% return per correct prediction
        return_per_wrong: float = -0.015,  # -1.5% per wrong prediction (asymmetric)
        confidence_threshold: float = 0.0,  # Only trade above this confidence
        verbose: bool = True,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation from model predictions.

        Converts predictions to hypothetical trade returns based on
        correctness and then runs simulation.

        Args:
            predictions: Model predictions (0, 1, 2)
            actuals: Actual labels (0, 1, 2)
            confidences: Confidence scores (optional)
            return_per_correct: Return when prediction is correct
            return_per_wrong: Return when prediction is wrong
            confidence_threshold: Only include predictions above this confidence
            verbose: Print progress

        Returns:
            MonteCarloResult
        """
        # Filter by confidence threshold if provided
        if confidences is not None and confidence_threshold > 0:
            mask = confidences >= confidence_threshold
            predictions = predictions[mask]
            actuals = actuals[mask]
            confidences = confidences[mask]

        # Convert predictions to trade returns
        correct = predictions == actuals
        trade_returns = np.where(correct, return_per_correct, return_per_wrong)

        if verbose:
            win_rate = np.mean(correct)
            print(f"Prediction-based Monte Carlo:")
            print(f"  Total predictions: {len(predictions):,}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Return per correct: {return_per_correct:.2%}")
            print(f"  Return per wrong: {return_per_wrong:.2%}")
            print()

        return self.simulate_from_trades(trade_returns, verbose=verbose)

    def _print_results(self, result: MonteCarloResult):
        """Print formatted results."""
        print("=" * 50)
        print("Monte Carlo Simulation Results")
        print("=" * 50)
        print()
        print("Return Distribution:")
        print(f"  Mean: {result.mean_return:.1%}")
        print(f"  Median: {result.median_return:.1%}")
        print(f"  5th percentile: {result.return_5th_percentile:.1%}")
        print(f"  95th percentile: {result.return_95th_percentile:.1%}")
        print(f"  Std Dev: {result.return_std:.1%}")
        print()
        print("Drawdown Distribution:")
        print(f"  Mean Max DD: {result.mean_max_drawdown:.1%}")
        print(f"  Median Max DD: {result.median_max_drawdown:.1%}")
        print(f"  95th percentile DD: {result.worst_drawdown:.1%}")
        print()
        print("Risk Metrics:")
        print(f"  P(Loss): {result.probability_of_loss:.1%}")
        print(f"  P(Ruin, DD>50%): {result.probability_of_ruin:.1%}")
        print(f"  VaR 95%: {result.var_95:.1%}")
        print(f"  CVaR 95%: {result.cvar_95:.1%}")
        print()
        print("Sharpe Ratio Distribution:")
        print(f"  Mean: {result.mean_sharpe:.2f}")
        print(f"  5th percentile: {result.sharpe_5th_percentile:.2f}")
        print(f"  95th percentile: {result.sharpe_95th_percentile:.2f}")


def quick_monte_carlo(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_simulations: int = 10000,
    verbose: bool = True,
) -> MonteCarloResult:
    """
    Quick Monte Carlo simulation from predictions with default settings.

    Args:
        predictions: Model predictions
        actuals: Actual labels
        n_simulations: Number of simulations
        verbose: Print results

    Returns:
        MonteCarloResult
    """
    sim = MonteCarloSimulator(n_simulations=n_simulations)
    return sim.simulate_from_predictions(predictions, actuals, verbose=verbose)
