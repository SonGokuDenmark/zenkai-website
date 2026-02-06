"""
Walk-Forward Optimization (WFO) for model validation.

Tests if model generalizes across time by training on rolling windows
and testing on subsequent periods.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from datetime import datetime


@dataclass
class WFOFold:
    """Single fold results from WFO."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int
    accuracy: float
    class_accuracies: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    confidences: np.ndarray


@dataclass
class WFOResult:
    """Complete WFO validation results."""
    folds: List[WFOFold]
    mean_accuracy: float
    std_accuracy: float
    wfe: float  # Walk-Forward Efficiency
    class_accuracies: Dict[str, float]
    total_test_samples: int


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization validator.

    Splits time series data into rolling train/test windows to validate
    that a model generalizes across different time periods.

    Example with months:
        With 2 years of data, 6-month train, 2-month test:
        Fold 1: Train Jan-Jun 2022, Test Jul-Aug 2022
        Fold 2: Train Mar-Aug 2022, Test Sep-Oct 2022
        ...

    Example with days:
        With 2 months of data, 21-day train, 7-day test:
        Fold 1: Train Dec 1-21, Test Dec 22-28
        Fold 2: Train Dec 8-28, Test Dec 29-Jan 4
        ...
    """

    def __init__(
        self,
        train_periods: int = 6,
        test_periods: int = 2,
        step_periods: int = 2,
        period_type: str = "months",  # "months" or "days"
        min_train_samples: int = 1000,
        # Legacy parameters for backwards compatibility
        train_months: int = None,
        test_months: int = None,
        step_months: int = None,
    ):
        """
        Initialize WFO validator.

        Args:
            train_periods: Number of periods for training each fold
            test_periods: Number of periods for testing each fold
            step_periods: Periods to step forward between folds
            period_type: "months" or "days"
            min_train_samples: Minimum training samples required per fold
        """
        # Handle legacy parameters
        if train_months is not None:
            train_periods = train_months
            period_type = "months"
        if test_months is not None:
            test_periods = test_months
        if step_months is not None:
            step_periods = step_months

        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_periods = step_periods
        self.period_type = period_type
        self.min_train_samples = min_train_samples

        # For backwards compatibility
        self.train_months = train_periods if period_type == "months" else None
        self.test_months = test_periods if period_type == "months" else None
        self.step_months = step_periods if period_type == "months" else None

    def create_folds(
        self,
        timestamps: np.ndarray,
        n_samples: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create train/test index splits based on timestamps.

        Args:
            timestamps: Array of datetime timestamps for each sample
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        if len(timestamps) != n_samples:
            raise ValueError(f"timestamps length {len(timestamps)} != n_samples {n_samples}")

        # Convert to pandas for easier date manipulation
        # Handle millisecond timestamps (common in financial data)
        ts = pd.to_datetime(timestamps, unit='ms', errors='coerce')
        # Fallback if already datetime
        if ts.isna().all():
            ts = pd.to_datetime(timestamps)
        min_date = ts.min()
        max_date = ts.max()

        print(f"  Data range: {min_date} to {max_date} ({(max_date - min_date).days} days)")

        # Determine offset type
        if self.period_type == "days":
            train_offset = pd.DateOffset(days=self.train_periods)
            test_offset = pd.DateOffset(days=self.test_periods)
            step_offset = pd.DateOffset(days=self.step_periods)
        else:  # months
            train_offset = pd.DateOffset(months=self.train_periods)
            test_offset = pd.DateOffset(months=self.test_periods)
            step_offset = pd.DateOffset(months=self.step_periods)

        folds = []
        current_train_start = min_date

        while True:
            # Calculate fold boundaries
            train_end = current_train_start + train_offset
            test_start = train_end
            test_end = test_start + test_offset

            # Check if we have enough data
            if test_end > max_date:
                break

            # Get indices for this fold
            train_mask = (ts >= current_train_start) & (ts < train_end)
            test_mask = (ts >= test_start) & (ts < test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            # Skip if not enough training data
            if len(train_idx) >= self.min_train_samples and len(test_idx) > 0:
                folds.append((train_idx, test_idx))

            # Step forward
            current_train_start += step_offset

        return folds

    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        model_factory: Callable[[], Any],
        fit_kwargs: Optional[Dict] = None,
        verbose: bool = True,
    ) -> WFOResult:
        """
        Run walk-forward validation.

        Args:
            X: Input features, shape (n_samples, seq_len, n_features) or (n_samples, n_features)
            y: Labels, shape (n_samples,)
            timestamps: Datetime for each sample
            model_factory: Callable that returns a fresh model instance
            fit_kwargs: Additional kwargs for model.fit()
            verbose: Print progress

        Returns:
            WFOResult with per-fold and aggregate metrics
        """
        fit_kwargs = fit_kwargs or {}

        # Create folds
        folds = self.create_folds(timestamps, len(X))

        if len(folds) == 0:
            raise ValueError("Not enough data for WFO validation with current settings")

        if verbose:
            print(f"Walk-Forward Validation: {len(folds)} folds")
            print(f"  Train: {self.train_months} months, Test: {self.test_months} months")
            print()

        fold_results = []
        ts = pd.to_datetime(timestamps)

        for i, (train_idx, test_idx) in enumerate(folds):
            # Get fold data
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Get timestamps for reporting
            train_start = ts[train_idx].min()
            train_end = ts[train_idx].max()
            test_start = ts[test_idx].min()
            test_end = ts[test_idx].max()

            if verbose:
                print(f"Fold {i+1}/{len(folds)}: Train {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')}, "
                      f"Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")

            # Create and train fresh model
            model = model_factory()

            # Split some train data for validation (last 15%)
            val_split = int(len(X_train) * 0.85)
            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            y_tr, y_val = y_train[:val_split], y_train[val_split:]

            model.fit(
                X_tr, y_tr,
                validation_data=(X_val, y_val),
                verbose=False,
                **fit_kwargs
            )

            # Predict on test set
            predictions, confidences = model.predict(X_test)

            # Convert predictions to same scale as y_test (0,1,2)
            predictions_mapped = predictions + 1  # -1,0,1 -> 0,1,2

            # Calculate accuracy
            accuracy = np.mean(predictions_mapped == y_test)

            # Per-class accuracy
            class_accuracies = {}
            for cls, name in [(0, "DOWN"), (1, "FLAT"), (2, "UP")]:
                mask = y_test == cls
                if mask.sum() > 0:
                    class_accuracies[name] = float(np.mean(predictions_mapped[mask] == cls))

            if verbose:
                print(f"  Accuracy: {accuracy:.1%} (n={len(y_test):,})")

            fold_results.append(WFOFold(
                fold_id=i,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
                train_samples=len(train_idx),
                test_samples=len(test_idx),
                accuracy=accuracy,
                class_accuracies=class_accuracies,
                predictions=predictions_mapped,
                actuals=y_test,
                confidences=confidences,
            ))

        # Aggregate results
        accuracies = [f.accuracy for f in fold_results]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        # Walk-Forward Efficiency: consistency of performance across folds
        # WFE = mean / (mean + std) - higher is better, 1.0 is perfectly consistent
        wfe = mean_accuracy / (mean_accuracy + std_accuracy) if (mean_accuracy + std_accuracy) > 0 else 0

        # Aggregate class accuracies
        class_accs_agg = {}
        for name in ["DOWN", "FLAT", "UP"]:
            accs = [f.class_accuracies.get(name, 0) for f in fold_results if name in f.class_accuracies]
            if accs:
                class_accs_agg[name] = np.mean(accs)

        total_test_samples = sum(f.test_samples for f in fold_results)

        if verbose:
            print()
            print("=" * 50)
            print("Walk-Forward Validation Summary")
            print("=" * 50)
            print(f"  Folds: {len(fold_results)}")
            print(f"  Mean Accuracy: {mean_accuracy:.1%}")
            print(f"  Std Accuracy: {std_accuracy:.1%}")
            print(f"  WFE (Walk-Forward Efficiency): {wfe:.3f}")
            print(f"  Total test samples: {total_test_samples:,}")
            print()
            print("Per-class accuracy:")
            for name, acc in class_accs_agg.items():
                print(f"  {name}: {acc:.1%}")

        return WFOResult(
            folds=fold_results,
            mean_accuracy=mean_accuracy,
            std_accuracy=std_accuracy,
            wfe=wfe,
            class_accuracies=class_accs_agg,
            total_test_samples=total_test_samples,
        )

    def validate_with_timestamps_from_df(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        timestamp_col: str,
        model_factory: Callable[[], Any],
        **kwargs
    ) -> WFOResult:
        """
        Convenience method when timestamps are in a DataFrame.

        Args:
            df: DataFrame containing timestamp column (aligned with X, y)
            X: Features
            y: Labels
            timestamp_col: Name of timestamp column in df
            model_factory: Model factory function
            **kwargs: Passed to validate()

        Returns:
            WFOResult
        """
        timestamps = df[timestamp_col].values
        return self.validate(X, y, timestamps, model_factory, **kwargs)
