"""
Hidden Markov Model based market regime detection.

Classifies market states into four regimes:
- TRENDING_UP: Strong upward momentum, positive returns
- TRENDING_DOWN: Strong downward momentum, negative returns
- RANGING: Low volatility, mean-reverting behavior
- HIGH_VOL: High volatility, unstable conditions
"""

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import pickle
from pathlib import Path


class RegimeType(str, Enum):
    """Market regime types."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOL = "HIGH_VOL"
    UNKNOWN = "UNKNOWN"


class HMMRegimeDetector:
    """
    Gaussian HMM-based market regime detector.

    Detects four market regimes based on returns, volatility, and volume patterns.
    Uses unsupervised learning to discover hidden states, then maps them to
    meaningful regime labels based on their statistical characteristics.

    Attributes:
        n_states: Number of hidden states (default: 4)
        features: List of feature column names to use
        covariance_type: Type of covariance for HMM
        model: Trained GaussianHMM model
        state_mapping: Dict mapping HMM state indices to RegimeType
    """

    def __init__(
        self,
        n_states: int = 4,
        features: Optional[List[str]] = None,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize HMM regime detector.

        Args:
            n_states: Number of hidden states
            features: Feature columns to use (default: returns_1, volatility, volume_ratio)
            covariance_type: Type of covariance ('full', 'diag', 'tied', 'spherical')
            n_iter: Maximum EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.features = features or ["returns_1", "volatility", "volume_ratio"]
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.state_mapping: Dict[int, RegimeType] = {}
        self.feature_scaler_params: Dict[str, Tuple[float, float]] = {}
        self._is_fitted = False

    def _resolve_features(self, df: pd.DataFrame) -> List[str]:
        """
        Resolve config feature names to actual column names.

        Args:
            df: DataFrame with features

        Returns:
            List of resolved feature names
        """
        feature_mapping = {
            "returns": "returns_1",
            "volatility": "volatility",
            "volume_ratio": "volume_ratio",
            "adx": "adx",
            "atr_pct": "atr_pct",
        }

        resolved = []
        for f in self.features:
            actual = feature_mapping.get(f, f)
            if actual in df.columns:
                resolved.append(actual)
            elif f in df.columns:
                resolved.append(f)

        return resolved

    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Prepare feature matrix for HMM.

        Args:
            df: DataFrame with computed features
            fit_scaler: Whether to fit the scaler (True during training)

        Returns:
            Scaled feature matrix (n_samples, n_features)
        """
        # Resolve feature names
        resolved_features = self._resolve_features(df)

        if len(resolved_features) == 0:
            raise ValueError(
                f"No valid features found. Requested: {self.features}, "
                f"Available: {list(df.columns)}"
            )

        # Update features to resolved names
        if fit_scaler:
            self.features = resolved_features

        # Extract features
        X = df[self.features].values.copy().astype(np.float64)

        # Handle NaN - forward fill then backward fill
        X_df = pd.DataFrame(X)
        X = X_df.ffill().bfill().to_numpy().copy()  # Ensure writeable array

        # Replace inf with NaN, then fill
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0)

        # Standardize features
        if fit_scaler:
            for i, feat in enumerate(self.features):
                col = X[:, i]

                # Remove outliers for computing robust mean/std
                # Use median and MAD-based estimation for robustness
                median = np.nanmedian(col)
                mad = np.nanmedian(np.abs(col - median))
                robust_std = mad * 1.4826  # Scale MAD to std estimate

                # Fallback to regular mean/std if MAD is too small
                if robust_std < 1e-8:
                    mean = np.nanmean(col)
                    std = np.nanstd(col)
                    std = std if std > 1e-8 else 1.0
                else:
                    # Clip outliers (>5 MAD from median) before computing mean/std
                    lower = median - 5 * robust_std
                    upper = median + 5 * robust_std
                    clipped = np.clip(col, lower, upper)
                    mean = np.nanmean(clipped)
                    std = np.nanstd(clipped)
                    std = std if std > 1e-8 else 1.0

                # Validate scaler params are sensible
                if np.isnan(mean) or np.isnan(std) or np.isinf(mean) or np.isinf(std):
                    mean, std = 0.0, 1.0

                self.feature_scaler_params[feat] = (float(mean), float(std))
                X[:, i] = (X[:, i] - mean) / std
        else:
            for i, feat in enumerate(self.features):
                mean, std = self.feature_scaler_params.get(feat, (0.0, 1.0))
                X[:, i] = (X[:, i] - mean) / std

        # Replace any remaining NaN/inf after scaling
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme values (>10 std) to prevent HMM issues
        X = np.clip(X, -10, 10)

        return X

    def fit(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        verbose: bool = True
    ) -> "HMMRegimeDetector":
        """
        Fit HMM on historical data.

        Args:
            df: DataFrame with features (from FeatureEngineer)
            symbol: Optional symbol filter
            verbose: Print progress

        Returns:
            self
        """
        # Filter by symbol if provided
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol].copy()

        if verbose:
            print(f"Fitting HMM on {len(df):,} samples...")

        # Prepare features
        X = self._prepare_features(df, fit_scaler=True)

        if verbose:
            print(f"  Features: {self.features}")
            print(f"  Shape: {X.shape}")

        # Initialize and fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.model.fit(X)

        if verbose:
            print(f"  HMM converged: {self.model.monitor_.converged}")

        # Predict states for labeling
        states = self.model.predict(X)

        # Map states to regime labels based on characteristics
        self._map_states_to_regimes(df, states, verbose)

        self._is_fitted = True
        return self

    def fit_cross_symbol(
        self,
        df: pd.DataFrame,
        symbols: Optional[List[str]] = None,
        max_samples: int = 500000,
        verbose: bool = True
    ) -> "HMMRegimeDetector":
        """
        Fit HMM across multiple symbols.

        Training on cross-symbol data makes the model more robust
        to different market conditions.

        Args:
            df: DataFrame with all symbols
            symbols: List of symbols to include (None = all)
            max_samples: Maximum samples to use (for memory efficiency)
            verbose: Print progress

        Returns:
            self
        """
        if symbols:
            df = df[df["symbol"].isin(symbols)].copy()

        # Sample if too large
        if len(df) > max_samples:
            if verbose:
                print(f"Sampling {max_samples:,} from {len(df):,} rows")
            df = df.sample(n=max_samples, random_state=self.random_state)

        if verbose:
            print(f"Fitting cross-symbol HMM on {len(df):,} samples...")
            if "symbol" in df.columns:
                print(f"  Symbols: {df['symbol'].unique().tolist()}")

        # Prepare features
        X = self._prepare_features(df, fit_scaler=True)

        if verbose:
            print(f"  Features: {self.features}")
            print(f"  Shape: {X.shape}")

        # Fit HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        self.model.fit(X)

        if verbose:
            print(f"  HMM converged: {self.model.monitor_.converged}")

        # Map states using prediction
        states = self.model.predict(X)
        self._map_states_to_regimes(df, states, verbose)

        self._is_fitted = True
        return self

    def _map_states_to_regimes(
        self,
        df: pd.DataFrame,
        states: np.ndarray,
        verbose: bool = True
    ) -> None:
        """
        Map HMM state indices to meaningful regime labels.

        Uses post-hoc analysis of state characteristics:
        - Mean return for each state
        - Mean volatility for each state
        - Mean ADX (trend strength) for each state

        Args:
            df: Original DataFrame with features
            states: Predicted HMM states
            verbose: Print mapping info
        """
        # Create temporary df with states
        analysis_df = df.iloc[:len(states)].copy()
        analysis_df["_hmm_state"] = states

        # Compute state characteristics
        state_stats = {}

        for state in range(self.n_states):
            mask = analysis_df["_hmm_state"] == state
            state_data = analysis_df[mask]

            if len(state_data) == 0:
                continue

            # Mean return
            mean_return = 0.0
            for col in ["returns_1", "log_returns", "returns"]:
                if col in state_data.columns:
                    mean_return = state_data[col].mean()
                    break

            # Mean volatility
            mean_vol = 0.0
            for col in ["volatility", "atr_pct", "bb_width"]:
                if col in state_data.columns:
                    mean_vol = state_data[col].mean()
                    break

            # Mean ADX (trend strength)
            mean_adx = 25.0  # Default neutral
            if "adx" in state_data.columns:
                mean_adx = state_data["adx"].mean()

            state_stats[state] = {
                "mean_return": float(mean_return) if not np.isnan(mean_return) else 0.0,
                "mean_volatility": float(mean_vol) if not np.isnan(mean_vol) else 0.0,
                "mean_adx": float(mean_adx) if not np.isnan(mean_adx) else 25.0,
                "count": len(state_data),
            }

        # Assign regimes based on characteristics
        self.state_mapping = self._assign_regimes(state_stats)

        if verbose:
            print("\n  State -> Regime Mapping:")
            for state, stats in state_stats.items():
                regime = self.state_mapping.get(state, RegimeType.UNKNOWN)
                print(f"    State {state} -> {regime.value}")
                print(f"      count={stats['count']:,}, return={stats['mean_return']:.6f}, "
                      f"vol={stats['mean_volatility']:.6f}, adx={stats['mean_adx']:.1f}")

    def _assign_regimes(
        self,
        state_stats: Dict[int, Dict[str, float]]
    ) -> Dict[int, RegimeType]:
        """
        Assign regime labels to states based on their statistics.

        Logic:
        1. HIGH_VOL: Highest volatility state
        2. TRENDING_UP: Highest positive return with decent ADX
        3. TRENDING_DOWN: Lowest (most negative) return with decent ADX
        4. RANGING: Lowest ADX (least trending)

        Args:
            state_stats: Statistics for each state

        Returns:
            Mapping from state index to RegimeType
        """
        mapping = {}
        states = list(state_stats.keys())

        if len(states) == 0:
            return mapping

        # Sort states by various metrics
        by_vol = sorted(states, key=lambda s: state_stats[s]["mean_volatility"], reverse=True)
        by_return = sorted(states, key=lambda s: state_stats[s]["mean_return"], reverse=True)
        by_adx = sorted(states, key=lambda s: state_stats[s]["mean_adx"])

        assigned = set()

        # 1. Highest volatility -> HIGH_VOL
        high_vol_state = by_vol[0]
        mapping[high_vol_state] = RegimeType.HIGH_VOL
        assigned.add(high_vol_state)

        # 2. Highest positive return (not HIGH_VOL) -> TRENDING_UP
        for state in by_return:
            if state not in assigned:
                mapping[state] = RegimeType.TRENDING_UP
                assigned.add(state)
                break

        # 3. Most negative return (not assigned) -> TRENDING_DOWN
        for state in reversed(by_return):
            if state not in assigned:
                mapping[state] = RegimeType.TRENDING_DOWN
                assigned.add(state)
                break

        # 4. Remaining -> RANGING
        for state in states:
            if state not in assigned:
                mapping[state] = RegimeType.RANGING
                assigned.add(state)

        return mapping

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime for each row.

        Args:
            df: DataFrame with features

        Returns:
            Array of RegimeType string values
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df, fit_scaler=False)
        states = self.model.predict(X)

        # Map to regime labels
        regimes = np.array([
            self.state_mapping.get(s, RegimeType.UNKNOWN).value
            for s in states
        ])

        return regimes

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities.

        Args:
            df: DataFrame with features

        Returns:
            Array of shape (n_samples, n_states) with probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_features(df, fit_scaler=False)
        return self.model.predict_proba(X)

    def predict_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 100000
    ) -> np.ndarray:
        """
        Predict regimes in batches for large datasets.

        Args:
            df: DataFrame with features
            batch_size: Number of rows per batch

        Returns:
            Array of regime labels
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = len(df)
        regimes = np.empty(n_samples, dtype=object)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_df = df.iloc[start_idx:end_idx]
            batch_regimes = self.predict(batch_df)
            regimes[start_idx:end_idx] = batch_regimes

        return regimes

    def add_regime_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add regime column to DataFrame.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with 'regime' column added
        """
        df = df.copy()
        df["regime"] = self.predict(df)
        return df

    def save(self, path: str) -> None:
        """
        Save fitted model to disk.

        Args:
            path: File path for model
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        model_data = {
            "model": self.model,
            "state_mapping": {k: v.value for k, v in self.state_mapping.items()},
            "feature_scaler_params": self.feature_scaler_params,
            "n_states": self.n_states,
            "features": self.features,
            "covariance_type": self.covariance_type,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeDetector":
        """
        Load fitted model from disk.

        Args:
            path: File path to model

        Returns:
            Loaded HMMRegimeDetector
        """
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        detector = cls(
            n_states=model_data["n_states"],
            features=model_data["features"],
            covariance_type=model_data["covariance_type"],
        )

        detector.model = model_data["model"]
        detector.state_mapping = {
            k: RegimeType(v) for k, v in model_data["state_mapping"].items()
        }
        detector.feature_scaler_params = model_data["feature_scaler_params"]
        detector._is_fitted = True

        return detector

    def get_regime_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get statistics about detected regimes.

        Args:
            df: DataFrame with regime column (or will add it)

        Returns:
            Dictionary with regime statistics
        """
        if "regime" not in df.columns:
            df = self.add_regime_to_df(df)

        stats = {}
        total = len(df)

        for regime in RegimeType:
            if regime == RegimeType.UNKNOWN:
                continue

            mask = df["regime"] == regime.value
            count = mask.sum()

            stats[regime.value] = {
                "count": int(count),
                "percentage": float(count / total * 100) if total > 0 else 0.0,
            }

            if count > 0:
                for col in ["returns_1", "log_returns"]:
                    if col in df.columns:
                        stats[regime.value]["mean_return"] = float(df.loc[mask, col].mean())
                        break
                for col in ["volatility", "atr_pct"]:
                    if col in df.columns:
                        stats[regime.value]["mean_volatility"] = float(df.loc[mask, col].mean())
                        break

        return stats

    def print_regime_summary(self, df: pd.DataFrame) -> None:
        """
        Print a summary of regime distribution.

        Args:
            df: DataFrame with or without regime column
        """
        stats = self.get_regime_stats(df)

        print("\nRegime Distribution:")
        print("-" * 50)
        for regime, data in sorted(stats.items()):
            print(f"  {regime:15s}: {data['count']:>10,} ({data['percentage']:5.1f}%)")
            if "mean_return" in data:
                print(f"                   return={data['mean_return']:.6f}")
        print("-" * 50)
