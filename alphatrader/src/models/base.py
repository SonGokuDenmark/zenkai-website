"""
Base interface for all trading models.

All models (LSTM, Transformer, RL, etc.) implement this interface,
making them interchangeable plugins in the training pipeline.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for trading models.

    All models must implement:
    - fit(): Train on data
    - predict(): Generate predictions with confidence
    - save()/load(): Checkpoint management
    - name: Model identifier
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for logging and checkpoints."""
        pass

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X: Input features, shape (n_samples, seq_len, n_features) or (n_samples, n_features)
            y: Target labels, shape (n_samples,) with values in {-1, 0, 1} or one-hot
            validation_data: Optional (X_val, y_val) tuple for validation
            **kwargs: Model-specific training params (epochs, batch_size, etc.)

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence scores.

        Args:
            X: Input features, same shape as training data

        Returns:
            Tuple of:
            - predictions: np.ndarray of class labels (-1, 0, 1), shape (n_samples,)
            - confidences: np.ndarray of confidence scores [0, 1], shape (n_samples,)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get class probabilities.

        Args:
            X: Input features

        Returns:
            probabilities: np.ndarray of shape (n_samples, n_classes)
                          columns are [P(down), P(flat), P(up)]
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model state to disk.

        Args:
            path: File path (without extension, implementation adds appropriate suffix)
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: File path to saved model

        Returns:
            Loaded model instance
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for reproducibility.

        Returns:
            Dict of hyperparameters and settings
        """
        return {"name": self.name}

    def summary(self) -> str:
        """
        Get human-readable model summary.

        Returns:
            String description of model architecture
        """
        return f"{self.name} model"


class ModelRegistry:
    """
    Registry for model implementations.

    Usage:
        # Register a model
        @ModelRegistry.register("lstm")
        class LSTMClassifier(BaseModel):
            ...

        # Create model by name
        model = ModelRegistry.create("lstm", hidden_size=128)
    """

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_cls: type):
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance by name."""
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        return cls._models[name](**kwargs)

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())
