"""
LSTM Classifier for market direction prediction.

First implementation of BaseModel interface.
Goal: Prove the pipeline works, then iterate.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseModel, ModelRegistry


@ModelRegistry.register("lstm")
class LSTMClassifier(BaseModel):
    """
    LSTM-based classifier for predicting market direction.

    Predicts: DOWN (-1), FLAT (0), UP (1)
    Outputs: prediction + confidence score
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        device: str = None,
    ):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between layers
            num_classes: Output classes (3 = DOWN/FLAT/UP)
            learning_rate: Adam learning rate
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Training state
        self.optimizer = None
        self.criterion = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_loss = float("inf")
        self.epochs_trained = 0

    @property
    def name(self) -> str:
        return f"lstm_h{self.hidden_size}_l{self.num_layers}"

    def _build_model(self) -> nn.Module:
        """Build the LSTM network."""
        return _LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=self.num_classes,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 15,
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5,
    ) -> "LSTMClassifier":
        """
        Train the LSTM model.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            y: Labels, shape (n_samples,) with values in {0, 1, 2} (mapped from -1, 0, 1)
            validation_data: Optional (X_val, y_val) tuple
            epochs: Max training epochs
            batch_size: Training batch size
            early_stopping_patience: Stop after N epochs without improvement
            verbose: Print progress
            checkpoint_dir: Directory for periodic checkpoints
            checkpoint_interval: Save checkpoint every N epochs

        Returns:
            self
        """
        # Store numpy arrays directly - convert to tensors batch by batch to save memory
        # This avoids doubling memory by converting float16 -> float32 all at once
        self._X_train = X
        self._y_train = y
        train_loader = _NumpyDataLoader(X, y, batch_size=batch_size, shuffle=True)

        # Validation data - also stored as numpy, converted batch by batch
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            self._X_val = X_val
            self._y_val = y_val
            val_loader = _NumpyDataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        # Calculate class weights to handle imbalance (FLAT is underrepresented)
        class_counts = np.bincount(y, minlength=self.num_classes)
        # Inverse frequency weighting: rare classes get higher weight
        class_weights = 1.0 / (class_counts + 1e-6)  # avoid division by zero
        class_weights = class_weights / class_weights.sum() * self.num_classes  # normalize
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        if verbose:
            print(f"  Class weights: DOWN={class_weights[0]:.2f}, FLAT={class_weights[1]:.2f}, UP={class_weights[2]:.2f}")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        patience_counter = 0

        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        for epoch in pbar:
            # Train
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            val_acc = None
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

            self.epochs_trained += 1

            # Update progress bar
            desc = f"Loss: {train_loss:.4f}"
            if val_loss is not None:
                desc += f" | Val: {val_loss:.4f} | Acc: {val_acc:.1%}"
            pbar.set_postfix_str(desc)

            # Periodic checkpoint
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                self.save(f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}")

        return self

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run validation and return (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return total_loss / len(val_loader), correct / total

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with confidence scores.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            predictions: np.ndarray of class labels (-1, 0, 1), shape (n_samples,)
            confidences: np.ndarray of confidence scores [0, 1], shape (n_samples,)
        """
        proba = self.predict_proba(X)

        # Get predicted class (0, 1, 2) and convert to (-1, 0, 1)
        class_indices = np.argmax(proba, axis=1)
        predictions = class_indices - 1  # Map 0,1,2 -> -1,0,1

        # Confidence is the max probability
        confidences = np.max(proba, axis=1)

        return predictions, confidences

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Get class probabilities.

        Args:
            X: Input sequences
            batch_size: Process in batches to avoid OOM

        Returns:
            probabilities: np.ndarray of shape (n_samples, 3)
                          columns are [P(down), P(flat), P(up)]
        """
        self.model.eval()
        all_proba = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                all_proba.append(proba)

        return np.vstack(all_proba)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy on a dataset.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            y: True labels, shape (n_samples,) with values in {0, 1, 2}

        Returns:
            Accuracy score (0-1)
        """
        self.model.eval()
        loader = _NumpyDataLoader(X, y, batch_size=256, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return correct / total

    def save(self, path: str) -> None:
        """
        Save model state to disk.

        Saves both the PyTorch model weights and the classifier config.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "epochs_trained": self.epochs_trained,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, f"{path}.pt")

        # Save config
        config = self.get_config()
        with open(f"{path}.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LSTMClassifier":
        """
        Load model from disk.

        Args:
            path: File path (without extension)

        Returns:
            Loaded LSTMClassifier instance
        """
        path = Path(path)

        # Load config
        with open(f"{path}.json", "r") as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            num_classes=config["num_classes"],
            learning_rate=config["learning_rate"],
        )

        # Load PyTorch state
        checkpoint = torch.load(f"{path}.pt", map_location=instance.device)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.epochs_trained = checkpoint["epochs_trained"]
        instance.best_val_loss = checkpoint["best_val_loss"]
        instance.history = checkpoint["history"]

        if checkpoint["optimizer_state_dict"]:
            instance.optimizer = torch.optim.Adam(instance.model.parameters())
            instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for reproducibility."""
        return {
            "name": self.name,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "epochs_trained": self.epochs_trained,
        }

    def summary(self) -> str:
        """Get human-readable model summary."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return (
            f"{self.name}\n"
            f"  Input size: {self.input_size}\n"
            f"  Hidden size: {self.hidden_size}\n"
            f"  Layers: {self.num_layers}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Parameters: {total_params:,} ({trainable_params:,} trainable)\n"
            f"  Device: {self.device}\n"
            f"  Epochs trained: {self.epochs_trained}"
        )


class _NumpyDataLoader:
    """
    Memory-efficient data loader that keeps data as numpy float16.

    Converts to float32 tensors batch-by-batch. Slower but uses much less RAM.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        # Keep as numpy - don't pre-convert to save RAM
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]

            # Convert batch to tensors on-the-fly
            batch_X = torch.from_numpy(self.X[batch_indices].astype(np.float32))
            batch_y = torch.from_numpy(self.y[batch_indices].astype(np.int64))

            yield batch_X, batch_y


class _LSTMNetwork(nn.Module):
    """Internal PyTorch LSTM network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size)

        Returns:
            Logits, shape (batch, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Classify
        out = self.dropout(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out
