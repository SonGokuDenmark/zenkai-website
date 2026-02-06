"""
xLSTM Classifier for market direction prediction.

Uses the official xlstm package (NX-AI) with sLSTM and mLSTM blocks.
Implements the same interface as LSTMClassifier for easy benchmarking.

xLSTM paper: https://arxiv.org/abs/2405.04517
- sLSTM: scalar LSTM with exponential gating
- mLSTM: matrix LSTM with memory mixing
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        sLSTMBlockConfig,
        FeedForwardConfig,
    )
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False
    print("Warning: xlstm package not installed. Install with: pip install xlstm")

from .base import BaseModel, ModelRegistry


@ModelRegistry.register("xlstm")
class xLSTMClassifier(BaseModel):
    """
    xLSTM-based classifier for predicting market direction.

    Uses alternating sLSTM and mLSTM blocks for sequence modeling.
    sLSTM: Better for short-term patterns (exponential gating)
    mLSTM: Better for long-term dependencies (matrix memory)

    Predicts: DOWN (-1), FLAT (0), UP (1)
    Outputs: prediction + confidence score
    """

    def __init__(
        self,
        input_size: int,
        embedding_dim: int = 128,
        num_blocks: int = 2,
        context_length: int = 50,
        dropout: float = 0.3,
        num_classes: int = 3,
        learning_rate: float = 0.001,
        device: str = None,
        block_type: str = "mixed",  # "mixed", "slstm", "mlstm"
    ):
        """
        Args:
            input_size: Number of input features per timestep
            embedding_dim: Internal embedding dimension (like hidden_size in LSTM)
            num_blocks: Number of xLSTM blocks
            context_length: Sequence length (must match input sequences)
            dropout: Dropout rate
            num_classes: Output classes (3 = DOWN/FLAT/UP)
            learning_rate: Adam learning rate
            device: 'cuda' or 'cpu' (auto-detected if None)
            block_type: "mixed" (alternating sLSTM/mLSTM), "slstm", or "mlstm"
        """
        if not XLSTM_AVAILABLE:
            raise ImportError("xlstm package not installed. Install with: pip install xlstm")

        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.context_length = context_length
        self.dropout = dropout
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.block_type = block_type

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
        return f"xlstm_{self.block_type}_e{self.embedding_dim}_b{self.num_blocks}"

    def _build_model(self) -> nn.Module:
        """Build the xLSTM network."""
        return _xLSTMNetwork(
            input_size=self.input_size,
            embedding_dim=self.embedding_dim,
            num_blocks=self.num_blocks,
            context_length=self.context_length,
            dropout=self.dropout,
            num_classes=self.num_classes,
            block_type=self.block_type,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 15,
        learning_rate: float = None,
        patience: int = None,  # Alias for early_stopping_patience
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5,
    ) -> "xLSTMClassifier":
        """
        Train the xLSTM model.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)
            y: Labels, shape (n_samples,) with values in {0, 1, 2}
            validation_data: Optional (X_val, y_val) tuple
            epochs: Max training epochs
            batch_size: Training batch size
            early_stopping_patience: Stop after N epochs without improvement
            learning_rate: Override learning rate
            patience: Alias for early_stopping_patience
            verbose: Print progress
            checkpoint_dir: Directory for periodic checkpoints
            checkpoint_interval: Save checkpoint every N epochs

        Returns:
            self
        """
        if patience is not None:
            early_stopping_patience = patience

        if learning_rate is not None:
            self.learning_rate = learning_rate

        # Store numpy arrays - convert batch by batch to save memory
        train_loader = _NumpyDataLoader(X, y, batch_size=batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_loader = _NumpyDataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        # Calculate class weights for imbalance handling
        class_counts = np.bincount(y, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * self.num_classes
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)

        if verbose:
            print(f"  Class weights: DOWN={class_weights[0]:.2f}, FLAT={class_weights[1]:.2f}, UP={class_weights[2]:.2f}")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Training loop
        best_state = None
        epochs_without_improvement = 0

        progress_bar = tqdm(range(epochs), desc="Training", disable=not verbose)

        for epoch in progress_bar:
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.history["train_loss"].append(train_loss)

            # Validation phase
            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                correct = 0
                total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                        _, predicted = torch.max(outputs, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                val_loss = np.mean(val_losses)
                val_acc = correct / total
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                progress_bar.set_postfix({
                    "Loss": f"{train_loss:.4f}",
                    "Val": f"{val_loss:.4f}",
                    "Acc": f"{val_acc:.1%}",
                })

            self.epochs_trained += 1

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Checkpointing
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = Path(checkpoint_dir) / f"xlstm_epoch_{epoch + 1}"
                self.save(str(checkpoint_path))

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            predictions: Array of predictions (-1, 0, 1)
            confidences: Array of confidence scores (0-1)
        """
        self.model.eval()
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1) - 1  # Map 0,1,2 -> -1,0,1
        confidences = np.max(proba, axis=1)
        return predictions, confidences

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features)

        Returns:
            probabilities: Array of shape (n_samples, num_classes)
        """
        self.model.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = torch.softmax(outputs, dim=1)

        return proba.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model accuracy.

        Args:
            X: Input sequences
            y: True labels (0, 1, 2)

        Returns:
            Accuracy score
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
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch state
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
    def load(cls, path: str) -> "xLSTMClassifier":
        """Load model from disk."""
        path = Path(path)

        # Load config
        with open(f"{path}.json", "r") as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            input_size=config["input_size"],
            embedding_dim=config["embedding_dim"],
            num_blocks=config["num_blocks"],
            context_length=config["context_length"],
            dropout=config["dropout"],
            num_classes=config["num_classes"],
            learning_rate=config["learning_rate"],
            block_type=config["block_type"],
        )

        # Load PyTorch state
        checkpoint = torch.load(f"{path}.pt", map_location=instance.device)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.epochs_trained = checkpoint["epochs_trained"]
        instance.best_val_loss = checkpoint["best_val_loss"]
        instance.history = checkpoint["history"]

        if checkpoint["optimizer_state_dict"]:
            instance.optimizer = torch.optim.AdamW(instance.model.parameters())
            instance.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return instance

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "name": self.name,
            "input_size": self.input_size,
            "embedding_dim": self.embedding_dim,
            "num_blocks": self.num_blocks,
            "context_length": self.context_length,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate,
            "block_type": self.block_type,
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
            f"  Embedding dim: {self.embedding_dim}\n"
            f"  Blocks: {self.num_blocks} ({self.block_type})\n"
            f"  Context length: {self.context_length}\n"
            f"  Dropout: {self.dropout}\n"
            f"  Parameters: {total_params:,} ({trainable_params:,} trainable)\n"
            f"  Device: {self.device}\n"
            f"  Epochs trained: {self.epochs_trained}"
        )


class _NumpyDataLoader:
    """Memory-efficient data loader that keeps data as numpy."""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
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

            batch_X = torch.from_numpy(self.X[batch_indices].astype(np.float32))
            batch_y = torch.from_numpy(self.y[batch_indices].astype(np.int64))

            yield batch_X, batch_y


class _xLSTMNetwork(nn.Module):
    """Internal xLSTM network using the xlstm package."""

    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        num_blocks: int,
        context_length: int,
        dropout: float,
        num_classes: int,
        block_type: str = "mixed",
    ):
        super().__init__()

        self.input_size = input_size
        self.embedding_dim = embedding_dim

        # Input projection: input_size -> embedding_dim
        self.input_proj = nn.Linear(input_size, embedding_dim)

        # Build block config based on type
        if block_type == "slstm":
            # All sLSTM blocks
            slstm_config = sLSTMBlockConfig()
            mlstm_config = None
        elif block_type == "mlstm":
            # All mLSTM blocks
            slstm_config = None
            mlstm_config = mLSTMBlockConfig()
        else:
            # Mixed: alternating sLSTM and mLSTM
            slstm_config = sLSTMBlockConfig()
            mlstm_config = mLSTMBlockConfig()

        # xLSTM block stack config
        config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            add_post_blocks_norm=True,
        )

        # xLSTM backbone
        self.xlstm = xLSTMBlockStack(config)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layers: take last timestep and classify
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(embedding_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size)

        Returns:
            Logits, shape (batch, num_classes)
        """
        # Project input features to embedding dimension
        x = self.input_proj(x)  # (batch, seq, embedding_dim)

        # xLSTM forward
        x = self.xlstm(x)  # (batch, seq, embedding_dim)

        # Take last timestep
        x = x[:, -1, :]  # (batch, embedding_dim)

        # Classification head
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
