"""
Custom logging configuration for AlphaTrader.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "alphatrader",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (creates if not exists)
        log_format: Custom log format string
        console: Whether to log to console

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrainingLogger:
    """
    Specialized logger for training metrics.

    Logs to both file and provides structured output for parsing.
    """

    def __init__(
        self,
        name: str = "training",
        log_dir: str = "logs",
        run_name: Optional[str] = None
    ):
        """
        Initialize training logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            run_name: Name for this training run
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_name = run_name
        self.log_file = self.log_dir / f"{run_name}.log"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.csv"

        # Set up main logger
        self.logger = setup_logger(
            name=name,
            level="INFO",
            log_file=str(self.log_file),
            console=True
        )

        # Initialize metrics CSV
        self._init_metrics_csv()

    def _init_metrics_csv(self):
        """Initialize the metrics CSV file with headers."""
        headers = [
            "timestamp", "epoch", "step", "loss", "accuracy",
            "val_loss", "val_accuracy", "learning_rate", "wfe",
            "gpu_temp", "gpu_usage", "memory_usage"
        ]
        with open(self.metrics_file, "w") as f:
            f.write(",".join(headers) + "\n")

    def log_metrics(
        self,
        epoch: int,
        step: int,
        loss: float,
        accuracy: float,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        wfe: Optional[float] = None,
        gpu_temp: Optional[float] = None,
        gpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None
    ):
        """
        Log training metrics to CSV.

        Args:
            epoch: Current epoch
            step: Current step within epoch
            loss: Training loss
            accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
            wfe: Walk-forward efficiency
            gpu_temp: GPU temperature
            gpu_usage: GPU utilization
            memory_usage: Memory usage
        """
        timestamp = datetime.now().isoformat()

        values = [
            timestamp,
            str(epoch),
            str(step),
            f"{loss:.6f}",
            f"{accuracy:.4f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{val_accuracy:.4f}" if val_accuracy is not None else "",
            f"{learning_rate:.2e}" if learning_rate is not None else "",
            f"{wfe:.2f}" if wfe is not None else "",
            f"{gpu_temp:.1f}" if gpu_temp is not None else "",
            f"{gpu_usage:.1f}" if gpu_usage is not None else "",
            f"{memory_usage:.1f}" if memory_usage is not None else "",
        ]

        with open(self.metrics_file, "a") as f:
            f.write(",".join(values) + "\n")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")

    def epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        accuracy: float,
        time_seconds: float
    ):
        """Log epoch end with summary."""
        self.logger.info(
            f"Epoch {epoch}/{total_epochs} completed - "
            f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%, "
            f"Time: {time_seconds:.1f}s"
        )

    def validation_results(
        self,
        epoch: int,
        val_loss: float,
        val_accuracy: float,
        wfe: Optional[float] = None,
        is_best: bool = False
    ):
        """Log validation results."""
        msg = (
            f"Validation at epoch {epoch} - "
            f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%"
        )
        if wfe is not None:
            msg += f", WFE: {wfe:.1f}%"
        if is_best:
            msg += " [NEW BEST]"
        self.logger.info(msg)

    def checkpoint_saved(self, epoch: int, path: str):
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved at epoch {epoch}: {path}")

    def training_complete(
        self,
        total_epochs: int,
        best_loss: float,
        best_accuracy: float,
        total_time_seconds: float
    ):
        """Log training completion."""
        hours = total_time_seconds // 3600
        minutes = (total_time_seconds % 3600) // 60
        time_str = f"{int(hours)}h {int(minutes)}m"

        self.logger.info(
            f"Training complete after {total_epochs} epochs - "
            f"Best Loss: {best_loss:.4f}, Best Accuracy: {best_accuracy:.2f}%, "
            f"Total Time: {time_str}"
        )
