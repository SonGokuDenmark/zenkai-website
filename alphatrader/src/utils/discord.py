"""
Discord webhook notifications for training progress.
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, Dict, Any, List
import traceback


class DiscordNotifier:
    """
    Sends notifications to Discord via webhook.

    Features:
    - Training start/complete/error notifications
    - Progress updates at configurable intervals
    - Rich embeds with metrics and charts
    - Error reporting with stack traces
    """

    # Discord embed color codes
    COLORS = {
        "info": 0x3498DB,      # Blue
        "success": 0x2ECC71,   # Green
        "warning": 0xF1C40F,   # Yellow
        "error": 0xE74C3C,     # Red
        "progress": 0x9B59B6,  # Purple
    }

    def __init__(
        self,
        webhook_url: str,
        username: str = "AlphaTrader",
        avatar_url: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL (or empty to use DISCORD_WEBHOOK env var)
            username: Bot username to display
            avatar_url: Bot avatar URL (optional)
            enabled: Whether notifications are enabled
        """
        # Prefer env var over config for security
        self.webhook_url = os.environ.get('DISCORD_WEBHOOK', webhook_url)
        self.username = username
        self.avatar_url = avatar_url
        self.enabled = enabled

    def _send(self, payload: Dict[str, Any]) -> bool:
        """
        Send payload to Discord webhook.

        Args:
            payload: Webhook payload

        Returns:
            True if successful
        """
        if not self.enabled or not self.webhook_url:
            return False

        payload["username"] = self.username
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Discord notification failed: {e}")
            return False

    def send_message(
        self,
        content: str,
        embed: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a simple message.

        Args:
            content: Message content
            embed: Optional embed dict

        Returns:
            True if successful
        """
        payload = {"content": content}
        if embed:
            payload["embeds"] = [embed]
        return self._send(payload)

    def send_embed(
        self,
        title: str,
        description: str,
        color: str = "info",
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: Optional[str] = None,
        thumbnail_url: Optional[str] = None
    ) -> bool:
        """
        Send a rich embed message.

        Args:
            title: Embed title
            description: Embed description
            color: Color name (info, success, warning, error, progress)
            fields: List of field dicts with name, value, inline
            footer: Footer text
            thumbnail_url: Thumbnail image URL

        Returns:
            True if successful
        """
        embed = {
            "title": title,
            "description": description,
            "color": self.COLORS.get(color, self.COLORS["info"]),
            "timestamp": datetime.utcnow().isoformat()
        }

        if fields:
            embed["fields"] = fields

        if footer:
            embed["footer"] = {"text": footer}

        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}

        return self._send({"embeds": [embed]})

    def notify_training_start(
        self,
        run_name: str,
        config: Dict[str, Any],
        total_epochs: int,
        total_samples: int
    ) -> bool:
        """
        Notify that training has started.

        Args:
            run_name: Name of the training run
            config: Training configuration
            total_epochs: Total epochs to train
            total_samples: Total training samples

        Returns:
            True if successful
        """
        fields = [
            {"name": "Run Name", "value": run_name, "inline": True},
            {"name": "Epochs", "value": str(total_epochs), "inline": True},
            {"name": "Samples", "value": f"{total_samples:,}", "inline": True},
            {"name": "Model", "value": config.get("model", {}).get("type", "LSTM"), "inline": True},
            {"name": "Batch Size", "value": str(config.get("training", {}).get("batch_size", 64)), "inline": True},
            {"name": "Learning Rate", "value": str(config.get("training", {}).get("learning_rate", 0.001)), "inline": True},
        ]

        symbols = config.get("data", {}).get("symbols", [])
        if symbols:
            fields.append({
                "name": "Symbols",
                "value": ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""),
                "inline": False
            })

        return self.send_embed(
            title="Training Started",
            description="AlphaTrader neural network training has begun.",
            color="info",
            fields=fields,
            footer="Started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def notify_progress(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        accuracy: float,
        learning_rate: float,
        eta_seconds: Optional[int] = None,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Notify training progress.

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            loss: Current loss
            accuracy: Current accuracy
            learning_rate: Current learning rate
            eta_seconds: Estimated time remaining in seconds
            extra_metrics: Additional metrics to display

        Returns:
            True if successful
        """
        progress_pct = (epoch / total_epochs) * 100
        progress_bar = self._make_progress_bar(progress_pct)

        fields = [
            {"name": "Epoch", "value": f"{epoch}/{total_epochs}", "inline": True},
            {"name": "Progress", "value": f"{progress_pct:.1f}%", "inline": True},
            {"name": "Loss", "value": f"{loss:.4f}", "inline": True},
            {"name": "Accuracy", "value": f"{accuracy:.2f}%", "inline": True},
            {"name": "Learning Rate", "value": f"{learning_rate:.2e}", "inline": True},
        ]

        if eta_seconds:
            hours = eta_seconds // 3600
            minutes = (eta_seconds % 3600) // 60
            eta_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            fields.append({"name": "ETA", "value": eta_str, "inline": True})

        if extra_metrics:
            for name, value in extra_metrics.items():
                if isinstance(value, float):
                    fields.append({"name": name, "value": f"{value:.4f}", "inline": True})
                else:
                    fields.append({"name": name, "value": str(value), "inline": True})

        return self.send_embed(
            title="Training Progress",
            description=progress_bar,
            color="progress",
            fields=fields
        )

    def notify_training_complete(
        self,
        run_name: str,
        final_loss: float,
        final_accuracy: float,
        best_wfe: Optional[float] = None,
        training_time_seconds: Optional[int] = None,
        best_epoch: Optional[int] = None
    ) -> bool:
        """
        Notify that training has completed successfully.

        Args:
            run_name: Name of the training run
            final_loss: Final loss value
            final_accuracy: Final accuracy
            best_wfe: Best walk-forward efficiency achieved
            training_time_seconds: Total training time
            best_epoch: Epoch with best metrics

        Returns:
            True if successful
        """
        fields = [
            {"name": "Run Name", "value": run_name, "inline": True},
            {"name": "Final Loss", "value": f"{final_loss:.4f}", "inline": True},
            {"name": "Final Accuracy", "value": f"{final_accuracy:.2f}%", "inline": True},
        ]

        if best_wfe is not None:
            fields.append({"name": "Best WFE", "value": f"{best_wfe:.1f}%", "inline": True})

        if best_epoch is not None:
            fields.append({"name": "Best Epoch", "value": str(best_epoch), "inline": True})

        if training_time_seconds:
            hours = training_time_seconds // 3600
            minutes = (training_time_seconds % 3600) // 60
            time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            fields.append({"name": "Training Time", "value": time_str, "inline": True})

        return self.send_embed(
            title="Training Complete",
            description="Neural network training has finished successfully!",
            color="success",
            fields=fields,
            footer="Completed at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def notify_error(
        self,
        error: Exception,
        context: str = "",
        include_traceback: bool = True
    ) -> bool:
        """
        Notify about an error.

        Args:
            error: The exception that occurred
            context: Additional context about what was happening
            include_traceback: Whether to include full traceback

        Returns:
            True if successful
        """
        error_type = type(error).__name__
        error_msg = str(error)

        description = f"**{error_type}**: {error_msg}"
        if context:
            description = f"**Context**: {context}\n\n" + description

        fields = []

        if include_traceback:
            tb = traceback.format_exc()
            # Truncate if too long (Discord limit is 1024 per field)
            if len(tb) > 900:
                tb = tb[:900] + "\n... (truncated)"
            fields.append({
                "name": "Traceback",
                "value": f"```python\n{tb}\n```",
                "inline": False
            })

        return self.send_embed(
            title="Training Error",
            description=description,
            color="error",
            fields=fields if fields else None,
            footer="Error at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def notify_checkpoint_saved(
        self,
        epoch: int,
        step: int,
        loss: float,
        checkpoint_path: str
    ) -> bool:
        """
        Notify that a checkpoint was saved.

        Args:
            epoch: Current epoch
            step: Current step
            loss: Current loss
            checkpoint_path: Path to saved checkpoint

        Returns:
            True if successful
        """
        return self.send_embed(
            title="Checkpoint Saved",
            description=f"Model checkpoint saved at epoch {epoch}, step {step}",
            color="info",
            fields=[
                {"name": "Loss", "value": f"{loss:.4f}", "inline": True},
                {"name": "Path", "value": checkpoint_path, "inline": False}
            ]
        )

    def notify_validation(
        self,
        epoch: int,
        wfe: float,
        oos_pnl: float,
        win_rate: float,
        is_best: bool = False
    ) -> bool:
        """
        Notify validation results.

        Args:
            epoch: Current epoch
            wfe: Walk-forward efficiency
            oos_pnl: Out-of-sample PnL percentage
            win_rate: Win rate percentage
            is_best: Whether this is the best validation so far

        Returns:
            True if successful
        """
        title = "New Best Validation!" if is_best else "Validation Results"
        color = "success" if is_best else "info"

        return self.send_embed(
            title=title,
            description=f"Validation completed at epoch {epoch}",
            color=color,
            fields=[
                {"name": "WFE", "value": f"{wfe:.1f}%", "inline": True},
                {"name": "OOS PnL", "value": f"{oos_pnl:+.2f}%", "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            ]
        )

    def notify_gpu_warning(
        self,
        temp: float,
        usage: float,
        action: str = "throttled"
    ) -> bool:
        """
        Notify about GPU temperature or usage warning.

        Args:
            temp: Current GPU temperature
            usage: Current GPU usage percentage
            action: Action taken (throttled, paused, etc.)

        Returns:
            True if successful
        """
        return self.send_embed(
            title="GPU Warning",
            description=f"GPU {action} due to high temperature/usage",
            color="warning",
            fields=[
                {"name": "Temperature", "value": f"{temp:.1f}C", "inline": True},
                {"name": "Usage", "value": f"{usage:.1f}%", "inline": True},
                {"name": "Action", "value": action.title(), "inline": True},
            ]
        )

    def _make_progress_bar(self, percent: float, width: int = 20) -> str:
        """Create a text-based progress bar."""
        filled = int(width * percent / 100)
        empty = width - filled
        bar = "" * filled + "" * empty
        return f"`[{bar}]` {percent:.1f}%"
