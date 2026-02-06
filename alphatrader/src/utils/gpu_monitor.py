"""
GPU monitoring and throttling for safe training.
"""

import time
import threading
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class GPUStats:
    """GPU statistics snapshot."""
    gpu_id: int
    name: str
    temperature: float  # Celsius
    usage: float  # Percentage
    memory_used: int  # MB
    memory_total: int  # MB
    memory_percent: float  # Percentage
    power_draw: Optional[float] = None  # Watts
    power_limit: Optional[float] = None  # Watts


class GPUMonitor:
    """
    Monitors GPU temperature and usage, with automatic throttling.

    Features:
    - Real-time GPU stats monitoring
    - Automatic throttling when temperature exceeds threshold
    - Configurable warning and critical thresholds
    - Callback support for notifications
    """

    def __init__(
        self,
        limit_percent: int = 75,
        max_temp: int = 80,
        throttle_temp: int = 75,
        check_interval: int = 30,
        on_throttle: Optional[Callable[[GPUStats], None]] = None,
        on_warning: Optional[Callable[[GPUStats], None]] = None
    ):
        """
        Initialize GPU monitor.

        Args:
            limit_percent: Target GPU utilization limit
            max_temp: Maximum temperature before pausing (Celsius)
            throttle_temp: Temperature to start throttling (Celsius)
            check_interval: Seconds between checks
            on_throttle: Callback when throttling occurs
            on_warning: Callback when warning threshold reached
        """
        self.limit_percent = limit_percent
        self.max_temp = max_temp
        self.throttle_temp = throttle_temp
        self.check_interval = check_interval
        self.on_throttle = on_throttle
        self.on_warning = on_warning

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._throttled = False
        self._paused = False

        # Try to import pynvml
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
            self._pynvml = pynvml
        except Exception as e:
            print(f"Warning: NVML not available, GPU monitoring disabled: {e}")

    def get_stats(self, gpu_id: int = 0) -> Optional[GPUStats]:
        """
        Get current GPU statistics.

        Args:
            gpu_id: GPU device ID

        Returns:
            GPUStats or None if not available
        """
        if not self._nvml_available:
            return None

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            name = self._pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            temp = self._pynvml.nvmlDeviceGetTemperature(
                handle, self._pynvml.NVML_TEMPERATURE_GPU
            )

            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            usage = util.gpu

            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used // (1024 * 1024)  # Convert to MB
            memory_total = mem_info.total // (1024 * 1024)
            memory_percent = (mem_info.used / mem_info.total) * 100

            # Try to get power info (not all GPUs support this)
            try:
                power_draw = self._pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
                power_limit = self._pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
            except Exception:
                power_draw = None
                power_limit = None

            return GPUStats(
                gpu_id=gpu_id,
                name=name,
                temperature=temp,
                usage=usage,
                memory_used=memory_used,
                memory_total=memory_total,
                memory_percent=memory_percent,
                power_draw=power_draw,
                power_limit=power_limit
            )

        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None

    def should_throttle(self, stats: GPUStats) -> bool:
        """Check if GPU should be throttled based on stats."""
        return stats.temperature >= self.throttle_temp

    def should_pause(self, stats: GPUStats) -> bool:
        """Check if training should be paused based on stats."""
        return stats.temperature >= self.max_temp

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            stats = self.get_stats()

            if stats:
                # Check for critical temperature
                if self.should_pause(stats):
                    if not self._paused:
                        self._paused = True
                        print(f"GPU CRITICAL: {stats.temperature}C - Training should pause!")
                        if self.on_warning:
                            self.on_warning(stats)

                # Check for throttle temperature
                elif self.should_throttle(stats):
                    if not self._throttled:
                        self._throttled = True
                        print(f"GPU THROTTLE: {stats.temperature}C - Reducing workload")
                        if self.on_throttle:
                            self.on_throttle(stats)

                # Reset flags if temperature is back to normal
                else:
                    if self._throttled or self._paused:
                        print(f"GPU NORMAL: {stats.temperature}C - Resuming normal operation")
                    self._throttled = False
                    self._paused = False

            time.sleep(self.check_interval)

    def start(self):
        """Start background monitoring."""
        if not self._nvml_available:
            print("GPU monitoring not available")
            return

        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("GPU monitoring started")

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        print("GPU monitoring stopped")

    def is_throttled(self) -> bool:
        """Check if GPU is currently throttled."""
        return self._throttled

    def is_paused(self) -> bool:
        """Check if training should be paused."""
        return self._paused

    def get_throttle_factor(self) -> float:
        """
        Get current throttle factor for adjusting batch size.

        Returns:
            1.0 if normal, 0.5-0.9 if throttled, 0.0 if paused
        """
        if self._paused:
            return 0.0
        if self._throttled:
            stats = self.get_stats()
            if stats:
                # Scale from 1.0 at throttle_temp to 0.5 at max_temp
                temp_range = self.max_temp - self.throttle_temp
                if temp_range > 0:
                    excess = stats.temperature - self.throttle_temp
                    factor = 1.0 - (excess / temp_range * 0.5)
                    return max(0.5, min(1.0, factor))
            return 0.75  # Default throttle factor
        return 1.0

    def print_stats(self, gpu_id: int = 0):
        """Print current GPU stats to console."""
        stats = self.get_stats(gpu_id)
        if stats:
            print(f"GPU: {stats.name}")
            print(f"  Temperature: {stats.temperature}C")
            print(f"  Usage: {stats.usage}%")
            print(f"  Memory: {stats.memory_used}/{stats.memory_total} MB ({stats.memory_percent:.1f}%)")
            if stats.power_draw:
                print(f"  Power: {stats.power_draw:.1f}W / {stats.power_limit:.1f}W")
        else:
            print("GPU stats not available")

    def set_power_limit(self, gpu_id: int = 0, limit_percent: int = 75):
        """
        Set GPU power limit (requires admin/root).

        Args:
            gpu_id: GPU device ID
            limit_percent: Percentage of max power limit
        """
        if not self._nvml_available:
            return

        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

            # Get min/max power limits
            min_limit = self._pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[0]
            max_limit = self._pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]

            # Calculate target limit
            target = min_limit + (max_limit - min_limit) * (limit_percent / 100)

            self._pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(target))
            print(f"GPU power limit set to {target / 1000:.0f}W ({limit_percent}%)")

        except Exception as e:
            print(f"Could not set power limit (may require admin): {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Cleanup on deletion."""
        if self._nvml_available:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
