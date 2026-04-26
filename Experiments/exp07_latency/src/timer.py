"""Precision timing utilities for latency benchmarking."""

import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class TimingResult:
    """Aggregated timing result for a benchmarked operation."""

    operation: str
    n_runs: int
    median_ns: int
    p5_ns: int
    p95_ns: int
    total_ns: int

    @property
    def median_ms(self) -> float:
        return self.median_ns / 1_000_000

    @property
    def median_us(self) -> float:
        return self.median_ns / 1_000

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "n_runs": self.n_runs,
            "median_ns": self.median_ns,
            "median_ms": self.median_ms,
            "median_us": self.median_us,
            "p5_ns": self.p5_ns,
            "p95_ns": self.p95_ns,
            "total_ns": self.total_ns,
        }


def time_operation(
    operation: str,
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    n_runs: int = 100,
    n_warmup: int = 10,
) -> TimingResult:
    """Time a function call with warmup and percentile reporting."""
    if kwargs is None:
        kwargs = {}

    for _ in range(n_warmup):
        func(*args, **kwargs)

    timings_ns = []
    for _ in range(n_runs):
        start = time.perf_counter_ns()
        func(*args, **kwargs)
        end = time.perf_counter_ns()
        timings_ns.append(end - start)

    arr = np.array(timings_ns)
    return TimingResult(
        operation=operation,
        n_runs=n_runs,
        median_ns=int(np.median(arr)),
        p5_ns=int(np.percentile(arr, 5)),
        p95_ns=int(np.percentile(arr, 95)),
        total_ns=int(np.sum(arr)),
    )


def system_info() -> dict:
    """Collect system information for reproducibility."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "python_version": sys.version,
        "os": f"{platform.system()} {platform.release()}",
        "ram_gb": round(_get_ram_gb(), 1),
        "platform": platform.platform(),
    }

    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["ollama_models"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["ollama_models"] = "unavailable"

    return info


def _get_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulong = ctypes.c_ulong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", c_ulong),
                        ("dwMemoryLoad", c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return stat.ullTotalPhys / (1024 ** 3)
            except Exception:
                pass
        return 0.0
