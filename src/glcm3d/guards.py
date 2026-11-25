"""Cross-platform RAM and disk-space guards for slabbed 3D GLCM computation."""

from __future__ import annotations

import ctypes
from pathlib import Path
import re
import shutil
import sys
import time
import warnings

_WARNED_NO_MEM = False


class GLCMMemoryError(MemoryError):
    """Raised when available RAM is insufficient for a GLCM slab or output array."""


def get_available_mem_bytes() -> float:
    """Return available physical RAM in bytes (Linux, Windows, or Inf fallback)."""
    global _WARNED_NO_MEM

    if sys.platform.startswith("linux"):
        avail = float("inf")
        try:
            text = Path("/proc/meminfo").read_text(encoding="utf-8")
            m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
            if m:
                avail = float(int(m.group(1)) * 1024)
        except OSError:
            pass
        try:
            lim = Path("/sys/fs/cgroup/memory.max").read_text(encoding="utf-8").strip()
            if lim != "max":
                cur = int(
                    Path("/sys/fs/cgroup/memory.current")
                    .read_text(encoding="utf-8")
                    .strip()
                )
                avail = min(avail, float(int(lim) - cur))
        except (OSError, ValueError):
            pass
        if avail < float("inf"):
            return max(0.0, avail)

    elif sys.platform == "win32":
        try:
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return float(stat.ullAvailPhys)
        except Exception:
            pass

    if not _WARNED_NO_MEM:
        warnings.warn(
            "Could not determine available memory; memory guard is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_NO_MEM = True
    return float("inf")


def ensure_memory(
    needed_bytes: float,
    min_free_bytes: float,
    max_wait_sec: float,
    poll_interval_sec: float = 10.0,
) -> None:
    """Block until enough RAM is free, or raise :class:`GLCMMemoryError`."""
    t0 = time.monotonic()
    while True:
        avail = get_available_mem_bytes()
        if avail >= needed_bytes + min_free_bytes:
            return
        if time.monotonic() - t0 > max_wait_sec:
            raise GLCMMemoryError(
                f"Need ~{needed_bytes / 2**30:.1f} GB (+{min_free_bytes / 2**30:.1f} GB reserve) "
                f"but only {avail / 2**30:.1f} GB is available. "
                "Close other programs, lower quant_level/R, or request fewer features."
            )
        time.sleep(min(poll_interval_sec, max(0.0, max_wait_sec - (time.monotonic() - t0))))


def get_free_disk_bytes(path: str | Path) -> int:
    """Return free bytes on the filesystem containing ``path``."""
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free


def ensure_disk(
    path: str | Path,
    needed_bytes: float,
    reserve_bytes: float,
    max_wait_sec: float,
    poll_interval_sec: float = 60.0,
) -> bool:
    """Return True if ``path`` has enough free disk space, else wait up to ``max_wait_sec``."""
    t0 = time.monotonic()
    while True:
        free = get_free_disk_bytes(path)
        if free >= needed_bytes + reserve_bytes:
            return True
        if time.monotonic() - t0 >= max_wait_sec:
            return False
        time.sleep(min(poll_interval_sec, max(0.0, max_wait_sec - (time.monotonic() - t0))))


def estimate_bytes_per_voxel(nl: int, n_offsets: int, n_features: int) -> int:
    """Rough peak bytes per voxel for one slab inside ``compute_windowed_glcm_features``."""
    b = 2  # si (uint8) + mask (bool)
    b += 2 * n_offsets  # si_shift (uint8) + valid_d (bool)
    b += 28 * nl  # marginals (float32 counts + float64 probabilities)
    b += 8 * 22  # float64 scalars (total_safe, u, s, hx, hy, hxy2, savgh, ...)
    b += 4 * n_features  # slab texture output (float32)
    b += 8 * 17  # feature accumulators (float64)
    b += 100  # transient box-filter copies and temporaries
    return b
