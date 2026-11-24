"""Configuration and feature registry for 3D GLCM texture extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

ProgressCallback = Callable[[float, float, str], None]

ALL_FEATURES: tuple[str, ...] = (
    "autoc",
    "contr",
    "corrm",
    "corrp",
    "cprom",
    "cshad",
    "dissi",
    "energ",
    "entro",
    "homom",
    "homop",
    "maxpr",
    "sosvh",
    "savgh",
    "svarh",
    "svarc",
    "senth",
    "dvarh",
    "denth",
    "inf1h",
    "inf2h",
    "indnc",
    "idmnc",
)

MARGINAL_ONLY_FEATURES: frozenset[str] = frozenset(
    {"savgh", "senth", "svarh", "svarc", "denth", "dvarh"}
)


@dataclass(frozen=True)
class GLCMConfig:
    """Parameters and safety guard settings for 3D GLCM processing."""

    d: int = 1
    neighbor_sizes: tuple[int, ...] = (1, 3)
    quant_level: int = 8
    features: tuple[str, ...] = ALL_FEATURES

    # Stack slicing (None = use all Z pages in each TIFF)
    num_z_slices: int | None = None
    time_index: int = 1

    # Global foreground intensity percentile clipping
    low_percentile: float = 0.5
    high_percentile: float = 99.5

    # Memory guard settings
    mem_fraction: float = 0.60
    min_free_gb: float = 2.0
    max_wait_sec: float = 120.0

    # Disk guard settings
    min_free_disk_gb: float = 50.0
    disk_wait_sec: float = 1.0
    compress: bool = True

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")
        if self.quant_level < 2:
            raise ValueError(f"quant_level must be >= 2, got {self.quant_level}")
        if not self.neighbor_sizes:
            raise ValueError("neighbor_sizes must not be empty")
        if any(r < 0 for r in self.neighbor_sizes):
            raise ValueError(f"neighbor_sizes must be >= 0, got {self.neighbor_sizes}")
        if not self.features:
            raise ValueError("features must not be empty")
        unknown = set(self.features) - set(ALL_FEATURES)
        if unknown:
            raise ValueError(f"Unknown GLCM features: {sorted(unknown)}")
        if not (0.0 < self.mem_fraction <= 1.0):
            raise ValueError(f"mem_fraction must be in (0, 1], got {self.mem_fraction}")
        if self.time_index < 1:
            raise ValueError(f"time_index must be >= 1, got {self.time_index}")
        if not (0.0 <= self.low_percentile < self.high_percentile <= 100.0):
            raise ValueError(
                f"Invalid percentiles: ({self.low_percentile}, {self.high_percentile})"
            )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        overrides: dict | None = None,
    ) -> "GLCMConfig":
        """Load GLCM settings from the repo's ``config.json`` (under the ``glcm`` key)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        sec = raw.get("glcm", raw)

        neighbor = sec.get("NeighborSizeList", sec.get("NeighborSize", (1, 3)))
        if isinstance(neighbor, int):
            neighbor_sizes: tuple[int, ...] = (neighbor,)
        else:
            neighbor_sizes = tuple(int(r) for r in neighbor)

        feats: Sequence[str] = sec.get("glcm_properties", ALL_FEATURES)
        kwargs: dict = {
            "d": int(sec.get("D", 1)),
            "neighbor_sizes": neighbor_sizes,
            "quant_level": int(sec.get("quantLevel", 8)),
            "features": tuple(feats),
        }
        if "num_z_slices" in sec and sec["num_z_slices"]:
            kwargs["num_z_slices"] = int(sec["num_z_slices"])
        if overrides:
            kwargs.update(overrides)
        return cls(**kwargs)
