"""
config.py — thin Python wrapper around config.json.

All actual values live in config.json. Edit that file, not this one.
"""

import json
from pathlib import Path

_root = Path(__file__).parent
_cfg  = json.loads((_root / "config.json").read_text())

# ── Derived paths (built from data_root + experiment) ─────────────────────

DATA_ROOT  = Path(_cfg["data_root"])
EXPERIMENT = _cfg["experiment"]

_base = DATA_ROOT / EXPERIMENT
_sel  = _base / ""
_base_ep = DATA_ROOT / _cfg["experiment_endpoint"]

PATHS = {
    "masked_stacks":     _sel / " ",
    "masks":             _sel / "trackmatemasks",
    "masked_output":     _sel / " ",
    "indiv_imgs":        _sel / " ",
    "ctfire_out":        _sel / " ",
    "ctfire_results":    _sel / " ",
    "texture3d":         _sel / "output_bksub_texture3D" / "stitched",
    "ctfire_results_ep": _base_ep / " ",
    "texture2d_ep":      _base_ep / " ",
}

EXTERNAL = {k: Path(v) for k, v in _cfg["external"].items()}

PARAMS = _cfg["params"]
