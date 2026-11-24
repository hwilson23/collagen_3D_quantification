"""Sanity and regression tests for the ``glcm3d`` package."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile

from glcm3d import (
    ALL_FEATURES,
    MARGINAL_ONLY_FEATURES,
    GLCMConfig,
    box_filter_3d,
    compute_chunked,
    compute_offsets,
    compute_windowed_glcm_features,
    quantize_volume,
    run_pipeline,
    shift_volume,
)
from glcm3d.guards import get_available_mem_bytes, get_free_disk_bytes
from glcm3d.io import make_out_name, read_tiff_volume


def _safe_xlogx_scalar(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float64)
    pos = arr > 0.0
    out[pos] = arr[pos] * np.log(arr[pos])
    return out


def _brute_force_glcm_features(
    si: np.ndarray,
    mask: np.ndarray,
    nl: int,
    d: int,
    radius: int,
    features: tuple[str, ...] = ALL_FEATURES,
) -> np.ndarray:
    """Independent per-voxel brute-force reference for windowed 3D GLCM."""
    offsets = compute_offsets(d)
    r_max, c_max, z_max = si.shape
    out = np.zeros((*si.shape, len(features)), dtype=np.float32)

    ii, jj = np.meshgrid(np.arange(1, nl + 1), np.arange(1, nl + 1), indexing="ij")

    for r in range(r_max):
        for c in range(c_max):
            for z in range(z_max):
                if not mask[r, c, z]:
                    continue

                glcm = np.zeros((nl, nl), dtype=np.float64)
                for wr in range(max(0, r - radius), min(r_max, r + radius + 1)):
                    for wc in range(max(0, c - radius), min(c_max, c + radius + 1)):
                        for wz in range(max(0, z - radius), min(z_max, z + radius + 1)):
                            if not mask[wr, wc, wz]:
                                continue
                            gi = int(si[wr, wc, wz]) - 1
                            for dr, dc, dz in offsets:
                                nr, nc, nz = wr + int(dr), wc + int(dc), wz + int(dz)
                                if (
                                    0 <= nr < r_max
                                    and 0 <= nc < c_max
                                    and 0 <= nz < z_max
                                    and mask[nr, nc, nz]
                                ):
                                    gj = int(si[nr, nc, nz]) - 1
                                    glcm[gi, gj] += 1.0

                total = glcm.sum()
                if total == 0.0:
                    continue

                pij = glcm / total
                px = pij.sum(axis=1)
                py = pij.sum(axis=0)
                ks = np.arange(1, nl + 1, dtype=np.float64)
                ux = float(np.sum(ks * px))
                uy = float(np.sum(ks * py))
                sx = float(np.sqrt(max(0.0, np.sum(((ks - ux) ** 2) * px))))
                sy = float(np.sqrt(max(0.0, np.sum(((ks - uy) ** 2) * py))))

                hx = float(-np.sum(_safe_xlogx_scalar(px)))
                hy = float(-np.sum(_safe_xlogx_scalar(py)))
                pxpy = np.outer(px, py)
                hxy = float(-np.sum(_safe_xlogx_scalar(pij)))
                pos = (pij > 0.0) & (pxpy > 0.0)
                hxy1 = float(-np.sum(pij[pos] * np.log(pxpy[pos])))
                hxy2 = float(-np.sum(_safe_xlogx_scalar(pxpy)))

                p_plus = np.zeros(2 * nl - 1, dtype=np.float64)
                p_minus = np.zeros(nl, dtype=np.float64)
                for a in range(nl):
                    for b in range(nl):
                        p_plus[a + b] += pij[a, b]
                        p_minus[abs(a - b)] += pij[a, b]

                s_vals = np.arange(2, 2 * nl + 1, dtype=np.float64)
                d_vals = np.arange(0, nl, dtype=np.float64)
                savgh = float(np.sum(s_vals * p_plus))
                senth = float(-np.sum(_safe_xlogx_scalar(p_plus)))
                svarh = float(np.sum(((s_vals - senth) ** 2) * p_plus))
                svarc = float(np.sum(((s_vals - savgh) ** 2) * p_plus))
                denth = float(-np.sum(_safe_xlogx_scalar(p_minus)))
                dvarh = float(np.sum((d_vals**2) * p_minus))

                dij = ii - jj
                abs_dij = np.abs(dij)
                autoc = float(np.sum(pij * (ii * jj)))

                vals = {
                    "autoc": autoc,
                    "contr": float(np.sum(pij * (dij**2))),
                    "corrm": (
                        float(np.sum(pij * (ii - ux) * (jj - uy)) / (sx * sy))
                        if sx * sy > 0.0
                        else 0.0
                    ),
                    "corrp": (
                        float((autoc - ux * uy) / (sx * sy))
                        if sx * sy > 0.0
                        else 0.0
                    ),
                    "cprom": float(np.sum(pij * ((ii + jj - ux - uy) ** 4))),
                    "cshad": float(np.sum(pij * ((ii + jj - ux - uy) ** 3))),
                    "dissi": float(np.sum(pij * abs_dij)),
                    "energ": float(np.sum(pij**2)),
                    "entro": hxy,
                    "homom": float(np.sum(pij / (1.0 + abs_dij))),
                    "homop": float(np.sum(pij / (1.0 + dij**2))),
                    "maxpr": float(np.max(pij)),
                    "sosvh": float(np.sum(pij * ((jj - uy) ** 2))),
                    "savgh": savgh,
                    "svarh": svarh,
                    "svarc": svarc,
                    "senth": senth,
                    "dvarh": dvarh,
                    "denth": denth,
                    "inf1h": (
                        float((hxy - hxy1) / max(hx, hy))
                        if max(hx, hy) > 0.0
                        else 0.0
                    ),
                    "inf2h": float(np.sqrt(max(0.0, 1.0 - np.exp(-2.0 * (hxy2 - hxy))))),
                    "indnc": float(np.sum(pij / (1.0 + abs_dij / nl))),
                    "idmnc": float(np.sum(pij / (1.0 + (dij / nl) ** 2))),
                }
                for f_idx, name in enumerate(features):
                    v = vals[name]
                    out[r, c, z, f_idx] = np.float32(v) if np.isfinite(v) else np.float32(0.0)

    return out


class TestGLCM3DCore(unittest.TestCase):
    def test_compute_offsets_d1(self) -> None:
        offsets = compute_offsets(1)
        # 6 face + 12 edge neighbors (round(sqrt(2))==1, round(sqrt(3))==2)
        self.assertEqual(offsets.shape, (18, 3))
        self.assertFalse(np.any(np.all(offsets == 0, axis=1)))
        # Symmetric: every offset has its negation present
        off_set = {tuple(row) for row in offsets.tolist()}
        for row in off_set:
            self.assertIn((-row[0], -row[1], -row[2]), off_set)

    def test_quantize_volume(self) -> None:
        vol = np.array([[[0.0, 5.0, 10.0, 15.0, 20.0]]])
        si = quantize_volume(vol, nl=5, gl=(0.0, 20.0))
        self.assertEqual(si.dtype, np.uint8)
        np.testing.assert_array_equal(si, np.array([[[1, 2, 3, 4, 5]]], dtype=np.uint8))

        # Degenerate GL(1) == GL(2) -> all ones
        si_const = quantize_volume(vol, nl=8, gl=(5.0, 5.0))
        np.testing.assert_array_equal(si_const, np.ones_like(vol, dtype=np.uint8))

        # Clamping outside range
        si_clamp = quantize_volume(np.array([[[-5.0, 100.0]]]), nl=8, gl=(0.0, 10.0))
        np.testing.assert_array_equal(si_clamp, np.array([[[1, 8]]], dtype=np.uint8))

    def test_box_filter_3d_matches_naive(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 5, size=(7, 6, 5)).astype(np.float32)
        for radius in (0, 1, 2):
            got = box_filter_3d(arr, radius)
            expected = np.zeros_like(arr, dtype=np.float64)
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    for z in range(arr.shape[2]):
                        expected[r, c, z] = arr[
                            max(0, r - radius) : min(arr.shape[0], r + radius + 1),
                            max(0, c - radius) : min(arr.shape[1], c + radius + 1),
                            max(0, z - radius) : min(arr.shape[2], z + radius + 1),
                        ].sum()
            np.testing.assert_allclose(got, expected.astype(np.float32), atol=0.0, rtol=0.0)

    def test_shift_volume(self) -> None:
        vol = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        shifted = shift_volume(vol, (1, 0, -1), fill_val=0)
        self.assertEqual(shifted[0, 1, 1], vol[1, 1, 0])
        self.assertEqual(shifted[2, 1, 1], 0)
        self.assertEqual(shifted[0, 1, 0], 0)

    def test_matches_brute_force_all_23_features(self) -> None:
        rng = np.random.default_rng(123)
        si = rng.integers(1, 5, size=(7, 8, 6), dtype=np.uint8)
        mask = rng.random(size=si.shape) > 0.25

        for radius in (1, 2):
            got = compute_windowed_glcm_features(si, mask, nl=4, d=1, radius=radius)
            ref = _brute_force_glcm_features(si, mask, nl=4, d=1, radius=radius)
            self.assertEqual(got.shape, (*si.shape, len(ALL_FEATURES)))
            for f_idx, name in enumerate(ALL_FEATURES):
                np.testing.assert_allclose(
                    got[..., f_idx],
                    ref[..., f_idx],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Mismatch for feature '{name}' at R={radius}",
                )

    def test_constant_volume_no_eps_artifacts(self) -> None:
        # Every voxel has gray level 3 -> single non-zero GLCM entry P(3,3) = 1
        si = np.full((6, 6, 6), 3, dtype=np.uint8)
        mask = np.ones((6, 6, 6), dtype=bool)
        tex = compute_windowed_glcm_features(si, mask, nl=8, d=1, radius=1)
        idx = {name: i for i, name in enumerate(ALL_FEATURES)}

        self.assertTrue(np.all(np.isfinite(tex)))
        np.testing.assert_allclose(tex[..., idx["energ"]], 1.0, atol=1e-6)
        np.testing.assert_allclose(tex[..., idx["maxpr"]], 1.0, atol=1e-6)
        np.testing.assert_allclose(tex[..., idx["homom"]], 1.0, atol=1e-6)
        np.testing.assert_allclose(tex[..., idx["homop"]], 1.0, atol=1e-6)
        np.testing.assert_allclose(tex[..., idx["autoc"]], 9.0, atol=1e-6)
        np.testing.assert_allclose(tex[..., idx["savgh"]], 6.0, atol=1e-6)
        # Exact zero entropy and variance (no -eps artifacts, no inf1h blowup)
        for zero_feat in ("contr", "dissi", "entro", "senth", "denth", "dvarh",
                          "sosvh", "svarc", "corrm", "corrp", "inf1h", "inf2h",
                          "cprom", "cshad"):
            np.testing.assert_allclose(
                tex[..., idx[zero_feat]],
                0.0,
                atol=1e-6,
                err_msg=f"Expected 0 for '{zero_feat}' on constant volume",
            )

    def test_isolated_single_voxel_returns_zeros(self) -> None:
        si = np.ones((5, 5, 5), dtype=np.uint8)
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[2, 2, 2] = True  # no valid neighbor pairs at D=1
        tex = compute_windowed_glcm_features(si, mask, nl=4, d=1, radius=1)
        self.assertTrue(np.all(np.isfinite(tex)))
        np.testing.assert_array_equal(tex, np.zeros_like(tex))

    def test_marginal_only_fast_path_matches_full(self) -> None:
        rng = np.random.default_rng(7)
        si = rng.integers(1, 9, size=(8, 7, 6), dtype=np.uint8)
        mask = rng.random(size=si.shape) > 0.2
        marg_feats = tuple(sorted(MARGINAL_ONLY_FEATURES))

        tex_marg = compute_windowed_glcm_features(si, mask, nl=8, d=1, radius=1, features=marg_feats)
        tex_full = compute_windowed_glcm_features(si, mask, nl=8, d=1, radius=1, features=ALL_FEATURES)
        full_idx = {name: i for i, name in enumerate(ALL_FEATURES)}

        for i, name in enumerate(marg_feats):
            np.testing.assert_array_equal(tex_marg[..., i], tex_full[..., full_idx[name]])


class TestGLCM3DChunkedAndPipeline(unittest.TestCase):
    def test_chunked_matches_whole_volume_exactly(self) -> None:
        rng = np.random.default_rng(99)
        si = rng.integers(1, 9, size=(8, 15, 7), dtype=np.uint8)
        mask = rng.random(size=si.shape) > 0.25
        # Leave an empty middle band along the longest axis to test empty-slab skipping
        mask[:, 6:9, :] = False

        whole = compute_windowed_glcm_features(si, mask, nl=8, d=1, radius=2)
        for core_size in (2, 3, 5):
            chunked = compute_chunked(
                si,
                mask,
                nl=8,
                d=1,
                radius=2,
                max_slab_override=core_size,
            )
            np.testing.assert_allclose(
                chunked,
                whole,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Chunked mismatch for core={core_size}",
            )

    def test_guards_return_positive_values(self) -> None:
        avail_mem = get_available_mem_bytes()
        self.assertGreater(avail_mem, 0)
        free_disk = get_free_disk_bytes(Path.cwd())
        self.assertGreater(free_disk, 0)

    def test_pipeline_end_to_end_and_resume(self) -> None:
        rng = np.random.default_rng(2026)
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            # Write a synthetic uint16 3D stack in (Z, Y, X) order
            vol_zyx = rng.integers(0, 500, size=(8, 12, 10), dtype=np.uint16)
            vol_zyx[:, :2, :] = 0  # background border
            tifffile.imwrite(str(in_dir / "sample1.tif"), vol_zyx)
            # Stale partial file should be cleaned up
            stale = out_dir / "stale_output.tif.part"
            stale.write_bytes(b"corrupt")

            cfg = GLCMConfig(
                d=1,
                neighbor_sizes=(1,),
                quant_level=4,
                features=("autoc", "contr", "entro", "sosvh"),
                min_free_disk_gb=0.0,
                min_free_gb=0.0,
            )
            list_file = Path(tmp) / "filelist.txt"

            res1 = run_pipeline(
                in_dir, out_dir, list_file=list_file, config=cfg, verbose=False
            )
            self.assertFalse(stale.exists())
            self.assertFalse(res1["stopped_early"])
            self.assertEqual(len(res1["processed"]), 4)
            self.assertEqual(res1["skipped"], [])

            # Verify output shape and dtype
            out_name = make_out_name("sample1", "contr", 1, 1, 4)
            out_vol = read_tiff_volume(out_dir / out_name)
            self.assertEqual(out_vol.dtype, np.float32)
            self.assertEqual(out_vol.shape, (12, 10, 8))  # (Y, X, Z)
            # Background border must remain 0
            np.testing.assert_array_equal(out_vol[:2, :, :], 0.0)

            # Second run should skip everything via filelist.txt
            res2 = run_pipeline(
                in_dir, out_dir, list_file=list_file, config=cfg, verbose=False
            )
            self.assertEqual(res2["processed"], [])
            self.assertEqual(res2["skipped"], ["sample1"])

    def test_ground_truth_spiral_smoke(self) -> None:
        gt_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ground_truth_data"
            / "spiral_cone_3dgauss.tif"
        )
        if not gt_path.is_file():
            self.skipTest("Ground truth spiral_cone_3dgauss.tif not present")

        vol = read_tiff_volume(gt_path)
        nz = np.argwhere(vol > 0)
        self.assertGreater(len(nz), 0, "Ground truth volume should have foreground voxels")
        # Center a small 24x24x16 subvolume on a foreground voxel
        cy, cx, cz = (int(v) for v in nz[len(nz) // 2])
        y0 = max(0, min(cy - 12, vol.shape[0] - 24))
        x0 = max(0, min(cx - 12, vol.shape[1] - 24))
        z0 = max(0, min(cz - 8, vol.shape[2] - 16))
        sub = vol[y0 : y0 + 24, x0 : x0 + 24, z0 : z0 + 16]
        mask = sub > 0
        self.assertTrue(np.any(mask), "Subvolume should contain foreground voxels")

        fg = sub[mask]
        lo, hi = np.percentile(fg, [0.5, 99.5])
        si = quantize_volume(sub, nl=8, gl=(float(lo), float(hi)))
        tex = compute_chunked(si, mask, nl=8, d=1, radius=1, max_slab_override=8)

        self.assertEqual(tex.shape, (*sub.shape, len(ALL_FEATURES)))
        self.assertTrue(np.all(np.isfinite(tex)))
        # Outside mask must be strictly 0
        np.testing.assert_array_equal(tex[~mask], 0.0)


if __name__ == "__main__":
    unittest.main()
