"""Automated tests for packaging, CLI (`python -m glcm3d`), single-file API, and Tkinter GUI."""

from __future__ import annotations

import gc
import importlib
import importlib.metadata as md
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
import tomllib
from types import SimpleNamespace
from typing import Sequence
import unittest
from unittest import mock
import weakref

import numpy as np
import tifffile

from glcm3d import (
    GLCMConfig,
    compute_chunked,
    compute_windowed_glcm_features,
    gui as glcm_gui,
    run_pipeline,
    run_single_file,
)
from glcm3d.io import make_out_name, read_tiff_volume


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPackagingAndCLI(unittest.TestCase):
    def test_pyproject_metadata_and_entry_points(self) -> None:
        pyproject_path = REPO_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data["project"]
        self.assertEqual(project["name"], "glcm3D")
        self.assertIn("Repository", project)
        self.assertEqual(project["scripts"]["glcm3d"], "glcm3d.cli:main")
        self.assertEqual(project["gui-scripts"]["glcm3d-gui"], "glcm3d.gui:main")
        self.assertEqual(
            data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"],
            ["src/glcm3d"],
        )

        dist = md.distribution("glcm3D")
        self.assertEqual(dist.metadata["Name"], "glcm3D")
        ep_map = {(ep.group, ep.name): ep.value for ep in dist.entry_points}
        self.assertEqual(ep_map[("console_scripts", "glcm3d")], "glcm3d.cli:main")
        self.assertEqual(ep_map[("gui_scripts", "glcm3d-gui")], "glcm3d.gui:main")

        scripts_dir = Path(sys.executable).parent
        ext = ".exe" if sys.platform == "win32" else ""
        self.assertTrue((scripts_dir / f"glcm3d{ext}").is_file())
        self.assertTrue((scripts_dir / f"glcm3d-gui{ext}").is_file())

    def test_python_m_glcm3d_help_matches_script_entrypoint(self) -> None:
        mod_res = subprocess.run(
            [sys.executable, "-m", "glcm3d", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(mod_res.returncode, 0, msg=mod_res.stderr)
        self.assertIn("usage: glcm3d", mod_res.stdout)

        ext = ".exe" if sys.platform == "win32" else ""
        exe_path = Path(sys.executable).parent / f"glcm3d{ext}"
        if not exe_path.is_file():
            found = shutil.which("glcm3d")
            self.assertIsNotNone(found, "glcm3d script executable not found")
            exe_path = Path(found)

        exe_res = subprocess.run(
            [str(exe_path), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(exe_res.returncode, 0, msg=exe_res.stderr)
        self.assertEqual(mod_res.stdout, exe_res.stdout)

    def test_python_m_glcm3d_gui_help(self) -> None:
        res = subprocess.run(
            [sys.executable, "-m", "glcm3d.gui", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("usage: glcm3d-gui", res.stdout)

    def test_python_m_glcm3d_runs_synthetic_stack(self) -> None:
        rng = np.random.default_rng(101)
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            vol = rng.integers(10, 400, size=(6, 8, 8), dtype=np.uint16)
            vol[:, :1, :] = 0
            tifffile.imwrite(str(in_dir / "stackA.tif"), vol)

            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "glcm3d",
                    "-i",
                    str(in_dir),
                    "-o",
                    str(out_dir),
                    "--radii",
                    "1",
                    "--quant-level",
                    "4",
                    "--features",
                    "contr",
                    "--min-free-disk-gb",
                    "0",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)
            out_file = out_dir / make_out_name("stackA", "contr", 1, 1, 4)
            self.assertTrue(out_file.is_file())
            out_vol = read_tiff_volume(out_file)
            self.assertEqual(out_vol.dtype, np.float32)
            self.assertEqual(out_vol.shape, (8, 8, 6))


class TestSingleFileAndProgress(unittest.TestCase):
    def test_core_and_chunked_progress_callbacks(self) -> None:
        rng = np.random.default_rng(7)
        si = rng.integers(1, 5, size=(6, 12, 6), dtype=np.uint8)
        mask = np.ones_like(si, dtype=bool)
        mask[:, 4:8, :] = False  # empty middle slab

        core_updates: list[tuple[float, float, str]] = []
        compute_windowed_glcm_features(
            si,
            mask,
            nl=4,
            d=1,
            radius=1,
            features=("contr", "energ"),
            progress_callback=lambda c, t, m: core_updates.append((c, t, m)),
        )
        self.assertEqual(len(core_updates), 8)  # 2 * nl
        self.assertEqual(core_updates[-1][:2], (8.0, 8.0))
        self.assertTrue(
            all(core_updates[i][0] <= core_updates[i + 1][0] for i in range(len(core_updates) - 1))
        )

        marg_updates: list[tuple[float, float, str]] = []
        compute_windowed_glcm_features(
            si,
            mask,
            nl=4,
            d=1,
            radius=1,
            features=("savgh",),
            progress_callback=lambda c, t, m: marg_updates.append((c, t, m)),
        )
        self.assertEqual(len(marg_updates), 4)
        self.assertEqual(marg_updates[-1][:2], (4.0, 4.0))

        chunk_updates: list[tuple[float, float, str]] = []
        compute_chunked(
            si,
            mask,
            nl=4,
            d=1,
            radius=1,
            features=("contr",),
            max_slab_override=4,
            progress_callback=lambda c, t, m: chunk_updates.append((c, t, m)),
        )
        self.assertGreater(len(chunk_updates), 3)
        self.assertTrue(any("empty, skipped" in msg for _, _, msg in chunk_updates))
        self.assertEqual(chunk_updates[-1][0], chunk_updates[-1][1])
        self.assertTrue(
            all(0.0 <= c <= t for c, t, _ in chunk_updates)
        )
        self.assertTrue(
            all(
                chunk_updates[i][0] <= chunk_updates[i + 1][0]
                for i in range(len(chunk_updates) - 1)
            )
        )

    def test_run_single_file_3d_2d_overwrite_empty_and_validation(self) -> None:
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            out_dir.mkdir()

            # Distractor .tif in same folder with different intensity should be ignored
            distractor = np.full((5, 6, 6), 60000, dtype=np.uint16)
            tifffile.imwrite(str(in_dir / "other.tif"), distractor)

            vol_zyx = rng.integers(10, 200, size=(6, 10, 8), dtype=np.uint16)
            vol_zyx[:, :2, :] = 0
            target_tif = in_dir / "sample_one.tif"
            tifffile.imwrite(str(target_tif), vol_zyx)

            stale = out_dir / "leftover.tif.part"
            stale.write_bytes(b"stale")

            cfg = GLCMConfig(
                d=1,
                neighbor_sizes=(1,),
                quant_level=4,
                features=("contr", "energ"),
                min_free_disk_gb=0.0,
                min_free_gb=0.0,
            )

            updates: list[tuple[float, float, str]] = []
            summary = run_single_file(
                target_tif,
                out_dir,
                config=cfg,
                progress_callback=lambda c, t, m: updates.append((c, t, m)),
                verbose=False,
            )

            self.assertFalse(stale.exists())
            self.assertFalse((out_dir.parent / "filelist.txt").exists())
            self.assertFalse(summary["stopped_early"])
            self.assertEqual(summary["skipped"], [])
            self.assertEqual(len(summary["processed"]), 2)
            # Range must come only from sample_one.tif (< 200), not distractor (60000)
            self.assertLessEqual(summary["intensity_range"][1], 200.0)

            self.assertGreater(len(updates), 2)
            self.assertTrue(all(0.0 <= c <= t for c, t, _ in updates))
            self.assertTrue(
                all(updates[i][0] <= updates[i + 1][0] for i in range(len(updates) - 1))
            )
            self.assertEqual(updates[-1][0], updates[-1][1])

            out_contr = out_dir / make_out_name("sample_one", "contr", 1, 1, 4)
            out_vol = read_tiff_volume(out_contr)
            self.assertEqual(out_vol.dtype, np.float32)
            self.assertEqual(out_vol.shape, (10, 8, 6))
            np.testing.assert_array_equal(out_vol[:2, :, :], 0.0)

            # overwrite=False skips existing outputs
            skip_summary = run_single_file(
                target_tif,
                out_dir,
                config=cfg,
                overwrite=False,
                verbose=False,
            )
            self.assertEqual(skip_summary["processed"], [])
            self.assertEqual(skip_summary["skipped"], ["sample_one"])

            # overwrite=True regenerates outputs
            redo_summary = run_single_file(
                target_tif,
                out_dir,
                config=cfg,
                overwrite=True,
                verbose=False,
            )
            self.assertEqual(len(redo_summary["processed"]), 2)

            # 2D .tiff support
            img_2d = rng.integers(5, 100, size=(10, 12), dtype=np.uint16)
            tif_2d = in_dir / "slice2d.tiff"
            tifffile.imwrite(str(tif_2d), img_2d)
            out_2d_dir = Path(tmp) / "out_2d"
            sum_2d = run_single_file(tif_2d, out_2d_dir, config=cfg, verbose=False)
            self.assertEqual(len(sum_2d["processed"]), 2)
            vol_2d_out = read_tiff_volume(
                out_2d_dir / make_out_name("slice2d", "contr", 1, 1, 4)
            )
            self.assertEqual(vol_2d_out.shape, (10, 12, 1))

            # Empty foreground handling
            empty_tif = in_dir / "empty.tif"
            tifffile.imwrite(str(empty_tif), np.zeros((5, 6, 6), dtype=np.uint16))
            out_empty = Path(tmp) / "out_empty"
            empty_updates: list[tuple[float, float, str]] = []
            empty_sum = run_single_file(
                empty_tif,
                out_empty,
                config=cfg,
                progress_callback=lambda c, t, m: empty_updates.append((c, t, m)),
                verbose=False,
            )
            self.assertEqual(empty_sum["processed"], [])
            self.assertEqual(empty_sum["skipped"], ["empty"])
            self.assertFalse(empty_sum["stopped_early"])
            self.assertEqual(empty_updates[-1][0], empty_updates[-1][1])

            # Validation errors
            with self.assertRaises(FileNotFoundError):
                run_single_file(in_dir / "missing.tif", out_dir, config=cfg, verbose=False)
            with self.assertRaises(ValueError):
                run_single_file(in_dir, out_dir, config=cfg, verbose=False)
            bad_ext = in_dir / "notes.txt"
            bad_ext.write_text("not a tiff", encoding="utf-8")
            with self.assertRaises(ValueError):
                run_single_file(bad_ext, out_dir, config=cfg, verbose=False)
            with self.assertRaises(NotADirectoryError):
                run_single_file(target_tif, bad_ext, config=cfg, verbose=False)

    def test_run_single_file_disk_guard_stopped_early(self) -> None:
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as tmp:
            tif_path = Path(tmp) / "sample.tif"
            out_dir = Path(tmp) / "out"
            tifffile.imwrite(
                str(tif_path), rng.integers(1, 50, size=(5, 6, 6), dtype=np.uint16)
            )
            cfg = GLCMConfig(
                d=1,
                neighbor_sizes=(1,),
                quant_level=4,
                features=("contr",),
                min_free_disk_gb=1e9,  # impossible reserve
                disk_wait_sec=0.0,
                min_free_gb=0.0,
            )
            summary = run_single_file(tif_path, out_dir, config=cfg, verbose=False)
            self.assertTrue(summary["stopped_early"])
            self.assertEqual(summary["processed"], [])

    def test_run_pipeline_progress_callback(self) -> None:
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            out_dir = Path(tmp) / "out"
            in_dir.mkdir()
            tifffile.imwrite(
                str(in_dir / "s1.tif"),
                rng.integers(5, 100, size=(5, 8, 8), dtype=np.uint16),
            )
            cfg = GLCMConfig(
                d=1,
                neighbor_sizes=(1,),
                quant_level=4,
                features=("contr",),
                min_free_disk_gb=0.0,
                min_free_gb=0.0,
            )
            updates: list[tuple[float, float, str]] = []
            res = run_pipeline(
                in_dir,
                out_dir,
                list_file=Path(tmp) / "filelist.txt",
                config=cfg,
                verbose=False,
                progress_callback=lambda c, t, m: updates.append((c, t, m)),
            )
            self.assertFalse(res["stopped_early"])
            self.assertEqual(len(res["processed"]), 1)
            self.assertGreater(len(updates), 1)
            self.assertTrue(all(0.0 <= c <= t for c, t, _ in updates))
            self.assertEqual(updates[-1][0], updates[-1][1])
            self.assertTrue(
                all(updates[i][0] <= updates[i + 1][0] for i in range(len(updates) - 1))
            )

    def test_compute_chunked_rejects_non_3d_and_empty_arrays(self) -> None:
        empty_000 = np.zeros((0, 0, 0), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_chunked(empty_000, empty_000.astype(bool), nl=4, d=1, radius=1)

        empty_044 = np.zeros((0, 4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_chunked(empty_044, empty_044.astype(bool), nl=4, d=1, radius=1)

        arr_2d = np.ones((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            compute_chunked(arr_2d, arr_2d.astype(bool), nl=4, d=1, radius=1)


class TestTkinterGUI(unittest.TestCase):
    def tearDown(self) -> None:
        gc.collect()

    def _make_root(self) -> tk.Tk:
        root = tk.Tk()
        root.withdraw()
        return root

    def _cleanup_gui(
        self,
        root: tk.Tk | None,
        app: glcm_gui.GLCMGuiApp | None = None,
        *,
        traces: Sequence[tuple[tk.Variable, str, str]] = (),
        destroy_via_app: bool = True,
    ) -> None:
        if app is not None and app._worker is not None:
            app._worker.join(timeout=5.0)
            self.assertFalse(
                app._worker.is_alive(),
                "GUI worker thread did not finish before root destruction",
            )
        for var, mode, cbname in traces:
            try:
                var.trace_remove(mode, cbname)
            except tk.TclError:
                pass
        if app is not None:
            if destroy_via_app:
                app.close()
            elif root is not None:
                try:
                    root.destroy()
                except tk.TclError:
                    pass
            app._on_complete = None
            app._worker = None
        elif root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_gui_initializes_controls_and_default_config(self) -> None:
        root = self._make_root()
        app: glcm_gui.GLCMGuiApp | None = None
        try:
            app = glcm_gui.create_app(root)
            for widget in (
                app.input_entry,
                app.output_entry,
                app.browse_input_button,
                app.browse_output_button,
                app.run_button,
                app.progressbar,
                app.status_label,
            ):
                self.assertEqual(widget.winfo_exists(), 1)
            self.assertEqual(str(app.progressbar["mode"]), "determinate")
            self.assertEqual(float(app.progressbar["maximum"]), 100.0)
            self.assertEqual(app.config.min_free_disk_gb, 0.0)
        finally:
            self._cleanup_gui(root, app)

    def test_gui_browse_callbacks_and_validation(self) -> None:
        root = self._make_root()
        app: glcm_gui.GLCMGuiApp | None = None
        try:
            stub_fd = SimpleNamespace(
                askopenfilename=lambda **kwargs: "C:/fake/input_stack.tif",
                askdirectory=lambda **kwargs: "C:/fake/output_dir",
            )
            app = glcm_gui.create_app(root, filedialog_mod=stub_fd)
            app.browse_input()
            app.browse_output()
            self.assertEqual(app.input_var.get(), "C:/fake/input_stack.tif")
            self.assertEqual(app.output_var.get(), "C:/fake/output_dir")

            # Empty input validation
            app.input_var.set("")
            self.assertFalse(app.start_run())
            self.assertIn("Error:", app.status_var.get())
            self.assertEqual(str(app.run_button.cget("state")), "normal")

            # Missing input file validation
            app.input_var.set("Z:/definitely/does/not/exist.tif")
            self.assertFalse(app.start_run())
            self.assertIn("Error:", app.status_var.get())

            with tempfile.TemporaryDirectory() as tmp:
                bad_file = Path(tmp) / "image.png"
                bad_file.write_bytes(b"png")
                app.input_var.set(str(bad_file))
                self.assertFalse(app.start_run())
                self.assertIn(".tif", app.status_var.get())

                good_tif = Path(tmp) / "ok.tif"
                tifffile.imwrite(str(good_tif), np.ones((4, 4, 4), dtype=np.uint16))
                app.input_var.set(str(good_tif))
                app.output_var.set("")
                self.assertFalse(app.start_run())
                self.assertIn("output", app.status_var.get().lower())

                app.output_var.set(str(bad_file))
                self.assertFalse(app.start_run())
                self.assertIn("directory", app.status_var.get().lower())
        finally:
            self._cleanup_gui(root, app)

    def test_gui_end_to_end_synthetic_tif_workflow(self) -> None:
        rng = np.random.default_rng(2027)
        with tempfile.TemporaryDirectory() as tmp:
            in_tif = Path(tmp) / "gui_sample.tif"
            out_dir = Path(tmp) / "gui_out"
            vol_zyx = rng.integers(10, 300, size=(6, 10, 10), dtype=np.uint16)
            vol_zyx[:, :2, :] = 0
            tifffile.imwrite(str(in_tif), vol_zyx)

            cfg = GLCMConfig(
                d=1,
                neighbor_sizes=(1,),
                quant_level=4,
                features=("contr", "energ"),
                min_free_disk_gb=0.0,
                min_free_gb=0.0,
            )

            root = self._make_root()
            app: glcm_gui.GLCMGuiApp | None = None
            trace_id: str | None = None
            results: list[dict | Exception] = []
            progress_samples: list[float] = []

            def on_complete(res: dict | Exception) -> None:
                results.append(res)
                root.quit()

            try:
                app = glcm_gui.create_app(root, config=cfg, on_complete=on_complete)
                app.input_var.set(str(in_tif))
                app.output_var.set(str(out_dir))

                def _record_var(*_args) -> None:
                    progress_samples.append(float(app.progress_var.get()))

                trace_id = app.progress_var.trace_add("write", _record_var)

                started = app.start_run()
                self.assertTrue(started)
                self.assertEqual(str(app.run_button.cget("state")), "disabled")

                timeout_id = root.after(15000, root.quit)
                root.mainloop()
                root.after_cancel(timeout_id)

                self.assertEqual(len(results), 1, "GUI run did not complete before timeout")
                self.assertIsInstance(results[0], dict)
                summary = results[0]
                assert isinstance(summary, dict)
                self.assertFalse(summary["stopped_early"])
                self.assertEqual(len(summary["processed"]), 2)

                self.assertEqual(app.progress_var.get(), 100.0)
                self.assertIn("Done:", app.status_var.get())
                self.assertEqual(str(app.run_button.cget("state")), "normal")
                self.assertGreaterEqual(len(progress_samples), 2)
                self.assertTrue(
                    all(
                        progress_samples[i] <= progress_samples[i + 1]
                        for i in range(len(progress_samples) - 1)
                    )
                )

                for feat in ("contr", "energ"):
                    out_file = out_dir / make_out_name("gui_sample", feat, 1, 1, 4)
                    self.assertTrue(out_file.is_file())
                    arr = read_tiff_volume(out_file)
                    self.assertEqual(arr.dtype, np.float32)
                    self.assertEqual(arr.shape, (10, 10, 6))
                    np.testing.assert_array_equal(arr[:2, :, :], 0.0)
            finally:
                traces = (
                    [(app.progress_var, "write", trace_id)]
                    if (app is not None and trace_id is not None)
                    else ()
                )
                self._cleanup_gui(root, app, traces=traces)
                results.clear()

    def test_gui_error_and_stopped_early_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_tif = Path(tmp) / "sample.tif"
            out_dir = Path(tmp) / "out"
            tifffile.imwrite(str(in_tif), np.ones((4, 4, 4), dtype=np.uint16))

            def _run_error_case() -> None:
                root = self._make_root()
                app: glcm_gui.GLCMGuiApp | None = None
                seen_err: list[dict | Exception] = []

                def failing_runner(*args, **kwargs):
                    raise RuntimeError("boom")

                try:
                    app = glcm_gui.create_app(
                        root,
                        runner=failing_runner,
                        on_complete=lambda res: (seen_err.append(res), root.quit()),
                    )
                    app.input_var.set(str(in_tif))
                    app.output_var.set(str(out_dir))
                    self.assertTrue(app.start_run())
                    tid = root.after(5000, root.quit)
                    root.mainloop()
                    root.after_cancel(tid)

                    self.assertIsInstance(seen_err[0], RuntimeError)
                    self.assertIn("Error: boom", app.status_var.get())
                    self.assertEqual(str(app.run_button.cget("state")), "normal")
                finally:
                    self._cleanup_gui(root, app)
                    seen_err.clear()

            def _run_stopped_early_case() -> None:
                root2 = self._make_root()
                app2: glcm_gui.GLCMGuiApp | None = None
                seen_stop: list[dict | Exception] = []

                def stopped_runner(*args, **kwargs):
                    return {
                        "processed": [],
                        "skipped": [],
                        "stopped_early": True,
                        "intensity_range": (1.0, 1.0),
                    }

                try:
                    app2 = glcm_gui.create_app(
                        root2,
                        runner=stopped_runner,
                        on_complete=lambda res: (seen_stop.append(res), root2.quit()),
                    )
                    app2.input_var.set(str(in_tif))
                    app2.output_var.set(str(out_dir))
                    self.assertTrue(app2.start_run())
                    tid2 = root2.after(5000, root2.quit)
                    root2.mainloop()
                    root2.after_cancel(tid2)

                    self.assertEqual(len(seen_stop), 1)
                    self.assertIn("Stopped early", app2.status_var.get())
                    self.assertEqual(str(app2.run_button.cget("state")), "normal")
                finally:
                    self._cleanup_gui(root2, app2)
                    seen_stop.clear()

            _run_error_case()
            gc.collect()
            _run_stopped_early_case()

    def test_gui_main_help_and_pythonw_none_streams(self) -> None:
        buf = io.StringIO()
        with mock.patch.object(sys, "stdout", buf):
            with self.assertRaises(SystemExit) as ctx:
                glcm_gui.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("glcm3d-gui", buf.getvalue())
        gc.collect()

        # Simulate pythonw.exe where sys.stdout and sys.stderr are None
        with (
            mock.patch.object(sys, "stdout", None),
            mock.patch.object(sys, "stderr", None),
        ):
            root = self._make_root()
            app: glcm_gui.GLCMGuiApp | None = None
            try:
                app = glcm_gui.create_app(
                    root,
                    runner=lambda *a, **kw: {
                        "processed": ["a.tif"],
                        "skipped": [],
                        "stopped_early": False,
                        "intensity_range": (1.0, 2.0),
                    },
                    on_complete=lambda _res: root.quit(),
                )
                with tempfile.TemporaryDirectory() as tmp:
                    tif = Path(tmp) / "x.tif"
                    tifffile.imwrite(str(tif), np.ones((2, 2, 2), dtype=np.uint16))
                    app.input_var.set(str(tif))
                    app.output_var.set(str(Path(tmp) / "out"))
                    self.assertTrue(app.start_run())
                    tid = root.after(5000, root.quit)
                    root.mainloop()
                    root.after_cancel(tid)
                    self.assertIn("Done:", app.status_var.get())
            finally:
                self._cleanup_gui(root, app)

    def test_gui_start_run_returns_false_after_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_tif = Path(tmp) / "sample.tif"
            out_dir = Path(tmp) / "out"
            tifffile.imwrite(str(in_tif), np.ones((4, 4, 4), dtype=np.uint16))

            root = self._make_root()
            app = glcm_gui.create_app(root)
            try:
                app.input_var.set(str(in_tif))
                app.output_var.set(str(out_dir))
                app.close()
                self.assertTrue(app._destroyed)
                self.assertFalse(app.start_run())
            finally:
                self._cleanup_gui(root, app)

    def test_gui_repeated_create_start_destroy_under_stressed_gc(self) -> None:
        rng = np.random.default_rng(99)
        old_thresh = gc.get_threshold()
        with tempfile.TemporaryDirectory() as tmp:
            in_tif = Path(tmp) / "stress.tif"
            vol = rng.integers(10, 200, size=(5, 6, 6), dtype=np.uint16)
            vol[:, :1, :] = 0
            tifffile.imwrite(str(in_tif), vol)

            def _run_cycle(idx: int) -> tuple[weakref.ref[tk.Tk], weakref.ref[glcm_gui.GLCMGuiApp]]:
                root = self._make_root()
                app: glcm_gui.GLCMGuiApp | None = None
                results: list[dict | Exception] = []
                progress_msgs: list[str] = []
                out_dir = Path(tmp) / f"out_{idx}"
                destroy_via_app = idx in (0, 2, 4, 5)

                try:
                    if idx in (0, 2, 4):
                        def err_runner(*_args, **_kwargs):
                            # Allocate cyclic garbage inside the worker thread
                            cyc: list = []
                            cyc.append({"idx": idx, "self": cyc})
                            raise RuntimeError(f"boom-{idx}")

                        runner = err_runner
                        cfg = None
                    elif idx in (1, 3):
                        def prog_runner(*_args, progress_callback=None, **_kwargs):
                            for step in range(1, 5):
                                cyc: dict = {}
                                cyc["loop"] = cyc
                                if progress_callback is not None:
                                    progress_callback(float(step), 4.0, f"step {step}")
                            return {
                                "processed": ["ok.tif"],
                                "skipped": [],
                                "stopped_early": False,
                                "intensity_range": (1.0, 2.0),
                            }

                        runner = prog_runner
                        cfg = None
                    else:
                        runner = None
                        cfg = GLCMConfig(
                            d=1,
                            neighbor_sizes=(1,),
                            quant_level=4,
                            features=("contr",),
                            min_free_disk_gb=0.0,
                            min_free_gb=0.0,
                        )

                    app = glcm_gui.create_app(
                        root,
                        runner=runner,
                        config=cfg,
                        on_complete=lambda res: (results.append(res), root.quit()),
                    )
                    app.input_var.set(str(in_tif))
                    app.output_var.set(str(out_dir))

                    trace_id: str | None = None
                    if idx in (1, 3):
                        def _on_status(*_args) -> None:
                            progress_msgs.append(app.status_var.get())

                        trace_id = app.status_var.trace_add("write", _on_status)

                    self.assertTrue(app.start_run())
                    tid = root.after(15000, root.quit)
                    root.mainloop()
                    root.after_cancel(tid)

                    self.assertEqual(len(results), 1, f"Cycle {idx} timed out")
                    self.assertEqual(str(app.run_button.cget("state")), "normal")
                    if idx in (0, 2, 4):
                        self.assertIsInstance(results[0], RuntimeError)
                        self.assertIn(f"Error: boom-{idx}", app.status_var.get())
                    elif idx in (1, 3):
                        self.assertEqual(app.progress_var.get(), 100.0)
                        self.assertEqual(app.status_var.get(), "Done: wrote 1 file(s).")
                        self.assertTrue(any("step " in msg for msg in progress_msgs))
                    else:
                        summary = results[0]
                        assert isinstance(summary, dict)
                        self.assertFalse(summary["stopped_early"])
                        self.assertEqual(len(summary["processed"]), 1)
                        self.assertEqual(app.progress_var.get(), 100.0)
                        self.assertEqual(app.status_var.get(), "Done: wrote 1 file(s).")
                        out_file = out_dir / make_out_name("stress", "contr", 1, 1, 4)
                        self.assertTrue(out_file.is_file())
                        arr = read_tiff_volume(out_file)
                        self.assertEqual(arr.dtype, np.float32)
                        self.assertEqual(arr.shape, (6, 6, 5))

                    return weakref.ref(root), weakref.ref(app)
                finally:
                    traces = (
                        [(app.status_var, "write", trace_id)]
                        if (app is not None and trace_id is not None)
                        else ()
                    )
                    self._cleanup_gui(
                        root,
                        app,
                        traces=traces,
                        destroy_via_app=destroy_via_app,
                    )
                    results.clear()

            gc.set_threshold(50, 2, 2)
            try:
                refs: list[tuple[weakref.ref[tk.Tk], weakref.ref[glcm_gui.GLCMGuiApp]]] = []
                for cycle_idx in range(6):
                    wr_root, wr_app = _run_cycle(cycle_idx)
                    refs.append((wr_root, wr_app))
                    self.assertIsNone(
                        wr_root(),
                        f"Cycle {cycle_idx} leaked tk.Tk root without manual gc.collect()",
                    )
                    self.assertIsNone(
                        wr_app(),
                        f"Cycle {cycle_idx} leaked GLCMGuiApp without manual gc.collect()",
                    )
            finally:
                gc.set_threshold(*old_thresh)
                gc.collect()


if __name__ == "__main__":
    unittest.main()
