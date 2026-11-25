"""Tkinter GUI application for single-file 3D GLCM texture extraction."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import os
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Any, Callable, Sequence

from .config import GLCMConfig
from .pipeline import run_single_file


def _runner_accepts_verbose(runner: Callable[..., Any]) -> bool:
    """Return True if *runner* accepts a ``verbose`` keyword argument."""
    try:
        sig = inspect.signature(runner)
    except (TypeError, ValueError):
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ) or ("verbose" in sig.parameters)


def _run_worker(
    runner: Callable[..., dict],
    in_str: str,
    out_str: str,
    runner_kwargs: dict[str, Any],
    out_queue: queue.Queue[tuple],
) -> None:
    """Execute *runner* in a background thread without holding any Tkinter refs."""

    def on_progress(current: float, total: float, message: str) -> None:
        out_queue.put(("progress", current, total, message))

    call_kwargs = dict(runner_kwargs)
    call_kwargs["progress_callback"] = on_progress
    try:
        summary = runner(in_str, out_str, **call_kwargs)
        out_queue.put(("done", summary))
    except Exception as exc:
        out_queue.put(("error", exc))


class GLCMGuiApp:
    """Single-file `.tif` selection and progress-tracking Tkinter GUI."""

    def __init__(
        self,
        root: tk.Tk | tk.Toplevel,
        *,
        runner: Callable[..., dict] | None = None,
        config: GLCMConfig | None = None,
        filedialog_mod: Any = None,
        on_complete: Callable[[dict | Exception], None] | None = None,
    ) -> None:
        self.root = root
        self.runner = runner or run_single_file
        self.config = config if config is not None else GLCMConfig(min_free_disk_gb=0.0)
        self._filedialog = filedialog_mod or filedialog
        self._on_complete = on_complete

        self._queue: queue.Queue[tuple] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False
        self._destroyed = False
        self._after_id: str | None = None

        self.frame = ttk.Frame(root, padding=12)
        self.frame.pack(fill="both", expand=True)
        self.frame.columnconfigure(1, weight=1)

        # Input .tif selection
        ttk.Label(self.frame, text="Input .tif file:").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.input_var = tk.StringVar(master=self.frame, value="")
        self.input_entry = ttk.Entry(self.frame, textvariable=self.input_var, width=52)
        self.input_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self.browse_input_button = ttk.Button(
            self.frame, text="Browse...", command=self.browse_input
        )
        self.browse_input_button.grid(row=0, column=2, padx=(8, 0), pady=4)

        # Output folder selection
        ttk.Label(self.frame, text="Output folder:").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.output_var = tk.StringVar(master=self.frame, value="")
        self.output_entry = ttk.Entry(self.frame, textvariable=self.output_var, width=52)
        self.output_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self.browse_output_button = ttk.Button(
            self.frame, text="Browse...", command=self.browse_output
        )
        self.browse_output_button.grid(row=1, column=2, padx=(8, 0), pady=4)

        # Run button
        self.run_button = ttk.Button(self.frame, text="Run", command=self.start_run)
        self.run_button.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 8))

        # Progress bar & status
        self.progress_var = tk.DoubleVar(master=self.frame, value=0.0)
        self.progressbar = ttk.Progressbar(
            self.frame,
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.progress_var,
        )
        self.progressbar.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 6))

        self.status_var = tk.StringVar(
            master=self.frame,
            value="Select an input .tif file and output folder.",
        )
        self.status_label = ttk.Label(
            self.frame, textvariable=self.status_var, anchor="w"
        )
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="ew")

        self.frame.bind("<Destroy>", self._on_destroy)
        if hasattr(self.root, "protocol"):
            try:
                self.root.protocol("WM_DELETE_WINDOW", self.close)
            except tk.TclError:
                pass

    def browse_input(self) -> None:
        chosen = self._filedialog.askopenfilename(
            title="Select input .tif file",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if chosen:
            self.input_var.set(str(chosen))

    def browse_output(self) -> None:
        chosen = self._filedialog.askdirectory(title="Select output folder")
        if chosen:
            self.output_var.set(str(chosen))

    def _cancel_after(self) -> None:
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _on_destroy(self, event: tk.Event | None = None) -> None:
        if event is None or event.widget is self.frame:
            self._destroyed = True
            self._cancel_after()
            self._on_complete = None
            self._worker = None

    def close(self) -> None:
        """Cancel pending polling callbacks and close the root window."""
        self._destroyed = True
        self._cancel_after()
        self._on_complete = None
        self._worker = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def start_run(self) -> bool:
        """Validate selections and launch single-file GLCM processing in a thread."""
        if self._running or self._destroyed:
            return False

        in_str = self.input_var.get().strip()
        out_str = self.output_var.get().strip()

        if not in_str:
            self.status_var.set("Error: Please select an input .tif file.")
            return False
        in_path = Path(in_str)
        if not in_path.exists() or not in_path.is_file():
            self.status_var.set(f"Error: Input file not found: {in_path}")
            return False
        if in_path.suffix.lower() not in {".tif", ".tiff"}:
            self.status_var.set(
                f"Error: Input file must be .tif or .tiff (got '{in_path.suffix}')."
            )
            return False

        if not out_str:
            self.status_var.set("Error: Please select an output folder.")
            return False
        out_path = Path(out_str)
        if out_path.exists() and not out_path.is_dir():
            self.status_var.set(
                f"Error: Output path is a file, expected a directory: {out_path}"
            )
            return False

        self._running = True
        self.run_button.configure(state="disabled")
        self.progress_var.set(0.0)
        self.status_var.set(f"Starting {in_path.name}...")

        # Drain any stale messages before starting a new run
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        runner_kwargs: dict[str, Any] = {"config": self.config}
        if _runner_accepts_verbose(self.runner):
            runner_kwargs["verbose"] = False

        self._worker = threading.Thread(
            target=_run_worker,
            args=(self.runner, in_str, out_str, runner_kwargs, self._queue),
            daemon=True,
        )
        self._worker.start()
        self._after_id = self.root.after(25, self._poll_queue)
        return True

    def _poll_queue(self) -> None:
        self._after_id = None
        if self._destroyed:
            return

        try:
            while not self._destroyed:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, current, total, msg = item
                    pct = (
                        min(100.0, max(0.0, 100.0 * float(current) / float(total)))
                        if float(total) > 0.0
                        else 0.0
                    )
                    self.progress_var.set(pct)
                    self.status_var.set(str(msg))
                elif kind == "done":
                    _, summary = item
                    self._running = False
                    self.run_button.configure(state="normal")
                    if summary.get("stopped_early"):
                        self.status_var.set(
                            "Stopped early: insufficient free disk space."
                        )
                    elif not summary.get("processed") and summary.get("skipped"):
                        self.progress_var.set(100.0)
                        skipped_names = ", ".join(summary["skipped"])
                        self.status_var.set(
                            f"Skipped (no foreground or already exists): {skipped_names}"
                        )
                    else:
                        self.progress_var.set(100.0)
                        n_written = len(summary.get("processed", []))
                        self.status_var.set(f"Done: wrote {n_written} file(s).")
                    if self._on_complete is not None:
                        self._on_complete(summary)
                    return
                elif kind == "error":
                    _, exc = item
                    self._running = False
                    self.run_button.configure(state="normal")
                    self.status_var.set(f"Error: {exc}")
                    if self._on_complete is not None:
                        self._on_complete(exc)
                    return
        except queue.Empty:
            pass
        except tk.TclError:
            self._destroyed = True
            return

        if self._running and not self._destroyed:
            try:
                self._after_id = self.root.after(25, self._poll_queue)
            except tk.TclError:
                self._destroyed = True


def create_app(root: tk.Tk | None = None, **kwargs: Any) -> GLCMGuiApp:
    """Create and configure the ``glcm3D`` Tkinter application."""
    if root is None:
        root = tk.Tk()
    root.title("glcm3D - Single-File Texture Extraction")
    root.minsize(520, 200)
    return GLCMGuiApp(root, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="glcm3d-gui",
        description="Launch the glcm3D single-file Tkinter GUI.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``glcm3d-gui`` and ``python -m glcm3d.gui``."""
    with contextlib.ExitStack() as stack:
        if sys.stdout is None:
            devnull_out = stack.enter_context(
                open(os.devnull, "w", encoding="utf-8")
            )
            stack.enter_context(contextlib.redirect_stdout(devnull_out))
        if sys.stderr is None:
            devnull_err = stack.enter_context(
                open(os.devnull, "w", encoding="utf-8")
            )
            stack.enter_context(contextlib.redirect_stderr(devnull_err))

        _build_parser().parse_args(argv)
        root = tk.Tk()
        create_app(root)
        root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
