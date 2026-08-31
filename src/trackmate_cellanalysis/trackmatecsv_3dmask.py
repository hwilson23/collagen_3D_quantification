import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import json
import numpy as np
import tifffile
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — avoids popping up windows during batch runs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture

# =============================================================================
# PARAMETERS — loaded from shared JSON config
# =============================================================================

CONFIG_PATH = r"C:\Users\hwilson23\Documents\GitHub\collagen_3D_quantification\config.json"  # <-- point this at your config.json

with open(CONFIG_PATH, 'r') as f:
    _cfg = json.load(f)

_mask_cfg   = _cfg['mask']
_params_cfg = _cfg['params']

csv_dir   = _mask_cfg['csv_path']    # folder containing per-position TrackMate CSVs
image_dir = _mask_cfg['image_path']  # folder containing per-position TIFF stacks
out_dir   = _mask_cfg['out_dir']

xy_pixel_um = _params_cfg['xy_pixel_um']   # microns per XY pixel
z_step_um   = _params_cfg['z_step']        # microns per Z slice
stacks      = _params_cfg['stacks']        # e.g. ["Pos0", "Pos1", "Pos2", "Pos4", "Pos5", "Pos6"]

# ---------------------------------------------------------------------------
# SHELL / DOUGHNUT RADII
# ---------------------------------------------------------------------------
# Each entry is [inner_radius_um, outer_radius_um] in config.json under mask.shell_radii_um.
#   - inner_radius_um = 0   -> solid sphere out to outer_radius_um (old behavior)
#   - inner_radius_um > 0   -> hollow shell: everything within inner_radius_um
#                               of the cell center is background (excluded),
#                               and only the region between inner and outer
#                               radius is painted with the cell's label.
# Example: outer radius 20 with inner radius 10 excluded -> [10, 20]
shell_radii_um = [tuple(pair) for pair in _mask_cfg['shell_radii_um']]

z_profile_disk_px = _mask_cfg['z_profile_disk_px']  # XY disk radius (pixels) used to compute Z intensity profile

# Boundary behaviour toggle:
#   True  → sphere is clipped at the Z boundary (truncated but included)
#   False → cell is excluded entirely if its center is within one radius of the boundary
clip_z_boundary = _mask_cfg['clip_z_boundary']


# =============================================================================
# RESOLVE PER-POSITION FILES
# =============================================================================

def find_position_file(directory, pos_tag, extension, prefix=None):
    """
    Find the single file in `directory` whose name contains `pos_tag`
    (e.g. 'Pos0') as a standalone token, not as a prefix of 'Pos10'/'Pos1' etc.,
    with the given extension.

    The (?!\\d) negative lookahead means 'Pos0' matches when followed by
    '-', '.', '_', or end-of-string, but NOT when followed by another digit —
    so searching for 'Pos1' won't accidentally match inside 'Pos10', and
    'Pos0' still matches an oddly-named file like '..._Pos0-1.tif'.

    If `prefix` is given, only filenames starting with that exact prefix are
    considered — e.g. prefix='C2-' selects 'C2-flucol..._Pos0.tif' while
    excluding 'C1-flucol..._Pos0.tif' and 'sub7000C1-flucol..._Pos0.tif'
    (the latter doesn't start with 'C1-', but this keeps channel selection
    explicit and safe regardless of naming variants).
    """
    candidates = [
        f for f in os.listdir(directory)
        if f.lower().endswith(extension.lower())
        and re.search(rf'{pos_tag}(?!\d)', f)
        and (prefix is None or f.startswith(prefix))
    ]
    if not candidates:
        prefix_note = f" with prefix '{prefix}'" if prefix else ""
        raise FileNotFoundError(
            f"No {extension} file matching '{pos_tag}'{prefix_note} in {directory}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple {extension} files matching '{pos_tag}' in {directory}: {candidates}")
    return os.path.join(directory, candidates[0])


# =============================================================================
# PARSE TRACKMATE CSV
# =============================================================================

def parse_spots_csv(csv_path):
    """Parse TrackMate spots CSV (auto-detected separator, 4 header rows).
    Returns list of per-particle dicts: [{frame: (x_um, y_um)}, ...]
    """
    df = pd.read_csv(csv_path, sep=None, engine='python', header=0, skiprows=[1, 2, 3])
    df = df[pd.to_numeric(df['TRACK_ID'], errors='coerce').notna()].copy()
    df = df.astype({'TRACK_ID': int, 'FRAME': int,
                    'POSITION_X': float, 'POSITION_Y': float})
    particles = defaultdict(dict)
    for _, row in df.iterrows():
        particles[row['TRACK_ID']][row['FRAME']] = (row['POSITION_X'], row['POSITION_Y'])
    return [particles[k] for k in sorted(particles.keys())]


# =============================================================================
# INTERPOLATE TRACKS
# =============================================================================

def interpolate_tracks(particles):
    """Linearly interpolate XY positions for gap frames within each track."""
    interpolated = []
    for track in particles:
        if not track:
            continue
        ts = sorted(track.keys())
        full_track = {}
        for i in range(len(ts) - 1):
            t0, t1 = ts[i], ts[i + 1]
            x0, y0 = track[t0]
            x1, y1 = track[t1]
            for t in range(t0, t1):
                alpha = (t - t0) / (t1 - t0)
                full_track[t] = (x0 + alpha * (x1 - x0),
                                 y0 + alpha * (y1 - y0))
        full_track[ts[-1]] = track[ts[-1]]
        interpolated.append(full_track)
    return interpolated


# =============================================================================
# Z PROFILE: find best focal Z slice for a cell
# =============================================================================


def gaussian(z, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((z - mu) / sigma) ** 2) + offset

def fit_z_gaussian(z_profile):
    z = np.arange(len(z_profile))
    try:
        p0 = [z_profile.max() - z_profile.min(),  # amp
              np.argmax(z_profile),                 # mu (initial guess = argmax)
              3.0,                                  # sigma (guess ~3 slices)
              z_profile.min()]                      # offset
        popt, _ = curve_fit(gaussian, z, z_profile, p0=p0, maxfev=2000)
        mu = popt[1]
        # Reject fit if center is outside the profile range
        if 0 <= mu <= len(z_profile) - 1:
            return float(mu), popt
    except RuntimeError:
        pass
    return float(np.argmax(z_profile)), None  # fallback to argmax

def find_z_peaks(z_profile, max_components=3):
    """
    Fit a 1- or 2-component GMM to the Z profile.
    Returns list of (z_center, weight) sorted by weight descending.
    Automatically selects number of components by BIC.
    """
    z = np.arange(len(z_profile))

    # Normalise profile to use as a probability-like weight
    p = z_profile - z_profile.min()
    p = p / p.sum() if p.sum() > 0 else np.ones(len(z)) / len(z)

    # Weighted sample: repeat each z index proportional to its intensity
    samples = np.repeat(z, np.round(p * 1000).astype(int))[:, None]

    best_bic = np.inf
    best_gmm = None
    best_n   = 1

    for n in range(1, max_components + 1):
        if len(samples) < n:
            break
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(samples)
        bic = gmm.bic(samples)
        if bic < best_bic:
            best_bic = best_gmm_bic = bic
            best_gmm = gmm
            best_n   = n

    centers = best_gmm.means_.flatten()
    weights = best_gmm.weights_.flatten()
    return sorted(zip(centers.tolist(), weights.tolist()),
                  key=lambda x: -x[1])

def find_z_center(image_tzyx, t, center_row, center_col,
                         disk_radius_px, n_z, label="", save_disk=False, disk_dir = None,
                         two_peak_pdf=None):
    H, W = image_tzyx.shape[2], image_tzyx.shape[3]
    rows, cols = np.ogrid[:H, :W]
    disk = ((rows - center_row)**2 + (cols - center_col)**2) <= disk_radius_px**2
    z_profile = np.array([
        image_tzyx[t,z][disk].mean() if disk.any() else 0.0
        for z in range(n_z)])

    peaks = find_z_peaks(z_profile, max_components=2)
    z_center = int(round(float(peaks[0][0])))  # scalar int, safe for paint_shell

    if len(peaks) > 1 and peaks[1][1] > 0.2:  # only warn if second peak is substantial
        print(f"    {label} ⚠ Two Z peaks: "
              f"z={peaks[0][0]:.1f} (w={peaks[0][1]:.2f}), "
              f"z={peaks[1][0]:.1f} (w={peaks[1][1]:.2f})")

        if two_peak_pdf is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(z_profile, marker='o', markersize=3, label='Z profile')
            colors = ['tab:red', 'tab:orange', 'tab:green']
            for i, (center, weight) in enumerate(peaks):
                ax.axvline(center, color=colors[i % len(colors)], linestyle='--',
                           label=f'peak {i+1}: z={center:.1f}, w={weight:.2f}')
            ax.set_xlabel('Z slice')
            ax.set_ylabel('Mean intensity')
            ax.set_title(f'Two Z peaks — particle {label}, frame t={t}')
            ax.legend(fontsize=8)
            fig.tight_layout()
            two_peak_pdf.savefig(fig)
            plt.close(fig)
    else:
        print(f"    {label} z_center={z_center}")

    return z_center, z_profile, peaks

# =============================================================================
# MAIN MASK BUILDER
# =============================================================================
def paint_shell(volume, label, row, col, z_center,
                r_xy_outer_px, r_z_outer_px,
                r_xy_inner_px, r_z_inner_px,
                clip_z_boundary=True):
    """
    Paint an ellipsoidal shell (doughnut) into `volume`.

    A voxel is painted with `label` if it lies WITHIN the outer ellipsoid
    AND OUTSIDE the inner ellipsoid. Passing r_xy_inner_px = r_z_inner_px = 0
    reduces this to a solid sphere/ellipsoid (old behavior).
    """
    n_z, H, W = volume.shape
    zz, yy, xx = np.ogrid[:n_z, :H, :W]

    outer_mask = (
        ((xx - col) / r_xy_outer_px) ** 2 +
        ((yy - row) / r_xy_outer_px) ** 2 +
        ((zz - z_center) / r_z_outer_px) ** 2
    ) <= 1

    if r_xy_inner_px > 0 and r_z_inner_px > 0:
        inner_mask = (
            ((xx - col) / r_xy_inner_px) ** 2 +
            ((yy - row) / r_xy_inner_px) ** 2 +
            ((zz - z_center) / r_z_inner_px) ** 2
        ) <= 1
        shell_mask = outer_mask & ~inner_mask
    else:
        shell_mask = outer_mask

    volume[shell_mask] = label
    return volume

def create_3d_shell_masks(image_tzyx, particles,
                          xy_pixel_um, z_step_um,
                          inner_radius_um, outer_radius_um, z_profile_disk_px,
                          clip_z_boundary=True, two_peak_pdf=None):
    """
    Build the full 4D Voronoi label volume AND per-cell 4D binary volumes,
    using a doughnut/shell shape between inner_radius_um and outer_radius_um.
    Set inner_radius_um = 0 for a solid sphere.

    Returns:
        label_ztyx  (Z, T, Y, X) int32  — Voronoi label (particle_id+1 or 0)
        per_cell    list of (Z, T, Y, X) uint8 arrays, one per particle
        t_min, t_max
    """
    n_t, n_z, H, W = image_tzyx.shape
    r_xy_outer_px = outer_radius_um / xy_pixel_um
    r_z_outer_px  = outer_radius_um / z_step_um
    r_xy_inner_px = inner_radius_um / xy_pixel_um
    r_z_inner_px  = inner_radius_um / z_step_um
    z_margin = int(np.ceil(r_z_outer_px))

    interpolated = interpolate_tracks(particles)
    n_particles  = len(interpolated)

    all_frames = [t for track in particles for t in track.keys()]
    t_min, t_max = min(all_frames), max(all_frames)
    n_frames = t_max - t_min + 1

    # Combined label volume (Z, T, Y, X)
    label_ztyx = np.zeros((n_z, n_frames, H, W), dtype=np.int32)

    # Per-cell binary volumes (Z, T, Y, X), stored as list of uint8
    per_cell = [np.zeros((n_z, n_frames, H, W), dtype=np.uint8)
                for _ in range(n_particles)]

    for t_abs in range(t_min, t_max + 1):
        t_rel = t_abs - t_min
        print(f"  Frame {t_abs}  ({t_rel + 1}/{n_frames})", end='\r')

        # Collect active cells
        active = []
        for particle_id, track in enumerate(interpolated):
            if t_abs not in track:
                continue
            x_um, y_um = track[t_abs]
            col = int(round(x_um / xy_pixel_um))
            row = int(round(y_um / xy_pixel_um))
            if 0 <= row < H and 0 <= col < W:
                active.append((particle_id, row, col))

        if not active:
            continue

        for particle_id, row, col in active:
            z_center, z_profile, peaks = find_z_center(
                image_tzyx, t_abs, row, col, z_profile_disk_px, n_z, particle_id, save_disk=True, disk_dir=out_dir,
                two_peak_pdf=two_peak_pdf)

            print(f"Z center for particle {particle_id} at t={t_abs}: {z_center}", end='\r')

            paint_shell(label_ztyx[:, t_rel, :, :], particle_id + 1, row, col, z_center,
                       r_xy_outer_px, r_z_outer_px, r_xy_inner_px, r_z_inner_px)
            paint_shell(per_cell[particle_id][:, t_rel, :, :], particle_id + 1, row, col, z_center,
                       r_xy_outer_px, r_z_outer_px, r_xy_inner_px, r_z_inner_px)
        print(f"label_ztyx shape: {label_ztyx.shape}, t={t_abs} painted with {len(active)} cells")
    return label_ztyx, per_cell, t_min, t_max


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    for pos_tag in stacks:

        # ── Resolve this position's CSV and image file from the config folders ──
        try:
            csv_path   = find_position_file(csv_dir, pos_tag, '.csv')
            image_path = find_position_file(image_dir, pos_tag, '.tif', prefix='C2-')
        except (FileNotFoundError, ValueError) as e:
            print(f"\n=== {pos_tag} === SKIPPED: {e}")
            continue

        print(f"\n=== {pos_tag} ===")
        print(f"  CSV:   {csv_path}")
        print(f"  Image: {image_path}")

        for inner_radius_um, outer_radius_um in shell_radii_um:
            # Derived parameters (for reporting)
            r_xy_outer_px = outer_radius_um / xy_pixel_um
            r_z_outer_px  = outer_radius_um / z_step_um
            r_xy_inner_px = inner_radius_um / xy_pixel_um
            r_z_inner_px  = inner_radius_um / z_step_um
            z_margin      = int(np.ceil(r_z_outer_px))

            # ── Load image ──────────────────────────────────────────────────────
            print("Loading image...")
            raw = tifffile.imread(image_path)
            print(f"  Raw shape: {raw.shape}")

            # Transpose (Z, T, X, Y) → (T, Z, Y, X)
            # Adjust axis order here if your file differs
            image_tzyx = raw
            print(f"  After transpose (T, Z, Y, X): {image_tzyx.shape}")

            n_t, n_z, H, W = image_tzyx.shape
            print(f"  T={n_t}  Z={n_z}  Y={H}  X={W}")

            # ── Parse CSV ───────────────────────────────────────────────────────
            print("Parsing TrackMate CSV...")
            particles = parse_spots_csv(csv_path)
            n_particles = len(particles)
            print(f"  {n_particles} tracks loaded")

            all_frames = [t for track in particles for t in track.keys()]
            t_min, t_max = min(all_frames), max(all_frames)
            print(f"  Frame range: {t_min}–{t_max}")
            if inner_radius_um > 0:
                print(f"  Shell: inner {inner_radius_um} µm -> outer {outer_radius_um} µm  "
                    f"= inner {r_xy_inner_px:.1f}/{r_z_inner_px:.1f} px (xy/z), "
                    f"outer {r_xy_outer_px:.1f}/{r_z_outer_px:.1f} px (xy/z)")
            else:
                print(f"  Solid sphere radius: {outer_radius_um} µm  "
                    f"= {r_xy_outer_px:.1f} XY px,  {r_z_outer_px:.1f} Z slices")
            print(f"  Z boundary mode: {'clip (truncate sphere)' if clip_z_boundary else 'exclude cell'}")
            if not clip_z_boundary:
                print(f"  Cells excluded if z_center < {z_margin} or > {n_z - z_margin - 1}")

            # rad_tag computed up front so it can be used for the two-peak PDF filename too
            if inner_radius_um > 0:
                rad_tag = f"r{int(inner_radius_um)}to{int(outer_radius_um)}um"
            else:
                rad_tag = f"r{int(outer_radius_um)}um"

            # ── Build masks ─────────────────────────────────────────────────────
            print("Building 3D shell masks...")

            two_peak_pdf_path = os.path.join(out_dir, f"z_profile_two_peaks_{pos_tag}_{rad_tag}.pdf")
            os.makedirs(out_dir, exist_ok=True)
            with PdfPages(two_peak_pdf_path) as two_peak_pdf:
                label_ztyx, per_cell, t_min, t_max = create_3d_shell_masks(
                    image_tzyx, particles,
                    xy_pixel_um, z_step_um,
                    inner_radius_um, outer_radius_um, z_profile_disk_px,
                    clip_z_boundary, two_peak_pdf=two_peak_pdf
                )
                n_two_peak_pages = two_peak_pdf.get_pagecount()

            if n_two_peak_pages == 0:
                os.remove(two_peak_pdf_path)  # nothing to show, don't leave an empty pdf behind
                print("  No two-peak Z profiles detected — no PDF saved.")
            else:
                print(f"  Saved {n_two_peak_pages} two-peak Z profile plot(s) → {two_peak_pdf_path}")

            # ── Save per-cell masks ─────────────────────────────────────────────
            cell_dir = os.path.join(out_dir, f"masks3d_per_cell_{pos_tag}_{rad_tag}")
            os.makedirs(cell_dir, exist_ok=True)
            print(f"Saving {n_particles} per-cell masks → {cell_dir}/")

            # write binary masks for each cell
            for pid, cell_vol in enumerate(per_cell):
                # Skip cells that were never painted (all zeros)
                if cell_vol.max() == 0:
                    print(f"  Particle {pid:04d}: no frames painted, skipping")
                    continue
                cell_path = os.path.join(cell_dir, f"{pos_tag}_{rad_tag}_cell_{pid:04d}.tif")

                cell_vol = np.transpose(cell_vol, (1, 0, 2, 3))  # (Z, T, Y, X) → (T, Z, Y, X)
                tifffile.imwrite(cell_path, np.uint16(cell_vol), imagej=True,
                                metadata={'axes': 'TZYX'})

            # write combined label volume
            label_path = os.path.join(out_dir, f"masks3d_combined_{pos_tag}_{rad_tag}.tif")
            label_ztyx = np.transpose(label_ztyx, (1, 0, 2, 3))  # (Z, T, Y, X) → (T, Z, Y, X)
            tifffile.imwrite(label_path, np.uint16(label_ztyx), imagej=True,
                            metadata={'axes': 'TZYX'})

            print(f"Done. {sum(1 for c in per_cell if c.max() > 0)} / {n_particles} "
                f"cells saved.")