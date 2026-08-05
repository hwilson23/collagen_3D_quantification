import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
import tifffile
import pandas as pd
import re
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture

# =============================================================================
# PARAMETERS — edit these
# =============================================================================

#csv_path          = r"G:\FluorescentCollagen\20260519_flucol_kpc_ows3\20260519_stained_analysis\AVG_C2-flucol594_bkokpcwhoescht_Pos6_spots.csv"
#image_path        = r"G:\FluorescentCollagen\20260519_flucol_kpc_ows3\20260519_hoescht_col_channels\C2-flucol594_bkokpcwhoescht_800nm_blank_44530_37010_blank_g558555...scht_800nm_blank_44530_37010_blank_g558555_poc0_1_MMStack_Pos6.tif"
#out_dir           = r"G:\FluorescentCollagen\20260519_flucol_kpc_ows3\20260519_stained_analysis\trackmatemasks"
csv_path = r"G:\UTSW_BJChang\SUM_Cell7_bksub_spots.csv"
image_path = r"G:\UTSW_BJChang\Cell7_bksub_CH00.tif"
out_dir = r"G:\UTSW_BJChang\bksub_trackmatemasks"

pos_match = re.search(r'Pos(\d+)', csv_path)
pos_tag   = f"Pos{pos_match.group(1)}" if pos_match else ""

#xy_pixel_um       = 0.276   # microns per XY pixel
xy_pixel_um = 1 #.104
#z_step_um         = 2.0     # microns per Z slice
z_step_um = 1 #0.3
#sphere_radii  = [10.0, 20.0, 30.0]  # sphere radius in microns (isotropic in physical space)
sphere_radii = [200,250,350]
z_profile_disk_px = 10      # XY disk radius (pixels) used to compute Z intensity profile

# Boundary behaviour toggle:
#   True  → sphere is clipped at the Z boundary (truncated but included)
#   False → cell is excluded entirely if its center is within one radius of the boundary
clip_z_boundary   = True


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
                         disk_radius_px, n_z, label="", save_disk=False, disk_dir = None):
    H, W = image_tzyx.shape[2], image_tzyx.shape[3]
    rows, cols = np.ogrid[:H, :W]
    disk = ((rows - center_row)**2 + (cols - center_col)**2) <= disk_radius_px**2
    z_profile = np.array([
        image_tzyx[t,z][disk].mean() if disk.any() else 0.0
        for z in range(n_z)])
    '''
    plt.figure()
    plt.plot(z_profile, label='Z profile')
    plt.xlabel('Z Slice')
    plt.ylabel('Intensity')
    plt.title('Z Profile Analysis')
    plt.legend()
    plt.show()
    '''

    peaks = find_z_peaks(z_profile, max_components=2)
    z_center = int(round(float(peaks[0][0])))  # scalar int, safe for paint_sphere

    if len(peaks) > 1 and peaks[1][1] > 0.2:  # only warn if second peak is substantial
        print(f"    {label} ⚠ Two Z peaks: "
              f"z={peaks[0][0]:.1f} (w={peaks[0][1]:.2f}), "
              f"z={peaks[1][0]:.1f} (w={peaks[1][1]:.2f})")
    else:
        print(f"    {label} z_center={z_center}")
    
    '''
    ## TODO: optional overalay but need to combine into z stack for saving
    if save_disk:
        img = image_tzyx[t, z_center].copy()

        # Normalize for viewing
        img = img.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()

        overlay = np.stack([img, img, img], axis=-1)

        # Color the disk red
        overlay[disk] = [1, 0, 0]

        tifffile.imwrite(
            os.path.join(disk_dir, "overlay", f"{pos_tag}_particle{label}_t{t:03d}_overlay.tif"),
            (overlay * 255).astype(np.uint8)
        )
    '''
    return z_center, z_profile, peaks

# =============================================================================
# MAIN MASK BUILDER
# =============================================================================
def paint_sphere(volume, label, row, col, z_center,
                 r_xy_px, r_z_px, clip_z_boundary=True):
    zz, yy, xx = np.ogrid[:n_z, :H, :W]

    mask = (
        ((xx - col)/r_xy_px)**2 +
        ((yy - row)/r_xy_px)**2 +
        ((zz - z_center)/r_z_px)**2
    ) <= 1
    
    volume[mask] = label
    return volume

def create_3d_sphere_masks(image_tzyx, particles,
                           xy_pixel_um, z_step_um,
                           sphere_radius_um, z_profile_disk_px,
                           clip_z_boundary=True):
    """
    Build the full 4D Voronoi label volume AND per-cell 4D binary volumes.

    Returns:
        label_ztyx  (Z, T, Y, X) int32  — Voronoi label (particle_id+1 or 0)
        per_cell    list of (Z, T, Y, X) uint8 arrays, one per particle
        t_min, t_max
    """
    n_t, n_z, H, W = image_tzyx.shape
    r_xy_px = sphere_radius_um / xy_pixel_um
    r_z_px  = sphere_radius_um / z_step_um
    z_margin = int(np.ceil(r_z_px))

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

        # Per-frame working volumes
        any_cell = np.zeros((n_z, H, W), dtype=bool)
        raw_seeds = np.zeros((n_z, H, W), dtype=np.int32)

        painted_ids = []
        for particle_id, row, col in active:
            z_center, z_profile, peaks = find_z_center(
                image_tzyx, t_abs, row, col, z_profile_disk_px, n_z,particle_id, save_disk=True, disk_dir=out_dir)
            
            print(f"Z center for particle {particle_id} at t={t_abs}: {z_center}", end='\r')
            
            paint_sphere(label_ztyx[:, t_rel, :, :], particle_id + 1, row, col, z_center, r_xy_px, r_z_px)
            paint_sphere(per_cell[particle_id][:, t_rel, :, :], particle_id + 1, row, col, z_center, r_xy_px, r_z_px)
        print(f"label_ztyx shape: {label_ztyx.shape}, t={t_abs} painted with {len(active)} cells")
    return label_ztyx, per_cell, t_min, t_max


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    for sphere_radius_um in sphere_radii:
        # Derived parameters (for reporting)
        r_xy_px    = sphere_radius_um / xy_pixel_um
        r_z_px     = sphere_radius_um / z_step_um
        z_margin   = int(np.ceil(r_z_px))

        # ── Load image ──────────────────────────────────────────────────────────
        print("Loading image...")
        raw = tifffile.imread(image_path)
        print(f"  Raw shape: {raw.shape}")

        # Transpose (Z, T, X, Y) → (T, Z, Y, X)
        # Adjust axis order here if your file differs
        image_tzyx = raw
        print(f"  After transpose (T, Z, Y, X): {image_tzyx.shape}")

        n_t, n_z, H, W = image_tzyx.shape
        print(f"  T={n_t}  Z={n_z}  Y={H}  X={W}")

        # ── Parse CSV ───────────────────────────────────────────────────────────
        print("Parsing TrackMate CSV...")
        particles = parse_spots_csv(csv_path)
        n_particles = len(particles)
        print(f"  {n_particles} tracks loaded")

        all_frames = [t for track in particles for t in track.keys()]
        t_min, t_max = min(all_frames), max(all_frames)
        print(f"  Frame range: {t_min}–{t_max}")
        print(f"  Sphere radius: {sphere_radius_um} µm  "
            f"= {r_xy_px:.1f} XY px,  {r_z_px:.1f} Z slices")
        print(f"  Z boundary mode: {'clip (truncate sphere)' if clip_z_boundary else 'exclude cell'}")
        if not clip_z_boundary:
            print(f"  Cells excluded if z_center < {z_margin} or > {n_z - z_margin - 1}")

        # ── Build masks ─────────────────────────────────────────────────────────
        print("Building 3D sphere masks...")
        label_ztyx, per_cell, t_min, t_max = create_3d_sphere_masks(
            image_tzyx, particles,
            xy_pixel_um, z_step_um,
            sphere_radius_um, z_profile_disk_px,
            clip_z_boundary
        )
        


        # ── Save per-cell masks ─────────────────────────────────────────────────
        
        rad_tag   = f"r{int(sphere_radius_um)}um"

        cell_dir = os.path.join(out_dir, f"masks3d_per_cell_{pos_tag}_{rad_tag}")
        os.makedirs(cell_dir, exist_ok=True)
        print(f"Saving {n_particles} per-cell masks → {cell_dir}/")

        
        #write binary masks for each cell
        for pid, cell_vol in enumerate(per_cell):
            # Skip cells that were never painted (all zeros)
            if cell_vol.max() == 0:
                print(f"  Particle {pid:04d}: no frames painted, skipping")
                continue
            cell_path = os.path.join(cell_dir, f"{pos_tag}_{rad_tag}_cell_{pid:04d}.tif")
            
            
            cell_vol = np.transpose(cell_vol, (1, 0, 2, 3))  # (Z, T, Y, X) → (T, Z, Y, X)
            tifffile.imwrite(cell_path, np.uint16(cell_vol), imagej=True,
                            metadata={'axes': 'TZYX'})
            
        #write combined label volume
        label_path = os.path.join(out_dir, f"masks3d_combined_{pos_tag}_{rad_tag}.tif")
        label_ztyx = np.transpose(label_ztyx, (1, 0, 2, 3))  # (Z, T, Y, X) → (T, Z, Y, X)
        tifffile.imwrite(label_path, np.uint16(label_ztyx), imagej=True,
                        metadata={'axes': 'TZYX'})

        print(f"Done. {sum(1 for c in per_cell if c.max() > 0)} / {n_particles} "
            f"cells saved.")