import numpy as np
import tifffile
import os

def create_spiral_volume(
    vol_size        = (200, 200, 200),  # voxels (z, y, x)
    voxel_size_um   = 1.0,              # microns per voxel
    tube_radius     = 4,                # microns - thickness of tube
    num_turns       = 5,                # number of outward turns
    max_radius      = 80,               # microns - max outward radius
    z_spread        = 0.5,              # how much spiral rises in Z per turn (0 = flat)
    output_file     = 'spiral_volume.tiff'
):

    print(f"Creating outward spiral volume...")
    print(f"Volume size:    {vol_size} voxels")
    print(f"Voxel size:     {voxel_size_um} micron")
    print(f"Tube radius:    {tube_radius} microns")
    print(f"Num turns:      {num_turns}")
    print(f"Max radius:     {max_radius} microns")

    # ------------------------------------------------------------------ #
    #  Volume setup                                                        #
    # ------------------------------------------------------------------ #
    nz, ny, nx = vol_size
    vol = np.zeros(vol_size, dtype=np.float32)

    # Center of volume in microns
    cx = (nx / 2) * voxel_size_um
    cy = (ny / 2) * voxel_size_um
    cz = (nz / 2) * voxel_size_um

    # ------------------------------------------------------------------ #
    #  Archimedean spiral centerline                                       #
    #  radius grows linearly with angle: r = max_radius * (t / t_max)     #
    # ------------------------------------------------------------------ #
    t_max    = num_turns * 2 * np.pi          # total angle swept in radians
    num_pts  = 50000                           # more points = smoother tube
    t        = np.linspace(0, t_max, num_pts) # angle parameter

    # Radius grows from 0 to max_radius as t goes from 0 to t_max
    radius   = max_radius * (t / t_max)

    # XY plane spiral (Archimedean - starts at center, grows outward)
    spiral_x = radius * np.cos(t)
    spiral_y = radius * np.sin(t)

    # Z can be flat or gently rising
    # z_spread=0 gives a flat 2D spiral, z_spread>0 gives a cone spiral
    spiral_z = z_spread * (t / t_max) * nz * voxel_size_um - (z_spread * nz * voxel_size_um / 2)

    # ------------------------------------------------------------------ #
    #  Fill voxels within tube_radius of spiral centerline                #
    # ------------------------------------------------------------------ #
    tube_radius2 = tube_radius ** 2
    r_vox        = int(np.ceil(tube_radius / voxel_size_um))
    last_pct     = -1

    print(f"\nFilling voxels along centerline ({num_pts} points)...")

    for p in range(num_pts):

        # Progress
        pct = int(p / num_pts * 100)
        if pct % 10 == 0 and pct != last_pct:
            print(f"  {pct}%")
            last_pct = pct

        # Centerline point in microns (relative to center)
        px = spiral_x[p]
        py = spiral_y[p]
        pz = spiral_z[p]

        # Convert to voxel coordinates
        vx = int(round((px + cx) / voxel_size_um))
        vy = int(round((py + cy) / voxel_size_um))
        vz = int(round((pz + cz) / voxel_size_um))

        # Bounding box
        x_start = max(0,  vx - r_vox)
        x_end   = min(nx, vx + r_vox + 1)
        y_start = max(0,  vy - r_vox)
        y_end   = min(ny, vy + r_vox + 1)
        z_start = max(0,  vz - r_vox)
        z_end   = min(nz, vz + r_vox + 1)

        if x_start >= x_end or y_start >= y_end or z_start >= z_end:
            continue

        # Local coordinate grids in microns (relative to volume center)
        lx = np.arange(x_start, x_end) * voxel_size_um - cx
        ly = np.arange(y_start, y_end) * voxel_size_um - cy
        lz = np.arange(z_start, z_end) * voxel_size_um - cz

        LX, LY, LZ = np.meshgrid(lx, ly, lz, indexing='ij')

        # Distance squared to this centerline point
        dist2 = (LX - px)**2 + (LY - py)**2 + (LZ - pz)**2

        inside = dist2 <= tube_radius2

        # Write into volume (z,y,x indexing)
        vol[z_start:z_end, y_start:y_end, x_start:x_end] = np.where(
            inside.transpose(2, 1, 0),
            1.0,
            vol[z_start:z_end, y_start:y_end, x_start:x_end]
        )

    # ------------------------------------------------------------------ #
    #  Intensity gradient along spiral length (optional)                  #
    #  gives each turn a slightly different brightness                     #
    # ------------------------------------------------------------------ #
    print("\nApplying intensity gradient along spiral...")
    z_gradient = np.linspace(0.3, 1.0, nz).reshape(nz, 1, 1)
    vol = vol * z_gradient

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #
    filled = np.sum(vol > 0)
    total  = vol.size
    print(f"\n--- Volume Stats ---")
    print(f"Dimensions:       {vol.shape} (z, y, x)")
    print(f"Physical size:    {nz} x {ny} x {nx} microns")
    print(f"Filled voxels:    {filled:,}")
    print(f"Percent filled:   {filled/total*100:.2f}%")
    print(f"Intensity range:  {vol.min():.3f} - {vol.max():.3f}")

    # ------------------------------------------------------------------ #
    #  Save main volume                                                    #
    # ------------------------------------------------------------------ #
    metadata = {
        'axes':    'ZYX',
        'unit':    'um',
        'spacing': voxel_size_um,
    }
    
    print(f"\nSaving {output_file}...")
    tifffile.imwrite(
        output_file,
        vol,
        metadata = {'axes':'ZXY', 'loop':False},
    )

    return vol


# ------------------------------------------------------------------ #
#  Variants - uncomment to try different spiral shapes                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    # Cone spiral (z_spread > 0)
    # spirals outward AND upward through the volume
    vol = create_spiral_volume(
         vol_size      = (45, 256, 512),
         voxel_size_um = 1.0,
         tube_radius   = 3,
         num_turns     = 10,
         max_radius    = 100,
         z_spread      = 2,            # rises through Z as it spirals out
         output_file   = 'spiral_cone.tiff'
     )