from pathlib import Path
import re
import numpy as np
import tifffile as tiff

# Folder containing quadrant TIFFs
input_folder = Path("G:\\FluorescentCollagen\\20260519_flucol_kpc_ows3\\selectedpos\\output_bksub_texture3D\\")
output_folder = input_folder / "G:\\FluorescentCollagen\\20260519_flucol_kpc_ows3\\selectedpos\\output_bksub_texture3D\\stitched"
output_folder.mkdir(exist_ok=True)

# Regex to detect quadrant number
quad_pattern = re.compile(r"_quad(\d+)")

# Group files by removing the quadrant part from filename
groups = {}

for file in input_folder.glob("*.tif*"):
    match = quad_pattern.search(file.name)

    if not match:
        continue

    quad_num = int(match.group(1))

    # Remove "_quadX" from grouping key
    key = re.sub(r"_quad\d+_t\d+", "", file.stem)

    groups.setdefault(key, {})[quad_num] = file

# ---- Stitch images ----
for key, quads in groups.items():

    # Require all 4 quadrants
    if not all(q in quads for q in [1, 2, 3, 4]):
        print(f"Skipping incomplete set: {key}")
        continue

    # Read quadrants
    q1 = tiff.imread(quads[1])
    q2 = tiff.imread(quads[2])
    q3 = tiff.imread(quads[3])
    q4 = tiff.imread(quads[4])

    q1 = np.moveaxis(q1, 0, -1)
    q2 = np.moveaxis(q2, 0, -1) 
    q3 = np.moveaxis(q3, 0, -1)
    q4 = np.moveaxis(q4, 0, -1)

   

    # Assumes all quadrants same shape
    h, w, z = q1.shape
    stitched = np.zeros((h * 2, w * 2,z), dtype=q1.dtype)
    

    # Layout:
    # q1 q2
    # q3 q4

    stitched[0:h, 0:w, 0:z] = q1
    stitched[0:h, w:2*w, 0:z] = q2
    stitched[h:2*h, 0:w, 0:z] = q3
    stitched[h:2*h, w:2*w, 0:z] = q4

    


    # Save stitched image
    output_path = output_folder / f"{key}_stitched.tif"

    stitched = np.moveaxis(stitched, -1, 0)
    tiff.imwrite(output_path, stitched)

    print(f"Saved: {output_path}")