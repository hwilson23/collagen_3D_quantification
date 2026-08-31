import os
import re
import tifffile as tiff
import pandas as pd
import numpy as np
import glob as glob
from concurrent.futures import ProcessPoolExecutor

def find_limits(input_path):
    texturetag = os.path.basename(input_path).split("3D")[0][-6:-1]
    image = tiff.imread(input_path, out='memmap')

    # single sort pass for both percentiles instead of two separate calls
    min_val, max_val = np.percentile(image, [1, 99])
    #min_val = image.min()
    #max_val = image.max()
    return texturetag, min_val, max_val

def normalize_one_file(args):
    ls, min_v, max_v, output_path = args
    filename = os.path.basename(ls)

    if max_v == min_v:
        print(f"Skipping {filename}: min == max ({min_v}), would divide by zero")
        return

    image = tiff.imread(ls, out = 'memmap')
    normalized_image = (image - min_v) / (max_v - min_v)
    tiff.imwrite(os.path.join(output_path, "norm_" + filename), normalized_image.astype('float32'))


def normalize_glcm_output(input_path, output_path):
    limits = {}
    tiff_files = glob.glob(os.path.join(input_path, "*.tif"))
    print(f"Found {len(tiff_files)} TIFF files in {input_path}")

    # ---- Phase 1: compute limits per texture tag, in parallel across files ----
    with ProcessPoolExecutor() as executor:
        for texturetag, min_val, max_val in executor.map(find_limits, tiff_files):
            min_key = f"{texturetag}_min"
            max_key = f"{texturetag}_max"
            limits[min_key] = min(limits.get(min_key, min_val), min_val)
            limits[max_key] = max(limits.get(max_key, max_val), max_val)

    print(f"Limits {limits}")

    # ---- Phase 2: normalize each file, in parallel, using precomputed limits ----
    tasks = []
    for ls in tiff_files:
        texturetagcurrent = os.path.basename(ls).split("3D")[0][-6:-1]
        min_v = limits.get(f"{texturetagcurrent}_min")
        max_v = limits.get(f"{texturetagcurrent}_max")
        tasks.append((ls, min_v, max_v, output_path))

    with ProcessPoolExecutor() as executor:
        list(executor.map(normalize_one_file, tasks))

def stitch_sections(output_path, output_stitched_path):
    print("Stitching normalized 3D TIFF images...")
    """
    Stitch together normalized 3D TIFF images into a single 3D TIFF image.

    Parameters:
    output_path (str): Path to the directory containing normalized 3D TIFF images.
    output_stitched_path (str): Path to save the stitched 3D TIFF image.
    """
    quad_pattern = re.compile(r"_quad(\d+)")
    eight_pattern = re.compile(r"_sect(\d+)")

    quad_groups = {}
    section_groups = {}

    # ---- Phase 1: find all groups of files ----
    for file in glob.glob(output_path + "/*.tif"):
        basename = os.path.basename(file)
        match_quad = quad_pattern.search(basename)
        match_eight = eight_pattern.search(basename)

        if match_quad:
            quad_num = int(match_quad.group(1))
            # Remove "_quadX" from grouping key
            key = re.sub(r"_quad\d+_t\d+", "", basename)
            quad_groups.setdefault(key, {})[quad_num] = file

        elif match_eight:
            eight_num = int(match_eight.group(1))
            # Remove "_sectX" from grouping key
            key = re.sub(r"_sect\d+_t\d+", "", basename)
            section_groups.setdefault(key, {})[eight_num] = file

        else:
            continue

    # ---- Phase 2a: stitch quadrant groups ----
    for key, quads in quad_groups.items():
        # Require all 4 quadrants
        if not all(q in quads for q in [1, 2, 3, 4]):
            print(f"Skipping incomplete set: {key}")
            continue

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
        stitched = np.zeros((h * 2, w * 2, z), dtype=q1.dtype)

        # Layout:
        # q1 q2
        # q3 q4
        stitched[0:h, 0:w, 0:z] = q1
        stitched[0:h, w:2*w, 0:z] = q2
        stitched[h:2*h, 0:w, 0:z] = q3
        stitched[h:2*h, w:2*w, 0:z] = q4

        stitched = np.moveaxis(stitched, -1, 0)
        tiff.imwrite(os.path.join(output_stitched_path, key), stitched)

    # ---- Phase 2b: stitch section groups ----
    for key, sections in section_groups.items():
        # Require all 8 sections
                
        if not all(s in sections for s in range(1, 9)):
            print(f"Skipping incomplete set: {key}")
            continue

        section_images = [tiff.imread(sections[s], out='memmap') for s in range(1, 9)]
        section_images = [np.moveaxis(img, 0, -1) for img in section_images]

        # Assumes all sections same shape
        h, w, z = section_images[0].shape
        stitched = np.zeros((h * 2, w * 4, z), dtype=section_images[0].dtype)

        # Layout:
        # s1 s2 s3 s4
        # s5 s6 s7 s8
        for i in range(2):
            for j in range(4):
                stitched[i*h:(i+1)*h, j*w:(j+1)*w, :] = section_images[i*4 + j]

        stitched = np.moveaxis(stitched, -1, 0)
        tiff.imwrite(os.path.join(output_stitched_path, key), stitched)



if __name__ == "__main__":
    #input_path = "G:\\UTSW_BJChang\\output_bksub_texture3D\\"
    input_path = "G:\\FluorescentCollagen\\20260519_flucol_kpc_ows3\\20260519_stained_analysis\\output_bksub_texture3D"
    output_path = "G:\\FluorescentCollagen\\20260519_flucol_kpc_ows3\\20260519_stained_analysis\\output_bksub_texture3D\\normalized\\"
    output_stitched_path = "G:\\FluorescentCollagen\\20260519_flucol_kpc_ows3\\20260519_stained_analysis\\output_bksub_texture3D\\stitched\\"
    print("Normalizing glcm output...")
    normalize_glcm_output(input_path, output_path)
    stitch_sections(output_path, output_stitched_path)
    print("done! :)")