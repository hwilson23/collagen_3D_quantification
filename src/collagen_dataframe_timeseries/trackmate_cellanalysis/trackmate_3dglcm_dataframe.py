import os
import sys
from pathlib import Path
import numpy as np
from numpy.f2py import main
import pandas as pd
import tifffile as tiff
import itertools
import re

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PATHS, EXTERNAL


def reshape_texture(df):
    # List of all possible value columns
    possible_values = ["concentration",
                        "type",
                        "roi",
                        "slice",
                        "timepoint",
                       "texture_mean",
                       "texture_median",
                       "texture_std", 
                       "texture3d_mean", 
                       "texture3d_median",
                       "texture3d_std", 
                       "distance3d", 
                       "neighbor3d", 
                       "bin_num3d"]
    
    # Filter to only include columns that exist in the dataframe
    available_values = [col for col in possible_values if col in df.columns]
    
    if not available_values:
        # If no values are available, just return the dataframe as-is
        return df
    
    pivotdf = df.pivot(
        index=["image_name","slice","position"],
        columns = ["mask_type","texture_type"],
        values = available_values
    )
    pivotdf.columns = [f"{mask}_{stat}_{tex}" for mask, stat, tex in pivotdf.columns]
    pivotdf = pivotdf.reset_index()

    return pivotdf

def compute_stats(pixels):
    pixels = pixels[pixels > 0]

    if len(pixels) == 0:
        return {
            "texture3d_mean": np.nan,
            "texture3d_median": np.nan,
            "texture3d_std": np.nan,
        }

    return {
        "texture3d_mean": np.mean(pixels),
        "texture3d_median": np.median(pixels),
        "texture3d_std": np.std(pixels),
    }

def image_stats_glcm3D(imagepath, mask_paths_dict, stackstats):

    nospace_name = os.path.basename(imagepath).replace(" ", "")

    img = tiff.imread(imagepath)
    img = np.moveaxis(img, 0, -1)

    idx = nospace_name.find("3D")
    timepoint = int(re.search(r'_t(\d+)', nospace_name).group(1))

    m = re.search(r'Pos(\d+)', nospace_name)
    pos = int(m.group(1)) if m else None
    # Load masks
    masks = {}
    for mask_name, folder_path in mask_paths_dict.items():
        for fname in os.listdir(folder_path):
        # must match BOTH position and mask type
            if (f"Pos{pos}" in fname) and (mask_name in fname):

                print(f"Loading mask: {fname} for position: {pos} and mask type: {mask_name}, timepoint is {timepoint}, cell number is {re.search(r'cell(\d+)', mask_name).group(1)}")
                full_path = os.path.join(folder_path, fname)

                mask_img = tiff.imread(full_path)
                mask_img = np.transpose(mask_img,(2,3,0,1))
                mask_img = mask_img[:,:,:,timepoint]

                masks[mask_name] = mask_img
    #print(masks)
    for z in range(img.shape[2]):

        currentim = img[:, :, z]

        # -----------------------------------
        # Whole image stats
        # -----------------------------------
        stats = compute_stats(currentim)

        imgstats = {
            "slice": z + 1,
            "image_name": nospace_name[:idx-7],
            "timepoint": nospace_name.split('_')[-7].split('t')[-1],
            "mask_type": "full",
            "texture_type": nospace_name.split('_')[-6],
            "position": int(m.group(1)) if m else np.nan,
            "distance3d": nospace_name.split('_')[-4],
            "neighbor3d": nospace_name.split('_')[-3],
            "bin_num3d": nospace_name.split('_')[-2],
            **stats
        }

        stackstats.append(imgstats)

        # -----------------------------------
        # Masked stats
        # -----------------------------------
        for mask_name, mask_stack in masks.items():

            current_mask = mask_stack[:, :, z]

            masked_pixels = currentim[current_mask > 0]

            stats = compute_stats(masked_pixels)
            #print("Stats for mask:", mask_name, stats)
            imgstats = {
                "slice": z + 1,
                "image_name": nospace_name[:idx-7],
                "timepoint": nospace_name.split('_')[-7].split('t')[-1],
                "mask_type": mask_name,
                "texture_type": nospace_name.split('_')[-6],
                "position": int(m.group(1)) if m else np.nan,
                "distance3d": nospace_name.split('_')[-4],
                "neighbor3d": nospace_name.split('_')[-3],
                "bin_num3d": nospace_name.split('_')[-2],
                **stats
            }

            stackstats.append(imgstats)

    return stackstats

def process_img_folder(folder, mask_paths, is_3d):
    stackstats = []
    if is_3d ==0:
        print("3d glcm required")
        stats = None
    elif is_3d ==1:
        for file in os.listdir(folder):
            if file.endswith((".tif",".tiff")):
                full = os.path.join(folder,file)
                stats = image_stats_glcm3D(
                                    full,
                                    mask_paths,
                                    stackstats
                                )
            
            #print(stats)
    return pd.DataFrame(stats)

def find_identical_columns(df):
    identical_groups = []
    checked = set()
    
    for col1, col2 in itertools.combinations(df.columns, 2):
        if col1 in checked or col2 in checked:
            continue
        # Compare row-by-row, ignoring NaNs
        if df[col1].equals(df[col2]):
            # Check if these are already part of a group
            added = False
            for group in identical_groups:
                if col1 in group or col2 in group:
                    group.add(col1)
                    group.add(col2)
                    added = True
                    break
            if not added:
                identical_groups.append({col1, col2})
            checked.add(col2)  # mark col2 as a duplicate
    
    return identical_groups


def collapse_identical_columns(df, groups):
    df = df.copy()
    dropped = []
    renamed = []

    for group in groups:
        keep = min(group, key=len)          # the name to keep
        dupes = group - {keep}              # all others to drop

        # Rename the kept column to your chosen base name (optional)
        base_name = keep  # or define your own logic here
        df = df.rename(columns={keep: base_name})

        # Drop the duplicates
        df = df.drop(columns=list(dupes))
        dropped.extend(dupes)
        renamed.append((keep, base_name, dupes))

    # Summary
    print("Collapsed columns:")
    for kept, new_name, dupes in renamed:
        print(f"  Kept '{new_name}', dropped: {dupes}")

    return df

if __name__ == "__main__":
    dftexture3D = pd.DataFrame()

    for pos in stacks:
        print(f"Processing position: {pos}")

        mask_paths = {
            "a1_masked":   str(PATHS[f"masks3d_per_cell_{pos}_r20um"]),
            "a2_masked":   str(PATHS[f"masks3d_per_cell_{pos}_r30um"]),
            "cell_masked": str(PATHS[f"masks3d_per_cell_{pos}_r10um"]),
        }
        dftexture3D = dftexture3D.append(process_img_folder(str(PATHS["texture3d"]), mask_paths, is_3d=1), ignore_index=True)
    
    dftexture3D = reshape_texture(dftexture3D)