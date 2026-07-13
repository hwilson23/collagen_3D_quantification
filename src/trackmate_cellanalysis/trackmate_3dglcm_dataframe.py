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
from config import PATHS, PARAMS


def reshape_texture(df):
    # Columns that are true metadata: constant per (image, slice, position, cell)
    # regardless of mask_type/texture_type -- these should NOT be pivoted, just
    # kept as a single column.
    metadata_cols = [
        "distance3d",
        "neighbor3d",
        "bin_num3d",
        "concentration",
        "type",
        "roi",
    ]
    metadata_cols = [col for col in metadata_cols if col in df.columns]

    # Columns that are genuinely computed per mask_type + texture_type -- these
    # SHOULD be pivoted and keep the suffix.
    stat_cols = [
        "texture_mean",
        "texture_median",
        "texture_std",
        "texture3d_mean",
        "texture3d_median",
        "texture3d_std",
    ]
    stat_cols = [col for col in stat_cols if col in df.columns]

    index_cols = ["image_name", "slice", "position", "cell","timepoint"]

    if not stat_cols:
        return df

    # --- Sanity check: confirm metadata cols are actually constant per index ---
    # If a "metadata" column secretly varies by mask_type/texture_type for the
    # same index group, collapsing it to one value would silently lose data.
    # Warn instead of guessing.
    safe_metadata_cols = []
    for col in metadata_cols:
        nunique_per_group = df.groupby(index_cols)[col].nunique(dropna=False)
        if (nunique_per_group <= 1).all():
            safe_metadata_cols.append(col)
        else:
            print(f"WARNING: '{col}' varies within an index group "
                  f"(image_name/slice/position/cell) -- leaving it out of "
                  f"collapse and it will NOT appear in the pivoted output. "
                  f"Inspect this column manually.")

    # --- Metadata: one row per index, no mask/texture suffix ---
    if safe_metadata_cols:
        meta_df = (
            df.groupby(index_cols)[safe_metadata_cols]
            .first()
            .reset_index()
        )
    else:
        meta_df = df[index_cols].drop_duplicates()

    # --- Real stats: pivot on mask_type + texture_type, keep suffix ---
    stat_df = df.pivot(
        index=index_cols,
        columns=["mask_type", "texture_type"],
        values=stat_cols,
    )
    stat_df.columns = [f"{stat}_{mask}_{tex}" for stat, mask, tex in stat_df.columns]
    stat_df = stat_df.reset_index()

    pivotdf = meta_df.merge(stat_df, on=index_cols, how="right")

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

def image_stats_glcm3D(pos, imagepath, mask_paths_dict, stackstats):

    nospace_name = os.path.basename(imagepath).replace(" ", "")

    img = tiff.imread(imagepath, out='memmap')
    img = np.moveaxis(img, 0, -1)
    

    idx = nospace_name.find("3D")
    timepoint = int(re.search(r'_t(\d+)', nospace_name).group(1))-1  # Adjusted for zero-based indexing

    # Load masks
    # Load masks
    masks = {}
    for mask_name, folder_path in mask_paths_dict.items():
        masks[mask_name] = {}
        for fname in os.listdir(folder_path):
            if (pos in fname) and (mask_name in fname):
                full_path = os.path.join(folder_path, fname)
                mask_img = tiff.imread(full_path, out='memmap')
                mask_img = np.transpose(mask_img,(2,3,1,0))
                mask_img = mask_img[:,:,:,timepoint]

                cell_num = re.search(r'cell_(\d+)', fname).group(1)
                masks[mask_name][cell_num] = {
                    'mask_stack': mask_img,
                    'timepoint': timepoint,
                    'cell': cell_num
                }
    #print(masks)
    for z in range(img.shape[2]):
        #print(f"Loading mask: {fname} for position: {pos} and mask type: {mask_name}, timepoint is {timepoint}, z is {z}, cell number is {re.search(r'cell_(\d+)', fname).group(1)}")

        currentim = img[:, :, z]

        # -----------------------------------
        # Whole image stats
        # -----------------------------------
        stats = compute_stats(currentim)

        imgstats = {
            "slice": z + 1,
            "image_name": nospace_name[:idx-7],
            "timepoint": int(re.search(r'_t(\d+)', nospace_name).group(1))-1,
            "mask_type": "full",
            "texture_type": nospace_name.split('_')[-6],
            "position": re.search(r'Pos(\d+)',pos).group(1),
            "distance3d": nospace_name.split('_')[-4],
            "neighbor3d": nospace_name.split('_')[-3],
            "bin_num3d": nospace_name.split('_')[-2],
            "cell": "full",
            **stats
        }

        stackstats.append(imgstats)

        # -----------------------------------
        # Masked stats
        # -----------------------------------
        for mask_type, cells in masks.items():
            for cell_num, data in cells.items():
                current_mask = data["mask_stack"][:, :, z]
                masked_pixels = currentim[current_mask > 0]
                stats = compute_stats(masked_pixels)

                imgstats = {
                    "slice": z + 1,
                    "cell": data["cell"],
                    "image_name": nospace_name[:idx-7],
                    "timepoint": data["timepoint"],
                    "mask_type": mask_type,
                    "texture_type": nospace_name.split('_')[-6],
                    "position": re.search(r'Pos(\d+)', pos).group(1),
                    "distance3d": nospace_name.split('_')[-4],
                    "neighbor3d": nospace_name.split('_')[-3],
                    "bin_num3d": nospace_name.split('_')[-2],
                    **stats
                }

                stackstats.append(imgstats)
            

    return stackstats

def process_img_folder(pos, folder, mask_paths, is_3d):
    stackstats = []
    if is_3d ==0:
        print("3d glcm required")
        stats = None
    elif is_3d ==1:
        for file in os.listdir(folder):
            if pos in file and file.endswith((".tif",".tiff")):
                full = os.path.join(folder,file)
                stats = image_stats_glcm3D(
                                    pos,
                                    full,
                                    mask_paths,
                                    stackstats
                                )
        if stats is None:
            print(f"No matching files found for position {pos} in folder {folder}.")
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

    for pos in PARAMS["stacks"]: 
        print(f"Processing position: {pos}")

        mask_paths = {
            "r20":   str(PATHS["masks"] / f"masks3d_per_cell_{pos}_r20um"),
            "r30":   str(PATHS["masks"] / f"masks3d_per_cell_{pos}_r30um"),
            "r10": str(PATHS["masks"] / f"masks3d_per_cell_{pos}_r10um"),
        }
        dftexture3D = pd.concat([dftexture3D, process_img_folder(pos, str(PATHS["texture3d"]), mask_paths, is_3d=1)], ignore_index=True)
        
    dftexture3D = reshape_texture(dftexture3D)

    groups = find_identical_columns(dftexture3D)
    if groups:
        print(f"Found {len(groups)} group(s) of identical columns:")
        for g in groups:
            print(" ", g)
        collapseddf = collapse_identical_columns(dftexture3D, groups)
        print(collapseddf.columns.values)
    else:
        collapseddf = dftexture3D
        print("No exactly identical columns found.")


    if any(col.startswith("n_") for col in collapseddf.columns):
        collapseddf = collapseddf.rename(columns={col: col.replace(col, "fibercount") for col in collapseddf.columns if col.startswith("n")})
    if any(col.startswith("z_depth_") for col in collapseddf.columns):
        collapseddf = collapseddf.rename(columns={col: col.replace(col, "z_depth") for col in collapseddf.columns if col.startswith("z_depth")})

    # Split into position dataframes if any type-like column exists
    unique_pos = collapseddf['position'].unique()
    for pos in unique_pos:
        pos_df = collapseddf[collapseddf['position'] == pos]
        
        pos_df.to_csv(f"current_final_dataframe_byslice_pos_{pos}_3Dtrackmate.csv", index=False)
        
        print("Saved position dataframes separately")

    # Also save the combined dataframe
    collapseddf.to_csv("finalcollapsed_dataframe_byslice_trackmate.csv", index=False)