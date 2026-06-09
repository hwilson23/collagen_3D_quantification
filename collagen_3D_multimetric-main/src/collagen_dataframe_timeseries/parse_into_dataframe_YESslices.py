import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile as tiff
import itertools
import re

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PATHS, EXTERNAL


def load_csv(file, columns=None):
    df = pd.read_csv(file)
    if columns:
        df = df[columns]
    return df 

def reshape_CA(df):
    # Extract base file name (starting with 'sub' and ending before mask type)
    # and mask type (e.g., a1masked, a2masked, cellmasked)
    def extract_base_and_mask(name):
        mask_types = ["a1_masked", "a2_masked", "cell_masked"]

        for mask in mask_types:
            prefix = mask + "_"
            if name.startswith(prefix):
                base_name = name[len(prefix):]
                mask = mask.replace("_masked", "")
                return base_name, mask

        return name, "unknown"
    df = df.copy()
    df[['base_name', 'mask_type']] = df['image_name'].apply(
        lambda x: pd.Series(extract_base_and_mask(str(x)))
    )
    df['position'] = df['base_name'].str.extract(r'Pos(\d+)')[0].astype(int)
    # Pivot so each feature is stored per mask type under the same base file name
    pivotdf = df.pivot_table(
    index=["base_name", "slice", "position"],
    columns=["mask_type", "hist_type"],
    values=["mean","n","std","median","z_depth"]
    )
    pivotdf.columns = [
    f"{val}_{mask}_{hist}"
    for val, mask, hist in pivotdf.columns
    ]
    pivotdf = pivotdf.reset_index().rename(columns={'base_name': 'image_name'})
    
    
    return pivotdf

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

def image_stats_glcm2D(imagepath, stackstats):
    img = tiff.imread(imagepath)
    img = np.moveaxis(img,0,-1)
    #print(f"im shape {img.shape}")
       

    #index of end filename
    idx = os.path.basename(imagepath).find("8bit")
    #print(os.path.basename(imagepath)[:idx+len("8bit.ome")])
    
    for z in range(img.shape[2]):
        currentim = img[:,:,z]
        imgstats = {
            "slice" : z+1,
            "image_name": os.path.basename(imagepath)[:idx+len("8bit.ome")],
            "texture_type": os.path.basename(imagepath).split('_')[5][:-3],
            "concentration": os.path.basename(imagepath).split('_')[2],
            "type": os.path.basename(imagepath).split('_')[1],
            "roi": os.path.basename(imagepath).split('_')[3],
            "texture_mean": np.mean(currentim[currentim>0]),
            "texture_median": np.median(currentim[currentim>0]),
            "texture_std": np.std(currentim[currentim>0])
        }
        stackstats.append(imgstats)
    return stackstats

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

    m = re.search(r'Pos(\d+)', nospace_name)
    pos = int(m.group(1)) if m else None
    # Load masks
    masks = {}
    for mask_name, folder_path in mask_paths_dict.items():
        for fname in os.listdir(folder_path):

        # must match BOTH position and mask type
            if (f"Pos{pos}" in fname) and (mask_name in fname):

                full_path = os.path.join(folder_path, fname)

                mask_img = tiff.imread(full_path)
                mask_img = np.moveaxis(mask_img, 0, -1)

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
        for file in os.listdir(folder):
            if file.endswith((".tif",".tiff")):
                full = os.path.join(folder,file)
                stats = image_stats_glcm2D(full,stackstats)
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

def extract_stack_key(filename):
    idx = os.path.basename(filename).find("png")
    #name of stack here will be used to collapse slices
    if idx != -1:
        return os.path.basename(filename)[:idx + len("png")]
    return None

def twombli_slice_data(df):

    df = df.copy()

    df['image_name_raw'] = df['image_name'].apply(extract_stack_key)

    # extract metadata FIRST from full filename
    df['slice'] = df['image_name_raw'].str.extract(r'_s(\d+)\.png')[0].astype(int)
    df['timepoint'] = df['image_name_raw'].str.extract(r'_t0*(\d+)')[0].astype(int)
    df['position'] = df['image_name_raw'].str.extract(r'Pos(\d+)')[0].astype(int)

    df['mask_type'] = "full"

    # stable join key (DO NOT modify further)
    
    df['image_name'] = df['image_name_raw'].apply(lambda x: re.sub(r'_s\d+\.png$', '', x).replace('.png', ''))

    group_cols = ['image_name', 'slice', 'timepoint', 'position']

    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = numeric_cols.drop(['timepoint','position','slice'], errors='ignore')

    agg_df = df.groupby(group_cols, as_index=False)[numeric_cols].mean()

    print('TWOMBLI spreadsheet processed')
    return agg_df

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



##main

# for timeseries - generating one csv for each time point
# since easier to iterate through or just select timepoint without large files

dfCA = load_csv(str(PATHS["ctfire_results"] / "ctfire_stats_per_slice.csv"))
dfCAreorg = reshape_CA(dfCA)
dfTWOMBLI = load_csv(str(EXTERNAL["twombli_csv"]))
dfTWOMBLI = dfTWOMBLI.dropna(subset=["image_name"])
dfTWOMBLI = twombli_slice_data(dfTWOMBLI)
#print(dfTWOMBLI)

#dftexture = process_img_folder(str(PATHS["texture2d_ep"]), is_3d=0)
#dftexture= reshape_texture(dftexture)

mask_paths = {
    "a1_masked":   str(PATHS["masks"]),
    "a2_masked":   str(PATHS["masks"]),
    "cell_masked": str(PATHS["masks"]),
}

dftexture3D = process_img_folder(str(PATHS["texture3d"]), mask_paths, is_3d=1)
dftexture3D = reshape_texture(dftexture3D)


print(dfCAreorg["image_name"].nunique(), len(dfCAreorg))
print(dfTWOMBLI["image_name"].nunique(), len(dfTWOMBLI))
#print(dftexture["image_name"].nunique(), len(dftexture))
print(dftexture3D["image_name"].nunique(), len(dftexture3D))

csvdf = pd.merge(dfCAreorg, dfTWOMBLI, on=["image_name","slice", "position"], how="left")

#mostdf = pd.merge(csvdf, dftexture, on=["image_name","slice"], how="left")
#fulldf = pd.merge(mostdf,dftexture3D, on=["image_name","slice"], how = "left")
fulldf = pd.merge(csvdf,dftexture3D, on=["image_name","slice", "position"], how = "left")


print(fulldf.head())
print(fulldf.columns.values)


groups = find_identical_columns(fulldf)
if groups:
    print(f"Found {len(groups)} group(s) of identical columns:")
    for g in groups:
        print(" ", g)
    collapseddf = collapse_identical_columns(fulldf, groups)
    print(collapseddf.columns.values)
else:
    collapseddf = fulldf
    print("No exactly identical columns found.")


if any(col.startswith("n_") for col in collapseddf.columns):
    collapseddf = collapseddf.rename(columns={col: col.replace(col, "fibercount") for col in collapseddf.columns if col.startswith("n")})
if any(col.startswith("z_depth_") for col in collapseddf.columns):
    collapseddf = collapseddf.rename(columns={col: col.replace(col, "z_depth") for col in collapseddf.columns if col.startswith("z_depth")})

# Split into FLU and SHG dataframes if any type-like column exists
unique_pos = collapseddf['position'].unique()
for pos in unique_pos:
    pos_df = collapseddf[collapseddf['position'] == pos]
    
    pos_df.to_csv(f"current_final_dataframe_byslice_pos_{pos}_3D.csv", index=False)
    
    print("Saved position dataframes separately")

# Also save the combined dataframe
collapseddf.to_csv("finalcollapsed_dataframe_byslice.csv", index=False)