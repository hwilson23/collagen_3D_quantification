import os
import numpy as np
import pandas as pd
import tifffile as tiff


def load_csv(file, columns=None):
    df = pd.read_csv(file)
    if columns:
        df = df[columns]
    return df 

def reshape_CA(df):
    pivotdf = df.pivot_table(
        index="image_name",
        columns = "hist_type",
        values = ["mean","total_fibers","std","median"]
    )
    pivotdf.columns = [f"{stat}_{hist}" for stat,hist in pivotdf.columns]
    pivotdf = pivotdf.reset_index()

    return pivotdf

def reshape_texture(df):
    # List of all possible value columns
    possible_values = ["concentration", "type", "roi", "texture_mean", "texture_median", "texture_std"]
    
    # Filter to only include columns that exist in the dataframe
    available_values = [col for col in possible_values if col in df.columns]
    
    if not available_values:
        # If no values are available, just return the dataframe as-is
        return df
    
    pivotdf = df.pivot(
        index="image_name",
        columns = "texture_type",
        values = available_values
    )
    pivotdf.columns = [f"{stat}_{tex}" for stat,tex in pivotdf.columns]
    pivotdf = pivotdf.reset_index()

    return pivotdf

def image_stats(imagepath):
    img = tiff.imread(imagepath)


    #index of end filename
    idx = os.path.basename(imagepath).find("8bit")
    print(os.path.basename(imagepath)[:idx+8])

    imgstats = {
        "image_name": os.path.basename(imagepath)[:idx+len("8bit.ome")],
        "texture_type": os.path.basename(imagepath).split('_')[5][:-3],
        "concentration": os.path.basename(imagepath).split('_')[2],
        "type": os.path.basename(imagepath).split('_')[1],
        "roi": os.path.basename(imagepath).split('_')[3],
        "texture_mean": np.mean(img[img>0]),
        "texture_median": np.median(img[img>0]),
        "texture_std": np.std(img[img>0])
    }
    return imgstats

def process_img_folder(folder):
    results = []

    for file in os.listdir(folder):
        if file.endswith((".tif",".tiff")):
            full = os.path.join(folder,file)
            stats = image_stats(full)
            results.append(stats)
            #print(results)
    return pd.DataFrame(results)

def extract_stack_key(filename):
    idx = os.path.basename(filename).find("ome")
    #name of stack here will be used to collapse slices
    if idx != -1:
        return os.path.basename(filename)[:idx + len("ome")]
    return None

##main

dfCA = load_csv("G:/FluorescentCollagen/20260427_flucol_ows3/flucol_crops/ctFIREout/results_test_masked_min30/ctfire_stats_stack_summary.csv")
dfCAreorg = reshape_CA(dfCA)
dfTWOMBLI = load_csv("C:/Users/hwilson23/Desktop/TWOMBLI-master/TWOMBLI_v1/Twombli_Results_concentration_shgandflu.csv")
dftexture = process_img_folder("G:/FluorescentCollagen/20260427_flucol_ows3/20260427_texturemapdata/texturemap")
dftexture= reshape_texture(dftexture)

#fix twombli slice data to means
df_noslices = dfTWOMBLI
df_noslices["stack_key"] = df_noslices["image_name"].apply(extract_stack_key)
df_noslices = df_noslices.groupby("stack_key").mean(numeric_only=True).reset_index()

df_noslices.rename(columns={"stack_key":"image_name"}, inplace=True)


print(dfCAreorg["image_name"].nunique(), len(dfCAreorg))
print(dfTWOMBLI["image_name"].nunique(), len(dfTWOMBLI))
print(df_noslices["image_name"].nunique(), len(df_noslices))
print(dftexture["image_name"].nunique(), len(dftexture))
csvdf = pd.merge(dfCAreorg, df_noslices, on="image_name", how="left")

fulldf = pd.merge(csvdf, dftexture, on="image_name", how="left")


print(fulldf)

fulldf.to_csv("final_dataframe.csv", index=False)