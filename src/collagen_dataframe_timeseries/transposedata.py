
import os
import numpy as np
import tifffile as tiff

#input_dir = r"G:\FluorescentCollagen\20260519_flucol_kpc_ows3\selectedpos\maskapplied_timeseries_stacks"
input_dir = r"G:\FluorescentCollagen\20260519_flucol_kpc_ows3\selectedpos\A1A2endpointmaskapplied"

filelist = [file for file in os.listdir(input_dir)]
print(f"Found {len(filelist)} files to process.")

for files in filelist:
    fname = os.path.basename(files)
    fpath = os.path.join(input_dir, files)
    img = tiff.imread(fpath)
    img = np.moveaxis(img,-1,0)
    print(img.shape)
    tiff.imwrite(fpath, img)