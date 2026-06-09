
import os
import sys
from pathlib import Path
import numpy as np
import tifffile as tiff
import re

sys.path.insert(0, str(Path(__file__).parents[2]))
from config import PATHS

# Directory paths
image_dir  = str(PATHS["masked_stacks"])
mask_dir   = str(PATHS["masks"])
output_dir = str(PATHS["masked_output"])

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def extract_position(filename):
    """Extract position identifier from filename (e.g., 'Pos0', 'Pos1')"""
    match = re.search(r'Pos\d+', filename)
    return match.group(0) if match else None

def find_mask(mask_dir, position, mask_type):
    """Find mask file matching position and mask type (cell, a1, a2)"""
    for file in os.listdir(mask_dir):
        if position in file and mask_type.lower() in file.lower():
            return os.path.join(mask_dir, file)
    return None

def apply_hierarchical_masks(image, cell_mask, a1_mask, a2_mask):
    """
    Apply masks separately and return three masked images:
    - cell_only: image with cell mask applied
    - a1_only: image with A1 mask applied (minus cell area)
    - a2_only: image with A2 mask applied (minus A1 and cell area)
    """
    # Normalize masks to 0/1
    if cell_mask is not None:
        cell_mask = (cell_mask > 0).astype(np.uint8)
    if a1_mask is not None:
        a1_mask = (a1_mask > 0).astype(np.uint8)
    if a2_mask is not None:
        a2_mask = (a2_mask > 0).astype(np.uint8)
    
    results = {}
    
    # Cell mask: remove cell area
    if cell_mask is not None:
        cell_region = 1-cell_mask
        results['cell'] = image * cell_region
    else:
        results['cell'] = None
    
    # A1 mask: keep A1 area minus cell
    if a1_mask is not None:
        a1_region = a1_mask - (cell_mask if cell_mask is not None else 0)
        results['a1'] = image * a1_region
    else:
        results['a1'] = None
    
    # A2 mask: keep A2 area minus A1 and cell
    if a2_mask is not None:
        exclude = (cell_mask if cell_mask is not None else 0) + (a1_mask if a1_mask is not None else 0)
        exclude = np.clip(exclude, 0, 1)
        a2_region = a2_mask * (1 - exclude)
        results['a2'] = image * a2_region
    else:
        results['a2'] = None
    
    return results

# Get all image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff'))]
print(f"Found {len(image_files)} image files to process.")

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    position = extract_position(image_file)
    
    if position is None:
        print(f"Skipping {image_file}: could not extract position")
        continue
    
    print(f"\nProcessing {image_file} (Position: {position})")
    
    # Load image
    img = tiff.imread(image_path)
    print(f"  Image shape: {img.shape}")
    
    # Find and load masks for this position
    cell_mask_path = find_mask(mask_dir, position, "cell")
    a1_mask_path = find_mask(mask_dir, position, "a1")
    a2_mask_path = find_mask(mask_dir, position, "a2")
    
    cell_mask = tiff.imread(cell_mask_path) if cell_mask_path else None
    a1_mask = tiff.imread(a1_mask_path) if a1_mask_path else None
    a2_mask = tiff.imread(a2_mask_path) if a2_mask_path else None
    
    if cell_mask is not None:
        print(f"  Cell mask found (shape: {cell_mask.shape})")
    if a1_mask is not None:
        print(f"  A1 mask found (shape: {a1_mask.shape})")
    if a2_mask is not None:
        print(f"  A2 mask found (shape: {a2_mask.shape})")
    
    # Apply hierarchical masks (returns dict with three separate masked images)
    masked_images = apply_hierarchical_masks(img, cell_mask, a1_mask, a2_mask)
    
    # Save each masked image separately
    if masked_images['cell'] is not None:
        output_file = os.path.join(output_dir, "cell_masked_" + image_file)
        tiff.imwrite(output_file, masked_images['cell'])
        print(f"  Saved cell mask to: {output_file}")
    
    if masked_images['a1'] is not None:
        output_file = os.path.join(output_dir, "a1_masked_" + image_file)
        tiff.imwrite(output_file, masked_images['a1'])
        print(f"  Saved A1 mask to: {output_file}")
    
    if masked_images['a2'] is not None:
        output_file = os.path.join(output_dir, "a2_masked_" + image_file)
        tiff.imwrite(output_file, masked_images['a2'])
        print(f"  Saved A2 mask to: {output_file}")
