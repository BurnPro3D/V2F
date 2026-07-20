import rasterio
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from rasterio.transform import Affine
import matplotlib.cm as cm
import copy
import sys

maindir = sys.argv[1] 
target_resolution = int(sys.argv[2])  # e.g., 4 for 4m resolution
chm_file = maindir + 'data/python_chm.tif'
chm_src = rasterio.open(chm_file)
chm = chm_src.read(1)

# calculate the native resolution of the CHM
native_resolution = chm_src.res[0]
block_size = int(target_resolution / native_resolution)
pixels_per_block = block_size ** 2

# Create a masked array where the placeholder value is masked
masked_array = np.ma.masked_where(chm < 1, chm)

# build a binary (0/1) canopy mask from CHM
canopy_yes_no = copy.deepcopy(chm)
canopy_yes_no[canopy_yes_no < 1 ] = 0
canopy_yes_no[canopy_yes_no >= 1 ] = 1

#calculate the lacunarity of each block

def calculate_lacunarity(binary_image, box_sizes):
    """
    Calculates lacunarity for a binary image using the gliding-box algorithm. 
    Lacunarity measures 'gappiness' or heterogeneity of a pattern. 
    Higher values indicate more gaps or heterogeneity.
    Values close to 1 indicate a more homogeneous pattern.
    """
    lacunarities = []
    
    # Ensure binary_image is a NumPy array
    img = np.array(binary_image)
    
    for r in box_sizes:
        masses = []
        # Glide a box of size r x r over the image
        for i in range(img.shape[0] - r + 1):
            for j in range(img.shape[1] - r + 1):
                box = img[i:i+r, j:j+r]
                mass = np.sum(box)
                masses.append(mass)
        
        # Calculate moments for this box size
        if len(masses) > 0:
            mean_mass = np.mean(masses)
            mean_sq_mass = np.mean(np.array(masses)**2)
            
            # Calculate lacunarity
            if mean_mass > 0:
                lacunarity_val = mean_sq_mass / (mean_mass**2)
                lacunarities.append(lacunarity_val)
            else:
                # Handle cases with zero mass to avoid division by zero
                lacunarities.append(np.nan)
        else:
            lacunarities.append(np.nan)
            
    return lacunarities


#downsample binary canopy mask 
downsampled_canopy = block_reduce(canopy_yes_no, block_size=(block_size, block_size), func=(np.nansum))

tif = rasterio.open(chm_file)
meta = tif.meta
meta['width'] = downsampled_canopy.shape[1]
meta['height'] = downsampled_canopy.shape[0]
meta['transform'] = tif.transform * Affine.scale(target_resolution, target_resolution)

with rasterio.open(maindir + 'data/canopy_cover.tif', 'w',**meta) as dest:
    dest.write(downsampled_canopy/16, 1)

# Define the step size
step = 8
lacunarity = []

# Calculate starting indices
row_starts = range(0, canopy_yes_no.shape[0], step)
col_starts = range(0, canopy_yes_no.shape[1], step)

# Extract subarrays
subarrays = []
for i in row_starts:
    for j in col_starts:
        subarray = canopy_yes_no[i:i+step, j:j+step]
        lacunarity.append(calculate_lacunarity(subarray, [4])[0])

lac = np.reshape(np.array(lacunarity), (len(row_starts), len(col_starts)))
lac[lac > np.nanquantile(lac, 0.99)] = np.nan

tif = rasterio.open(chm_file)
meta = tif.meta
meta['width'] = lac.shape[1]
meta['height'] = lac.shape[0]
meta['transform'] = tif.transform * Affine.scale(8.0, 8.0)

with rasterio.open(maindir + 'data/lacunarity.tif', 'w',**meta) as dest:
    dest.write(lac, 1)
