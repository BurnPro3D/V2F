"""
PURPOSE
-------
Take the generated shrub polygons (from generate_shrubs.py) and turn them
into a 3-D voxel grid of shrub bulk density and fuel depth that matches the
QUIC-Fire fire grid, then merge that shrub layer into the existing
tree-bulk-density (treesrhof.dat) and fuel-depth (treesfueldepth.dat) QUIC-
Fire input files -- producing "_adj" (adjusted) versions of both that
include shrubs.

Steps:
    1. Rasterize shrub polygons (height / presence / fine-fuel load) onto
       the fire grid.
    2. (Optional) If field-measured fuel load/depth for the first (ground)
       layer are available, splice them into the LANDFIRE-derived first
       layer before shrubs are added.
    3. Build a 3-D shrub bulk-density voxel grid from shrub height + fine-
       fuel density, one voxel layer at a time up to each shrub's height.
    4. Add the shrub voxel grid on top of the existing tree bulk density /
       fuel depth grids, re-apply the road mask, and write the merged
       "_adj.dat" files.
    5. Save the shrub voxel grid and topo grid as .npy for downstream use
       (e.g. plotting, validation).

Run top-to-bottom. Config comes from CLI args (sys.argv) -- see CONFIG below.
"""

import json
import math
import sys
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.transform import Affine
from scipy import ndimage
from scipy.io import FortranFile

# config
project_folder = sys.argv[1]   # folder containing data/quicfire_inputs.zip, data/generated_shrubs.geojson, treesrhof.dat, etc.
site_name = sys.argv[2] 
site = sys.argv[3]      

# Optional field-measured overrides (if enough data to run field_data_fuel_loading.py)
# If you have field-measured fuel load/depth for the ground (first) layer,
# pass their .dat paths here; otherwise pass "NA" to skip and keep the
# LANDFIRE/QUIC-Fire-derived first layer as-is.
field_fuel_load_dat = sys.argv[4]    # e.g. project_folder + "data/field_data_fuel_load.dat", or "NA"
field_fuel_depth_dat = sys.argv[5]   # e.g. project_folder + "data/field_data_fuel_depth.dat", or "NA"


# functions
def read_dat_file(filename, nz, ny, nx, order="C"):
    """
    Read in a .dat file as a numpy array.

    Parameters
    ----------
    filename : Path or str
    nz, ny, nx : int   Grid dimensions
    order : str         Array order (default "C")

    Returns
    -------
    ndarray  3D array with shape (nz, ny, nx)
    """
    if isinstance(filename, str):
        filename = (filename)

    with open(filename, "rb") as fin:
        arr = (
            FortranFile(fin)
            .read_reals(dtype="float32")
            .reshape((nz, ny, nx), order=order)
        )
    return arr


def write_np_array_to_dat(array, dat_name, output_dir, dtype=np.float32):
    """
    Write a numpy array to a fortran binary file. Array must be cast to the
    appropriate data type before calling this function.
    """
    with FortranFile(output_dir + dat_name, "w") as f:
        f.write_record(array)


def get_raster_values(raster_file, geom):
    """
    Compute the mean raster value under each polygon in `geom`.

    Parameters
    ----------
    raster_file : str               Path to a single-band raster (.tif)
    geom        : gpd.GeoDataFrame  Polygons to sample

    Returns
    -------
    list  Mean raster value per polygon (NaN-safe)
    """
    with rasterio.open(raster_file) as src:
        values = []
        for i in np.arange(geom.shape[0]):
            target_geom = [geom.geometry.iloc[i]]
            out_image, out_transform = mask(
                src,
                target_geom,
                crop=True,
                all_touched=True,
                nodata=-9999,
                filled=False,
            )
            values.append(np.nanmean(out_image[0, :, :]))
    return values


def raster_flip(raster):
    """Flip each z-layer of a 3D (nz, ny, nx) array vertically (north-up -> south-up)."""
    flipped = np.zeros(raster.shape)
    for i in np.arange(raster.shape[0]):
        flipped[i, :, :] = np.flipud(raster[i, :, :])
    return flipped


# load shrubs and metadata

shrubs = gpd.read_file(project_folder + "data/generated_shrubs.geojson")
shrubs["height"] = shrubs["z"]
shrubs["binary"] = 1

crs = 5070
shrubs = shrubs.to_crs(crs)

# Unpack the QUIC-Fire input zip to get the fire grid dimensions/transform
zip_path = project_folder + "quicfire_inputs.zip"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(project_folder + "data/")

with open(project_folder + "data/metadata.json", "r") as json_file:
    metadata = json.load(json_file)

nx = metadata["fire_grid"]["nx"]
ny = metadata["fire_grid"]["ny"]
nz = metadata["fire_grid"]["nz"]
transform = Affine(*metadata["fire_grid"]["transform"])


# rasterize shrubs
with rasterio.open(
    project_folder + "shrubs.tif", "w+",
    driver="GTiff", height=ny, width=nx, count=1,
    dtype=np.float64, crs=5070, transform=transform,
) as out:
    out_arr = out.read(1)
    shapes = ((geom, value) for geom, value in zip(shrubs.geometry, shrubs.height))
    shrub_rast = features.rasterize(
        shapes=shapes, fill=0.0001, out=out_arr, transform=out.transform, all_touched=True
    )
    out.write_band(1, shrub_rast)


# load existing fuel load (from LANDFIRE) or field-measured fuel load (if available)

fuel_load_dat = read_dat_file(project_folder + "data/treesrhof.dat", nz, ny, nx, order="C")


if field_fuel_load_dat != "NA":
    # Field-measured fuel load only covers the ground layer
    fuel_load_first_layer = read_dat_file(project_folder + field_fuel_load_dat, 1, ny , nx, order="C")
    fuel_load_dat[0, :, :] = fuel_load_first_layer




# get fine fuel load for shrubs if not available

if "fines_load" not in shrubs.columns:
    # Save the ground-layer fuel load dat as a raster, then sample it onto
    # each shrub polygon to get a per-shrub fine-fuel load
    with rasterio.open(
        project_folder + "data/landfire_fuel_load.tif", "w+",
        driver="GTiff", height=ny, width=nx, count=1,
        dtype=np.float64, crs=5070, transform=transform,
    ) as out:
        out.write_band(1, np.flipud(fuel_load_dat[0, :, :]))

    shrub_load_landfire = get_raster_values(project_folder + "data/landfire_fuel_load.tif", shrubs)
    shrubs["fines_load"] = shrub_load_landfire


# rasterize shrub height, presence, and fine-fuel load onto the fire grid

shapes = ((geom, value) for geom, value in zip(shrubs.geometry, shrubs.height))
shrub_rast_height = features.rasterize(
    shapes=shapes, fill=0.0001, out_shape=out_arr.shape, transform=transform, all_touched=True
)

shapes = ((geom, value) for geom, value in zip(shrubs.geometry, shrubs.binary))
shrub_rast_binary = features.rasterize(
    shapes=shapes, fill=0.0001, out_shape=out_arr.shape, transform=transform, all_touched=True
)

shapes = ((geom, value) for geom, value in zip(shrubs.geometry, shrubs.fines_load))
shrub_fuel_density = features.rasterize(
    shapes=shapes, fill=0.0001, out_shape=out_arr.shape, transform=transform, all_touched=True
)


shrub_ht = shrub_rast_height 

# build 3D shrub voxels

max_shrub_ht = np.max(shrub_ht)
shrub_load_3d_array = np.zeros((shrub_ht.shape[0], shrub_ht.shape[1], math.ceil(max_shrub_ht)))

for i in np.arange(shrub_ht.shape[0]):
    for j in np.arange(shrub_ht.shape[1]):
        temp_ht = shrub_ht[i, j]

        if temp_ht == 0:
            continue

        shrub_density = shrub_fuel_density[i, j]

        if temp_ht <= 1:
            # partial (or exactly full) single voxel
            shrub_load_3d_array[i, j, 0] = shrub_density * temp_ht
            continue

        cell_ht = int(math.floor(temp_ht))
        remaining_ht = temp_ht - cell_ht

        for k in np.arange(cell_ht):
            shrub_load_3d_array[i, j, k] = shrub_density

        if remaining_ht != 0:
            shrub_load_3d_array[i, j, cell_ht] = shrub_density * abs(remaining_ht)


# shrub grid to fire grid

fuel_load_dat_lf = read_dat_file(project_folder + "data/treesrhof.dat", nz, ny, nx, order="C")
road_mask = fuel_load_dat_lf == 0
road_mask = road_mask[0, :, :]

new_shrub_load_3d_array = np.transpose(shrub_load_3d_array, (2, 0, 1))
new_shrub_load_3d_array = raster_flip(new_shrub_load_3d_array)

# Save shrub voxel grid and topo grid for downstream use (plotting, validation)
topo = read_dat_file(project_folder + "data/topo.dat", 1, ny, nx, order="C")
np.save(project_folder + f"data/{site}_topo.npy", topo)
np.save(project_folder + f"data/{site}_shrubs.npy", new_shrub_load_3d_array)


# add shrub density to existing fuel density

shrub_load_adj = np.zeros(fuel_load_dat.shape)
shrub_voxels = new_shrub_load_3d_array.shape[0]

# Add shrub density into the layers the shrub grid actually covers, and keep tree-only density unchanged above that
shrub_load_adj[0:shrub_voxels, :, :] = new_shrub_load_3d_array + fuel_load_dat[0:shrub_voxels, :, :]
shrub_load_adj[shrub_voxels:, :, :] = fuel_load_dat[shrub_voxels:, :, :]

# Zero out fuel load on roads in the ground layer
first_layer = shrub_load_adj[0, :, :]
first_layer[road_mask] = 0
shrub_load_adj[0, :, :] = first_layer

write_np_array_to_dat(
    shrub_load_adj.astype("float32"), "treesrhof_adj.dat", project_folder + "data/", dtype=np.float32
)


# merge shrub height into fuel depth

fuel_depth = read_dat_file(project_folder + "data/treesfueldepth.dat", nz, ny, nx, order="C")

if field_fuel_depth_dat != "NA":
    fuel_depth_first_layer = read_dat_file(project_folder + field_fuel_depth_dat, 1, ny , nx , order="C")
    fuel_depth[0, :, :] = fuel_depth_first_layer

shrub_height_pad = np.zeros(fuel_load_dat.shape)
new_shrub_rast_height = np.flipud(shrub_rast_height)
shrub_height_pad[0, :, :] = new_shrub_rast_height + fuel_depth[0, :, :]

# Zero out fuel depth on roads in the ground layer
first_layer = shrub_height_pad[0, :, :]
first_layer[road_mask] = 0
shrub_height_pad[0, :, :] = first_layer

write_np_array_to_dat(
    shrub_height_pad.astype("float32"), "treesfueldepth_adj.dat", project_folder + "data/", dtype=np.float32
)

