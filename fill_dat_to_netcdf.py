import numpy as np
import json
import sys
from pathlib import Path
from scipy.io import FortranFile
from affine import Affine
import xarray as xr
import rioxarray  #(registers the .rio accessor on xarray objects)

project_folder = sys.argv[1]
site = sys.argv[2]


def read_dat_file(filename, nz, ny, nx, order="C"):
    """
    Read in a .dat file as a numpy array.

    Parameters
    ----------
    filename : Path or str
        The path to the .dat file to read.
    nz : int
        The number of cells in the z-direction.
    ny : int
        The number of cells in the y-direction.
    nx : int
        The number of cells in the x-direction.
    order : str, optional
        The order of the array. Default is "C".

    Returns
    -------
    ndarray
        A 3D numpy array representing the data in the .dat file. The array
        has dimensions (nz, ny, nx).
    """

    if isinstance(filename, str):
        filename = Path(filename)

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
        f.write_record(array.astype(dtype).flatten(order="C"))


def save_netcdf(transform, nx, ny, nz, arr, netcdf_savename):
    """
    Save a (z, y, x) numpy array as a georeferenced NetCDF file, using an
    affine transform to build the x/y coordinate arrays and attaching the
    appropriate CRS.
    """
    aff = Affine(*transform)

    # Build coordinate arrays for each grid cell.
    cols = np.arange(nx)
    rows = np.arange(ny)

    xs, _ = aff * (cols, np.zeros(nx))  # x coords (only col varies)
    _, ys = aff * (np.zeros(ny), rows)  # y coords (only row varies)

    # Shift coordinates to represent cell centers rather than cell corners.
    xs = xs + aff.a / 2
    ys = ys + aff.e / 2

    da = xr.DataArray(
        arr,
        dims=("z", "y", "x"),
        coords={"z": np.arange(1, nz + 1), "y": ys, "x": xs},
        name="fuel_load",
    )

    # Attach the affine transform and CRS so the NetCDF is georeferenced.
    da = da.rio.write_transform(aff)
    da.rio.write_crs("EPSG:5070", inplace=True)

    da.to_netcdf(netcdf_savename)


# Load grid dimensions and affine transform from metadata.
with open(project_folder + "data/metadata.json", "r") as json_file:
    metadata = json.load(json_file)

nx = metadata["fire_grid"]["nx"]
ny = metadata["fire_grid"]["ny"]
nz = metadata["fire_grid"]["nz"]

lowerx = metadata["fire_grid"]["transform"][2]
lowery = metadata["fire_grid"]["transform"][5]

transform = metadata["fire_grid"]["transform"]

# Read raw .dat arrays.
fuel_load_dat_adj = read_dat_file(project_folder + "data/treesrhof_adj.dat", nz, ny, nx, order="C")
fuel_depth_dat_adj = read_dat_file(project_folder + "data/treesfueldepth_adj.dat", nz, ny, nx, order="C")
topo = read_dat_file(project_folder + "data/topo.dat", 1, ny, nx, order="C")
treesmoist = read_dat_file(project_folder + "data/treesmoist.dat", nz, ny, nx, order="C")
sav = read_dat_file(project_folder + "data/treesss.dat", nz, ny, nx, order="C")

# Note: arrays are kept in their raw (as-read) row order. Orientation is handled at plot time (e.g. imshow(..., origin='lower')) 

# Replace 0s with a sentinel value (1.23456) in all layers EXCEPT the surface layer (index 0)

first_layer_load = fuel_load_dat_adj[0, :, :]
fuel_load_dat_adj[fuel_load_dat_adj == 0] = 1.23456
fuel_load_dat_adj[0, :, :] = first_layer_load

first_layer_depth = fuel_depth_dat_adj[0, :, :]
fuel_depth_dat_adj[fuel_depth_dat_adj == 0] = 1.23456
fuel_depth_dat_adj[0, :, :] = first_layer_depth

# Save each array as a georeferenced NetCDF file.
save_netcdf(transform, nx, ny, nz, fuel_load_dat_adj, project_folder + "data/treesrhof_adj.nc")
save_netcdf(transform, nx, ny, nz, fuel_depth_dat_adj, project_folder + "data/treesrfueldepth_adj.nc")
save_netcdf(transform, nx, ny, 1, topo, project_folder + "data/topo.nc")
save_netcdf(transform, nx, ny, nz, treesmoist, project_folder + "data/treesmoist.nc")
save_netcdf(transform, nx, ny, nz, sav, project_folder + "data/treesss.nc")

shrubs  = np.load(project_folder + f"data/{site}_shrubs.npy")

save_netcdf(transform, nx, ny, shrubs.shape[0], shrubs, project_folder + "data/shrubs.nc")

