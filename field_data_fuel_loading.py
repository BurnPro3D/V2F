import ee
import pandas as pd
import numpy as np
import geopandas as gpd

import copy
import geojson
import datetime
import rasterio
from skimage.measure import block_reduce
import sys
from scipy.io import FortranFile

import cv2
import json

# for the field data
from pykrige.uk import UniversalKriging
from pykrige.compat import train_test_split  # For splitting data
from scipy.interpolate import griddata

import os

project_dir = sys.argv[1]

plot_dir = sys.argv[2]
site_name = sys.argv[3]
maindir = project_dir

ee_project = sys.argv[4]

field_fuel_load_dat = sys.argv[5]
field_fuel_depth_dat = sys.argv[6]


ee.Authenticate()
ee.Initialize(project=ee_project)


def calculate_slope_manual(dtm_array, cell_size):
    # Calculate gradients in x and y directions
    dz_dx = np.gradient(dtm_array, axis=1) / cell_size
    dz_dy = np.gradient(dtm_array, axis=0) / cell_size

    # Calculate slope in radians
    slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # Convert to degrees
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees


def calculate_aspect(dem_array, cell_size):
    """
    Compute terrain aspect in degrees (0-360, North = 0) using Horn's method.

    Parameters
    ----------
    dem_array : np.ndarray  2-D elevation raster
    cell_size : float       Pixel resolution in map units

    Returns
    -------
    np.ndarray  Aspect in degrees (same shape as dem_array); edges = NaN
    """
    padded = np.pad(dem_array.astype(float), 1, mode="edge")

    # Horn's weighted finite differences
    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:]) -
        (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8 * cell_size)

    dz_dy = (
        (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) -
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
    ) / (8 * cell_size)

    flat = (dz_dx == 0) & (dz_dy == 0)
    aspect_deg = np.degrees(np.arctan2(dz_dy, -dz_dx))
    aspect_deg = 90 - aspect_deg
    aspect_deg[aspect_deg < 0] += 360
    aspect_deg[flat] = np.nan

    # Keep a 1-pixel NaN border to match the loop version's behaviour
    aspect_out = aspect_deg.copy()
    aspect_out[0, :] = np.nan
    aspect_out[-1, :] = np.nan
    aspect_out[:, 0] = np.nan
    aspect_out[:, -1] = np.nan
    return aspect_out



def create_reduce_region_function(geometry,
                                   reducer=ee.Reducer.mean(),
                                   scale=1000,
                                   crs='EPSG:4326',
                                   bestEffort=True,
                                   maxPixels=1e13,
                                   tileScale=16):
    """Creates a region reduction function (see ee.Image.reduceRegion() docs)."""

    def reduce_region_function(img):
        stat = img.reduceRegion(
            reducer=reducer,
            geometry=geometry,
            scale=scale,
            crs=crs,
            bestEffort=bestEffort,
            maxPixels=maxPixels,
            tileScale=tileScale)

        return ee.Feature(geometry, stat).set({'millis': img.date().millis()})
    return reduce_region_function


def fc_to_dict(fc):
    """Transfer feature properties from an ee.FeatureCollection into an ee.Dictionary."""
    prop_names = fc.first().propertyNames()
    prop_lists = fc.reduceColumns(
        reducer=ee.Reducer.toList().repeat(prop_names.size()),
        selectors=prop_names).get('list')

    return ee.Dictionary.fromLists(prop_names, prop_lists)


def add_date_info(df):
    """Add human-readable date component columns to a DataFrame with a 'millis' column."""
    df['Timestamp'] = pd.to_datetime(df['millis'], unit='ms')
    df['Year'] = pd.DatetimeIndex(df['Timestamp']).year
    df['Month'] = pd.DatetimeIndex(df['Timestamp']).month
    df['Day'] = pd.DatetimeIndex(df['Timestamp']).day
    df['DOY'] = pd.DatetimeIndex(df['Timestamp']).dayofyear
    return df

def getInfo_with_retry(ee_object, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return ee_object.getInfo()
        except ee.ee_exception.EEException as e:
            if attempt == retries - 1:
                raise
            print(f"getInfo failed ({e}), retrying in {delay}s...")
            datetime.time.sleep(delay)

            
def get_image_as_table(collection_name, date_range, variable, aoi, proj='EPSG:4326'):
    """Pull a per-image regional statistic (mean over `aoi`) for an EE ImageCollection into a DataFrame."""
    ic = ee.ImageCollection(collection_name).filterDate(date_range).select(variable)

    reduce_ic = create_reduce_region_function(
        geometry=aoi, reducer=ee.Reducer.mean(), scale=1000, crs=proj)

    stat_fc = ee.FeatureCollection(ic.map(reduce_ic)).filter(
        ee.Filter.notNull(ic.first().bandNames()))

    ic_dict = getInfo_with_retry(fc_to_dict(stat_fc))
    ic_df = pd.DataFrame(ic_dict)
    ic_df = add_date_info(ic_df)
    return ic_df


def read_dat_file(filename, nz, ny, nx, order="C"):
    """
    Read in a .dat file as a numpy array.

    Parameters
    ----------
    filename : Path or str
    nz, ny, nx : int
        Number of cells in the z-, y-, and x-directions.
    order : str, optional
        Default "C".

    Returns
    -------
    ndarray of shape (nz, ny, nx)
    """

    with open(filename, "rb") as fin:
        arr = (
            FortranFile(fin)
            .read_reals(dtype="float32")
            .reshape((nz, ny, nx), order=order)
        )

    return arr


boundary_file = project_dir + "data/ff_domain.geojson"

# Read with geopandas so we can see/verify the actual CRS, rather than assuming
boundary_gdf = gpd.read_file(boundary_file)
print("Original CRS:", boundary_gdf.crs)

# Reproject to EPSG:4326 if it isn't already
boundary_gdf_4326 = boundary_gdf.to_crs('EPSG:4326')

# Convert back to a plain geojson dict/FeatureCollection for Earth Engine
gjson = json.loads(boundary_gdf_4326.to_json())
geojson_fc = ee.FeatureCollection(gjson)

today = ee.Date(pd.to_datetime('today'))
date_range = ee.DateRange(today.advance(-20, 'years'), today)
aoi = geojson_fc.geometry()

ndvi_df = get_image_as_table("MODIS/061/MOD13A2", date_range, 'NDVI', aoi, proj='EPSG:4326')
ndvi_df['NDVI'] = ndvi_df['NDVI'] / 10000   # MOD13A2 NDVI is scaled by 10000 in the raw product


avg_ndvi = ndvi_df.groupby(['DOY'])['NDVI'].mean().reset_index()

# smooth out ndvi graph
new_ndvi = np.convolve(avg_ndvi.NDVI, np.ones(3)/3, mode='same')
avg_ndvi.NDVI = new_ndvi
additional_data = avg_ndvi[0:3]

# interpolate NDVI values for the whole year
x_values = list(avg_ndvi.DOY)
y_values = list(avg_ndvi.NDVI)

x_values.extend(list(additional_data.DOY + 365))
y_values.extend(list(additional_data.NDVI))

interp_values = [np.nan for i in range(366)]

for i in np.arange(len(x_values) - 1):
    m = (y_values[i+1] - y_values[i]) / (x_values[i+1] - x_values[i])
    b = y_values[i] - (m * x_values[i])

    for d in np.arange(x_values[i], x_values[i+1]):
        if d < 366:
            interp_values[d-1] = (m * d) + b


# figure out where the canopy loss should be based on inflection point of interpolated values
dy = np.diff(interp_values)
idx_max_dy = np.argmax(dy)

maxndvi = np.max(avg_ndvi.NDVI)
minndvi = np.min(avg_ndvi.NDVI)

percentage_canopy_loss = 1 - ((np.array(interp_values) - minndvi) / (maxndvi - minndvi))

infl = copy.deepcopy(dy)
infl[np.where(dy < 0)] = 0
infl[np.where(dy > 0)] = 1
infl = np.diff(infl)

foliage_growth = np.where(infl == -1)[0]
# assumes at least 2 sign-change points exist in `infl` 

percentage_canopy_loss[:np.where(infl != 0)[0][-2]] = 0

interp_ndvi = pd.DataFrame({
    'DOY': np.arange(1, 367),
    'NDVI': interp_values,
    'pct_canopy_loss': percentage_canopy_loss
})


# calculate the number of years since last burn

yearburned = []
firstburn = []
for y in np.arange(-1, -20, -1):
    date_range_burn = ee.DateRange(today.advance(int(y - 1), 'years'), today.advance(int(y), 'years'))

    firecci = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(date_range_burn).select('BurnDate')

    reduce_firecci = create_reduce_region_function(
        geometry=aoi, reducer=ee.Reducer.mean(), scale=1000, crs='EPSG:4326')

    firecci_stat_fc = ee.FeatureCollection(firecci.map(reduce_firecci)).filter(
        ee.Filter.notNull(firecci.first().bandNames()))

    if len(firecci_stat_fc.getInfo()['features']) != 0:

        firecci_dict = fc_to_dict(firecci_stat_fc).getInfo()
        firecci_df = pd.DataFrame(firecci_dict)

        firecci_df = add_date_info(firecci_df)
        firecci_df.BurnDate

        yearburned.append(np.unique(firecci_df.Year)[0])
        firstburn.append(np.min(firecci_df.BurnDate))


year = datetime.date.today().year
if len(yearburned) != 0:
    years_since_fire = year - np.max(yearburned)
else:
    years_since_fire = 20


with open(project_dir + 'data/metadata.json', 'r') as json_file:
    metadata = json.load(json_file)
nx = metadata['fire_grid']['nx']
ny = metadata['fire_grid']['ny']
nz = metadata['fire_grid']['nz']

tree_density = read_dat_file(project_dir + 'data/treesrhof.dat', nz, ny, nx, order="C")

reversed_voxels = tree_density[1:-1, :, :]
heightmap = np.argmax(reversed_voxels, axis=0)

annual_biomass = np.sum(tree_density, axis=0)


# use olson model to determine the litter left since last fire
# need to use aet (annual evapotranspiration) calculated from MODIS AET Product to find the decomposition rate

aet_df = get_image_as_table("IDAHO_EPSCOR/TERRACLIMATE", date_range, 'aet', aoi, proj='EPSG:4326')
avg_annual_et = aet_df.groupby(['Year']).mean('ET').reset_index()

aet = avg_annual_et.aet

# get the proportion of leaf fall every year: for example, coniferous would be 0.3, deciduous would be 1
falling_leaves = 0.3


# k = decomposition rate /yr
k = 0.0139 + 0.0003 * aet
t = years_since_fire
B0 = 0.04  # amount of litter at time 0 aka amount of litter not removed since last fire

litter = annual_biomass * falling_leaves * years_since_fire

n_years_with_data = len(k)

biomass = 0
for y in np.arange(t):
    k_y = k[y] if y < n_years_with_data else k[n_years_with_data - 1]
    # FIXED: correct Olson formula -- B0 term added on its own instead of scaled by litter/k[y].
    biomass += (litter / k_y) * (1 - np.exp(-k_y * t)) + B0 * np.exp(-k_y * t)

# blur the biomass a bit
biomass = cv2.GaussianBlur(biomass, (5, 5), 2)


lf_surface_loading = tree_density[0, :, :]
lf_surface_depth = read_dat_file(project_dir + 'data/treesfueldepth.dat', nz, ny, nx, order="C")[0, :, :]

# biomass to depth
biomass_kg = biomass / 10000

vol = 1 / 10000
depth = biomass * vol

new_fuel_load_kg = np.where(biomass_kg == 0, lf_surface_loading, biomass_kg + lf_surface_loading)
new_fuel_depth_m = np.where(depth == 0, lf_surface_depth, depth + lf_surface_depth)


def extract_raster_vals(raster_path, points_gdf):
    """Sample a single-band raster at each point in a GeoDataFrame (assumes matching CRS)."""
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)

        extracted_values = []

        for i, row in points_gdf.iterrows():
            point_geom = row['geometry']
            x, y = point_geom.x, point_geom.y

            row_idx, col_idx = src.index(x, y)

            if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                pixel_value = raster_data[row_idx, col_idx]
                extracted_values.append(pixel_value)
            else:
                extracted_values.append(None)

    return extracted_values


tree = '04_tree.csv'
field_trees = pd.read_csv(plot_dir + tree)
plots_df = pd.read_csv(plot_dir + '01_plot_identification.csv')

fuels_df = pd.read_csv(plot_dir + 'plot_fueldepth.csv')
fuel_load_df = pd.read_csv(plot_dir + 'prefire_plot_fuel_loads.csv')

plots_df = plots_df[plots_df.site_name == site_name]
plots_df = plots_df[plots_df.inventory_pre_post_fire_label == 'Prefire']

field_fuel = fuels_df.merge(plots_df, on='inventory_id')
field_fuel = field_fuel.merge(fuel_load_df, on='inventory_id')

boundary = gpd.read_file(project_dir + 'data/ff_domain.geojson')
burn_unit = gpd.read_file(project_dir + 'data/boundary.geojson')


field_fuel_gdf = gpd.GeoDataFrame(
    field_fuel,
    geometry=gpd.points_from_xy(field_fuel.plot_coord_x, field_fuel.plot_coord_y),
    crs=np.unique(field_fuel.plot_coord_srs)[0]
)
boundary = boundary.to_crs(field_fuel_gdf.crs)
field_fuel_gdf = field_fuel_gdf.sjoin(boundary)

crs = 5070
field_fuel_gdf = field_fuel_gdf.to_crs(crs)

dtm = rasterio.open(project_dir + 'data/dtm.tif')
# calculate and write raster of slope and aspect
file_path = project_dir + 'data/slope.tif'


# calculate and write raster of slope and aspect
slope_path = project_dir + 'data/slope.tif'
aspect_path = project_dir + 'data/aspect.tif'
dtm_path = project_dir + 'data/dtm_filled.tif'

if (os.path.exists(slope_path) & os.path.exists(aspect_path) & os.path.exists(dtm_path)):
    print(f"The files exist.")

else:
    print('Load the DTM')
    with rasterio.open(project_dir + 'data/dtm.tif') as src:
        dtm_array = src.read(1)
        cell_size = src.res[0]

    print('fix dtm')
    dtm_array[dtm_array == -9999] = np.nan

    image_with_nan = dtm_array
    valid_coords = np.argwhere(~np.isnan(image_with_nan))
    valid_values = image_with_nan[~np.isnan(image_with_nan)]

    all_coords = np.indices(image_with_nan.shape).reshape(2, -1).T

    print('interpolate nan')
    interpolated_image = griddata(valid_coords, valid_values, all_coords, method='linear')

    filled_image = interpolated_image.reshape(image_with_nan.shape)

    print('save dtm')
    dtm_filled = filled_image
    with rasterio.open(project_dir + 'data/dtm_filled.tif', 'w+', **dtm.profile) as out:
        out.write_band(1, dtm_filled)

    print('calc slope')
    slope_map = calculate_slope_manual(dtm_filled, cell_size)

    print('calc aspect')
    aspect_map = calculate_aspect(dtm_filled, cell_size)

    with rasterio.open(project_dir + 'data/slope.tif', 'w+', **dtm.profile) as out:
        out.write_band(1, slope_map)

    with rasterio.open(project_dir + 'data/aspect.tif', 'w+', **dtm.profile) as out:
        out.write_band(1, aspect_map)


import rasterio
import geopandas as gpd


slope = rasterio.open(project_dir + 'data/slope.tif')
slope_flip = slope.read(1)

# FIXED: nanmean instead of nansum -- slope is a continuous quantity (degrees); summing 4 values
# per 2x2 block inflated it ~4x. Each output cell now holds the average slope of its sub-pixels.
slope_flip = block_reduce(slope_flip, block_size=(2, 2), func=(np.nanmean))
slope_flip[slope_flip < 0] = 0


aspect = rasterio.open(project_dir + 'data/aspect.tif')
aspect_flip = aspect.read(1)


def block_reduce_circular_mean(angles_deg, block_size):
    """
    Average a circular quantity (compass bearings, 0-360 deg) over non-overlapping blocks.
    A plain mean of raw degrees is wrong for circular data (e.g. 350 deg and 10 deg should
    average to ~0 deg/north, not ~180 deg/south). Converts to unit vectors, averages the vector
    components per block, then converts back to an angle with atan2.
    """
    angles_rad = np.radians(angles_deg)
    sin_component = block_reduce(np.sin(angles_rad), block_size=block_size, func=np.nanmean)
    cos_component = block_reduce(np.cos(angles_rad), block_size=block_size, func=np.nanmean)
    mean_angle_deg = np.degrees(np.arctan2(sin_component, cos_component))
    mean_angle_deg = np.mod(mean_angle_deg, 360)  # wrap negative angles into [0, 360)
    return mean_angle_deg


aspect_flip = block_reduce_circular_mean(aspect_flip, block_size=(2, 2))
aspect_flip[aspect_flip < 0] = 0


field_fuel_gdf['dtm'] = extract_raster_vals(project_dir + 'data/dtm_filled.tif', field_fuel_gdf)
field_fuel_gdf['slope'] = extract_raster_vals(project_dir + 'data/slope.tif', field_fuel_gdf)
field_fuel_gdf['aspect'] = extract_raster_vals(project_dir + 'data/aspect.tif', field_fuel_gdf)
field_fuel_gdf['chm'] = extract_raster_vals(project_dir + 'data/python_chm.tif', field_fuel_gdf)


dtm = rasterio.open(project_dir + 'data/dtm_filled.tif')
dtm_flip = dtm.read(1)
dtm_flip = block_reduce(dtm_flip, block_size=(2, 2), func=(np.nanmean))
dtm_flip[dtm_flip < 0] = 0

chm = rasterio.open(project_dir + 'data/python_chm.tif')
chm_flip = chm.read(1)
chm_flip = block_reduce(chm_flip, block_size=(2, 2), func=(np.nanmean))
chm_flip[chm_flip < 0] = 0


min_x, min_y, max_x, max_y = boundary.to_crs(crs).total_bounds
burn_unit = burn_unit.to_crs(crs)

geometry_string = (burn_unit.geometry[0].wkt)
geometry_string = geometry_string.replace('(', '')
geometry_string = geometry_string.replace(')', '')
geometry_string = geometry_string.replace('MULTIPOLYGON ', '')

geometry_string_list = geometry_string.split(', ')
polygon_vertices = [[float(g.split(' ')[0]), float(g.split(' ')[1])] for g in geometry_string_list]

from matplotlib.path import Path as mplPath
poly_path = mplPath(polygon_vertices)


# make sure all the points are within the bounds

field_fuel_gdf2 = field_fuel_gdf[field_fuel_gdf.geometry.y < max_y]
field_fuel_gdf2 = field_fuel_gdf2[field_fuel_gdf2.geometry.y > min_y]
field_fuel_gdf2 = field_fuel_gdf2[field_fuel_gdf2.geometry.x < max_x]
field_fuel_gdf2 = field_fuel_gdf2[field_fuel_gdf2.geometry.x > min_x]
field_fuel_gdf2 = field_fuel_gdf2.reset_index()

train_df, test_df = train_test_split(field_fuel_gdf2, test_size=5/field_fuel_gdf2.shape[0], random_state=42)


min_x, min_y, max_x, max_y = boundary.to_crs(crs).total_bounds


# Define the grid for interpolation
grid_x = np.arange(min_x, max_x, 1)
grid_y = np.arange(min_y, max_y, 1)
print(grid_y.shape, grid_x.shape)


# leave one out cross validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

d_or_l = 'fuelbed_depth_m'  # fuelbed_depth_m, fines_load_nolit, fines_load

x = field_fuel_gdf2.geometry.x
y = field_fuel_gdf2.geometry.y
z = field_fuel_gdf2[d_or_l]


arr = [  # cc_flip_na,
    dtm_flip,
    slope_flip,
    aspect_flip,
    chm_flip]


rmse_list = []
r2 = []
mae = []

for ia, a in enumerate(arr):
    predictions = []
    actual_values = []
    
    for i in np.arange((field_fuel_gdf2.shape[0])):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        z_train = np.delete(z, i)

        x_test = x[i]
        y_test = y[i]
        z_actual = z[i]

        OK = UniversalKriging(
            x_train,
            y_train,
            z_train,
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
            drift_terms=['external_Z'],
            external_drift=a,
            external_drift_x=grid_x,
            external_drift_y=grid_y
        )

        z_pred, ss = OK.execute('points', x_test, y_test)
        predictions.append(z_pred[0])
        actual_values.append(z_actual)

    rmse_list.append(np.sqrt(mean_squared_error(actual_values, predictions)))
    r2.append(r2_score(actual_values, predictions))
    mae.append(mean_absolute_error(actual_values, predictions))


train_df, test_df = train_test_split(field_fuel_gdf2, test_size=3/field_fuel_gdf2.shape[0], random_state=42)

which_model = 3

UK_depth = UniversalKriging(
    np.array(train_df.geometry.x),
    np.array(train_df.geometry.y),
    np.array(train_df['fuelbed_depth_m']),
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False,
    drift_terms=['external_Z'],
    external_drift=arr[which_model],
    external_drift_x=grid_x,
    external_drift_y=grid_y
)
UK_load = UniversalKriging(
    np.array(train_df.geometry.x),
    np.array(train_df.geometry.y),
    np.array(train_df['fines_load_nolit']),
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False,
    drift_terms=['external_Z'],
    external_drift=arr[which_model],
    external_drift_x=grid_x,
    external_drift_y=grid_y
)

# Evaluate the method on grid
Z_pk_krig, sigma_squared_p_krig = UK_depth.execute('grid', grid_x, grid_y)
filled_with_nan_depth = Z_pk_krig.filled(np.nan)

Z_pk_krig, sigma_squared_p_krig = UK_load.execute('grid', grid_x, grid_y)
filled_with_nan_load = Z_pk_krig.filled(np.nan)


final_fuel_load = filled_with_nan_load + biomass_kg[:, :-1]
final_fuel_depth = filled_with_nan_depth  # + depth

final_fuel_depth_road = np.where(lf_surface_depth[:, :-1] == 0, 0, filled_with_nan_depth)

final_fuel_load_road = np.where(lf_surface_depth[:, :-1] == 0, 0, final_fuel_load)

# save as dat file
def write_np_array_to_dat(array, dat_name, output_dir, dtype=np.float32):
    array = np.asarray(array, dtype=dtype, order="C")  # force C order
    with FortranFile(output_dir + dat_name, "w") as f:
        f.write_record(array)


def read_dat_file(filename, nz, ny, nx, order="C"):
    """
    Read in a .dat file as a numpy array. (Duplicate of the definition earlier in this file --
    identical body; harmless but worth removing the duplicate.)
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


final_fuel_load_road_ss = np.zeros((1, ny, nx))
final_fuel_depth_road_ss = np.zeros((1, ny, nx))

final_fuel_load_road_ss[0, 0:final_fuel_load_road.shape[0], 0:final_fuel_load_road.shape[1]] = final_fuel_load_road
final_fuel_depth_road_ss[0, 0:final_fuel_depth_road.shape[0], 0:final_fuel_depth_road.shape[1]] = final_fuel_depth_road

write_np_array_to_dat(final_fuel_load_road_ss, field_fuel_load_dat, project_dir, dtype=np.float32)
write_np_array_to_dat(final_fuel_depth_road_ss, field_fuel_depth_dat, project_dir, dtype=np.float32)


