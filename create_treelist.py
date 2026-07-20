"""
create_treelist.py

Merged / cleaned-up version of:
    - 2a_create_treelist_field_data.ipynb
    - 2a_create_treelist_no_field_data.ipynb

PURPOSE
-------
Build a modified tree list for a site by taking ALS (airborne laser scan)
detected trees, optionally padding tree density up to a target trees-per-acre
(TPA) using synthetic trees, then predicting DBH, species, crown ratio and
crown base height (CBH) for every tree using either:

    (A) real field-plot inventory data (FIA-style plot CSVs), or
    (B) a FastFuels-API-generated synthetic tree inventory, used when no
        field data is available (plot_folder == "NA").

Both source notebooks were almost identical after the training-treelist is
built — this script keeps ONE copy of all the shared modelling logic and
just branches where the two notebooks actually differed:

    1. How `training_treelist` is built (field CSVs vs FastFuels API)
    2. Slope/aspect: the no-field-data notebook cached slope/aspect/dtm to
       disk and skipped recomputation if the files already existed -- that
       (better) behavior is kept here for both paths.

Run top-to-bottom. Update the CONFIG block for your site before running.
"""

import math
import os
import random
import time
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import shapely
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG
# ============================================================
# `project_folder` must contain:
#   ff_domain.geojson         - site boundary
#   data/treetops_radius.geojson  - ALS-detected tree locations/attributes
#   data/dtm.tif               - digital terrain model
#
# `plot_folder`:
#   - Path to a folder of field-plot CSVs (01_plot_identification.csv,
#     04_tree.csv) if you have real field data for this site, OR
#   - "NA" if you don't have field data. In that case a synthetic tree
#     inventory is pulled from the FastFuels API instead.
#
# `site_name` is only used when plot_folder != "NA" (must match a
# `site_name` value in 01_plot_identification.csv).



# ---- Paths / site identity ----
project_folder = sys.argv[1]       # folder containing data/ff_domain.geojson, data/treetops_radius.geojson, data/dtm.tif
plot_folder = sys.argv[2]          # field-plot CSV folder, or "NA" if no field data
site_name = sys.argv[3]            # only used if plot_folder != "NA"
save_mod_treelist = project_folder + sys.argv[4]    # output CSV path, e.g. project_folder + "data/treelist_mod_jun30_added_trees.csv"
 
# ---- Modelling options ----
avg_tpa = None if sys.argv[5] == "None" else float(sys.argv[5])  # target trees-per-acre; None = derive automatically
add_trees = sys.argv[6].lower() == "true"                        # whether to add synthetic trees to reach avg_tpa
 
# ---- FastFuels API settings (only used when plot_folder == "NA") ----
FASTFUELS_API_KEY = sys.argv[7]
FASTFUELS_BASE = "https://api-v2-prod-nyvjyh5ywa-uw.a.run.app"
FASTFUELS_RES = 1  # grid resolution (meters) used to pad/align the domain



# ============================================================
# HELPER FUNCTIONS
# ============================================================

def RSE(y_true, y_predicted):
    """
    Residual Standard Error -- square root of RSS / (n - 2).
    Used to characterise prediction spread for adding simulation noise.

    Parameters
    ----------
    y_true      : array-like  Observed values
    y_predicted : array-like  Fitted/predicted values

    Returns
    -------
    float  RSE value
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)
    RSS = np.sum(np.square(y_true - y_predicted))
    rse = math.sqrt(RSS / (len(y_true) - 2))
    return rse


def lin_reg(df, x_name, y_name, predict_x, simulate=False):
    """
    Fit a simple OLS linear regression and return predictions.

    Parameters
    ----------
    df        : pd.DataFrame  Training data
    x_name    : str           Predictor column name
    y_name    : str           Response column name
    predict_x : np.ndarray    Values to predict on (must be shaped (-1, 1))
    simulate  : bool          If True, add Gaussian noise scaled by RSE

    Returns
    -------
    list  Predicted (or simulated) values
    """
    df = df[[x_name, y_name]].dropna()

    x_arr = np.array(df[x_name]).reshape(-1, 1)
    y_arr = np.array(df[y_name]).reshape(-1, 1)

    mod = LinearRegression()
    mod.fit(x_arr, y_arr)

    predictions_true = mod.predict(x_arr)
    predictions = mod.predict(predict_x)

    if simulate:
        sd = RSE(y_arr, predictions_true)
        simulations = [np.random.normal(d, sd)[0] for d in predictions]
        return simulations
    else:
        predictions = [d[0] for d in predictions]
        return predictions


def calculate_slope_manual(dtm_array, cell_size):
    """
    Compute terrain slope in degrees using numpy gradient (central differences).

    Parameters
    ----------
    dtm_array : np.ndarray  2-D elevation raster
    cell_size : float       Pixel resolution in map units

    Returns
    -------
    np.ndarray  Slope in degrees (same shape as dtm_array)
    """
    # Calculate gradients in x and y directions
    dz_dx = np.gradient(dtm_array, axis=1) / cell_size
    dz_dy = np.gradient(dtm_array, axis=0) / cell_size

    # Calculate slope in radians
    slope_radians = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

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


def lookup_slope_aspect(geom_x, geom_y, slope_map, aspect_map, origin_x, origin_y, res):
    """
    Vectorised raster lookup for slope and aspect at a set of point coordinates.
    Returns two lists: slope_val, aspect_val (rounded ints or NaN).
    """
    col_idx = ((geom_x - origin_x) / res).astype(int)
    row_idx = ((origin_y - geom_y) / res).astype(int)

    slp_vals = slope_map[row_idx, col_idx]
    asp_vals = aspect_map[row_idx, col_idx]

    # Round non-NaN values to integers
    slope_out = np.where(np.isnan(slp_vals), np.nan, np.round(slp_vals))
    aspect_out = np.where(np.isnan(asp_vals), np.nan, np.round(asp_vals))
    return slope_out.tolist(), aspect_out.tolist()


def poll(url, headers):
    """GET `url` every 5s until the FastFuels resource is completed or failed."""
    while True:
        r = requests.get(url, headers=headers).json()
        if r["status"] in ("completed", "failed"):
            assert r["status"] == "completed", r
            return r
        time.sleep(5)


def build_treelist_from_fastfuels(project_folder, res, api_key, base_url):
    """
    (Used only when plot_folder == 'NA', i.e. no field data available.)

    Build a synthetic training tree inventory via the FastFuels v2 API:
    creates a domain from the site boundary, masks roads, builds a
    TreeMap-based tree inventory (with trees near roads removed), and
    downloads the result as a CSV.

    Returns
    -------
    pd.DataFrame  Raw FastFuels tree inventory export
    """
    headers = {"api-key": api_key}
    import geojson  # local import: only needed for this code path

    # 1. Domain, padded to a clean lattice at `res` meters
    domain_file = gpd.read_file(project_folder + "data/ff_domain.geojson")
    domain_file = domain_file.to_crs(5070)
    boundary_geojson = geojson.loads(domain_file.to_json())
    boundary_geojson["pad_to_resolution"] = res

    domain = requests.post(f"{base_url}/domains", headers=headers, json=boundary_geojson).json()
    domain_id = domain["id"]
    grids = f"{base_url}/domains/{domain_id}/grids"
    invs = f"{base_url}/domains/{domain_id}/inventories"

    # 2. Road feature from OpenStreetMap (used to mask fuels & thin trees near roads)
    road = requests.post(
        f"{base_url}/domains/{domain_id}/features/road/osm",
        headers=headers, json={"name": "OSM roads"},
    ).json()
    poll(f"{base_url}/domains/{domain_id}/features/{road['id']}", headers)

    # 3. Canopy fuels: TreeMap PIM grid -> tree inventory (trees near roads removed)
    pim = requests.post(
        f"{grids}/pim/treemap", headers=headers, json={"name": "TreeMap PIM grid"},
    ).json()
    poll(f"{grids}/{pim['id']}", headers)

    inventory = requests.post(
        f"{invs}/tree/pim",
        headers=headers,
        json={
            "name": "Tree inventory (trees near roads removed)",
            "source_pim_grid_id": pim["id"],
            "seed": 42,
            "modifications": [{
                "conditions": [{"source": "feature", "operator": "within",
                                "feature_id": road["id"], "buffer_m": 5}],
                "actions": [{"modifier": "remove"}],
            }],
        },
    ).json()
    poll(f"{invs}/{inventory['id']}", headers)

    # 4. Export the tree inventory as CSV and download it
    export_treelist = requests.post(
        f"{invs}/{inventory['id']}/exports/csv",
        headers=headers,
        json={"name": "Tree inventory export", "tags": ["trees"]},
    ).json()
    export = poll(f"{base_url}/exports/{export_treelist['id']}", headers)

    signed_url = export["signed_url"]
    output_file = project_folder + "data/fftl.csv"

    response = requests.get(signed_url)
    response.raise_for_status()
    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"Successfully downloaded FastFuels tree inventory to {output_file}")
    return pd.read_csv(output_file)


# ============================================================
# LOAD SPATIAL DATA
# ============================================================

# Site boundary and ALS-detected tree locations
boundary = gpd.read_file(project_folder + "data/ff_domain.geojson")
als_trees = gpd.read_file(project_folder + "data/treetops_radius.geojson")

# Reproject boundary to match ALS data CRS before any spatial ops
boundary = boundary.to_crs(als_trees.crs)

# Load FIA species code lookup table (SPCD <-> scientific name)
tree_species_code = pd.read_csv(
    "./reference/FIATreeSpeciesCode.csv"
)

# Restrict ALS trees to those inside the boundary
als_trees_in_boundary = als_trees.sjoin(boundary)
als_trees_in_boundary = als_trees_in_boundary.drop(["index_right"], axis=1)


# ============================================================
# OPTIONAL: ADD SYNTHETIC TREES TO REACH TARGET DENSITY
# ============================================================
# Compares ALS-detected TPA against a target TPA (either given, derived
# from field plots, or 1.5x the ALS TPA) and, if the ALS trees fall
# short, scatters extra "placeholder" trees into a weighted random grid
# (denser cells get more new trees) to make up the difference.

if add_trees:
    # Compute observed (ALS) TPA from boundary area
    boundary_area_acres = boundary.area / 4047
    als_tpa = (als_trees_in_boundary.shape[0] / boundary_area_acres)[0]

    if avg_tpa is None:
        avg_tpa = als_tpa * 1.5

    # If field plots exist, derive a better avg TPA estimate from them
    if plot_folder != "NA":
        field_trees_for_tpa = pd.read_csv(plot_folder + "04_tree.csv")
        plots_df = pd.read_csv(plot_folder + "01_plot_identification.csv")
        plots_df = plots_df[plots_df.site_name == site_name]
        field_trees_for_tpa = field_trees_for_tpa.merge(plots_df, on="inventory_id")

        tpa_list = []
        plot_size = 1 / 10
        for i in np.unique(field_trees_for_tpa.inventory_id):
            ft_sub = field_trees_for_tpa[field_trees_for_tpa.inventory_id == i]
            n_trees = ft_sub.shape[0]
            tpa_list.append(n_trees * 10)  # scale per-plot count up to per-acre

        avg_tpa = np.median(np.array(tpa_list))

    # How many synthetic trees are needed to reach avg_tpa
    trees_to_add = round((boundary_area_acres * avg_tpa) - als_trees_in_boundary.shape[0])[0]
    print(f"Trees to add: {trees_to_add}")

    # Build a grid over the ALS trees and count trees per cell
    xmin, ymin, xmax, ymax = als_trees_in_boundary.total_bounds
    cell_size = 10

    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            grid_cells.append(shapely.geometry.box(x0, y0, x1, y1))

    cell = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs=als_trees_in_boundary.crs)
    cell["ind_name"] = np.arange(cell.shape[0])

    count_trees = gpd.sjoin(als_trees_in_boundary, cell, how="left", predicate="within")
    count_trees["n_trees"] = 1
    count_trees["trees"] = 0
    count_trees["n_trees"] = count_trees["n_trees"].fillna(0)
    count_trees["trees"] = count_trees["trees"].fillna(0)
    count_trees["n_trees"] = count_trees["n_trees"] + count_trees["trees"]
    count_trees["n_trees"] = count_trees.n_trees.replace(0, np.nan)

    dissolved = count_trees.dissolve(by="index_right", aggfunc="count")
    cell.loc[dissolved.index, "n_trees"] = dissolved.n_trees.values
    cell["proportions"] = cell.n_trees / np.nansum(cell.n_trees)
    cell_na_remove = cell.dropna()
    cell.to_file(project_folder + "data/cell_tree_density.geojson")

    # Weighted-random assignment: denser cells get proportionally more new trees
    where_to_add = random.choices(
        population=np.array(cell_na_remove["ind_name"]),
        weights=np.array(cell_na_remove["n_trees"]),
        k=int(trees_to_add),
    )

    # Sample one random point inside each chosen cell -> new tree location
    loc_in_grid = []
    for i in np.arange(len(where_to_add)):
        loc_in_grid.append(
            cell[cell.ind_name == where_to_add[i]].sample_points(size=1).reset_index(drop=True)[0]
        )

    # Start the "new trees" table as a copy of the real ALS trees
    als_trees_simple = pd.DataFrame({
        "HT": als_trees_in_boundary.HT,
        "MAX_CROWN_RADIUS": als_trees_in_boundary.MAX_CROWN_RADIUS,
        "x": als_trees_in_boundary.geometry.x,
        "y": als_trees_in_boundary.geometry.y,
    })
    new_trees_gdf = gpd.GeoDataFrame(
        als_trees_simple,
        geometry=gpd.points_from_xy(als_trees_simple.x, als_trees_simple.y),
        crs=als_trees_in_boundary.crs,
    )


# ============================================================
# SLOPE & ASPECT FROM DTM (cached to disk if already computed)
# ============================================================
# Computing slope/aspect for a large DTM is slow, so once computed the
# results are written to slope.tif / aspect.tif / dtm_filled.tif and
# re-used on subsequent runs instead of being recomputed.

slope_path = project_folder + "data/slope.tif"
aspect_path = project_folder + "data/aspect.tif"
dtm_filled_path = project_folder + "data/dtm_filled.tif"

if os.path.exists(slope_path) and os.path.exists(aspect_path) and os.path.exists(dtm_filled_path):
    print("Slope/aspect files already exist, loading from disk.")
else:
    print("Computing slope/aspect from DTM...")
    dtm = rasterio.open(project_folder + "data/dtm.tif")

    with rasterio.open(project_folder + "data/dtm.tif") as src:
        dtm_array = src.read(1).astype(float)
        cell_size = src.res[0]  # assumes square pixels

    # Treat nodata (-9999) as NaN, then fill holes via linear interpolation
    dtm_array[dtm_array == -9999] = np.nan
    valid_coords = np.argwhere(~np.isnan(dtm_array))
    valid_values = dtm_array[~np.isnan(dtm_array)]
    all_coords = np.indices(dtm_array.shape).reshape(2, -1).T
    interpolated = griddata(valid_coords, valid_values, all_coords, method="linear")
    dtm_filled = interpolated.reshape(dtm_array.shape)

    with rasterio.open(dtm_filled_path, "w+", **dtm.profile) as out:
        out.write_band(1, dtm_filled)

    slope_map = calculate_slope_manual(dtm_filled, cell_size)
    aspect_map = calculate_aspect(dtm_filled, cell_size)

    with rasterio.open(slope_path, "w+", **dtm.profile) as out:
        out.write_band(1, slope_map)
    with rasterio.open(aspect_path, "w+", **dtm.profile) as out:
        out.write_band(1, aspect_map)

# Always (re)load from disk so behavior is identical whether we just
# computed them or they already existed
slope_map = rasterio.open(slope_path).read(1)
aspect_map = rasterio.open(aspect_path).read(1)


# ============================================================
# BUILD TRAINING TREELIST
# ============================================================
# This is the branch point between the two source notebooks:
#   - plot_folder != "NA": build training_treelist from real field-plot CSVs
#   - plot_folder == "NA": build training_treelist from a FastFuels-API
#     synthetic tree inventory instead

if plot_folder != "NA":
    # ---- Path A: real field plot data ----
    field_trees = pd.read_csv(plot_folder + "04_tree.csv")
    plots_df = pd.read_csv(plot_folder + "01_plot_identification.csv")
    plots_df = plots_df[plots_df.site_name == site_name]
    field_trees = field_trees.merge(plots_df, on="inventory_id")
    field_trees_crs = np.unique(field_trees.plot_coord_srs)[0]

    # Rename/derive standard columns
    field_trees["HT"] = field_trees.tree_ht
    field_trees["DIA"] = field_trees.tree_dbh
    field_trees["SCI_NAME"] = field_trees.tree_sp_scientific_name
    field_trees["STATUSCD"] = np.where(field_trees.tree_status == "Live", 1, 0)
    field_trees["CR"] = field_trees.tree_htlcb / field_trees.tree_ht  # crown ratio
    field_trees["X"] = field_trees.plot_coord_x
    field_trees["Y"] = field_trees.plot_coord_y

    # Join FIA species codes
    field_trees = field_trees.merge(tree_species_code, on="SCI_NAME")
    field_trees = field_trees[["SPCD", "STATUSCD", "DIA", "HT", "CR", "X", "Y"]]
    field_trees = field_trees.fillna(value={"CR": 0.6})  # default CR when missing

    field_trees_gpd = gpd.GeoDataFrame(
        field_trees,
        geometry=gpd.points_from_xy(field_trees.X, field_trees.Y),
        crs=field_trees_crs,
    )
    training_treelist = field_trees_gpd.to_crs(boundary.crs)
    training_treelist["CBH"] = training_treelist["CR"] * training_treelist["HT"]

else:
    # ---- Path B: no field data -> synthetic inventory from FastFuels API ----
    ff_treelist = build_treelist_from_fastfuels(
        project_folder, FASTFUELS_RES, FASTFUELS_API_KEY, FASTFUELS_BASE
    )

    training_treelist = ff_treelist
    training_treelist["HT"] = training_treelist["height"]
    training_treelist["DIA"] = training_treelist["dbh"]
    training_treelist["CR"] = training_treelist["crown_ratio"]
    training_treelist["X"] = training_treelist["x"]
    training_treelist["Y"] = training_treelist["y"]
    training_treelist["SPCD"] = training_treelist["fia_species_code"]
    training_treelist["CBH"] = training_treelist["HT"] / 2  # no CBH field available; approximate
    training_treelist = gpd.GeoDataFrame(
        training_treelist,
        geometry=gpd.points_from_xy(training_treelist.X, training_treelist.Y),
        crs=5070,
    )


# ============================================================
# OPTIONAL: GENERATE SYNTHETIC TREE HEIGHTS (add_trees path)
# ============================================================
# Sample new tree heights from a normal distribution fit to the training
# treelist's heights (rejecting samples outside the observed range), then
# attach a default crown radius (25th percentile of ALS crown radii).

if add_trees:
    mu, std = norm.fit(training_treelist.HT)
    ht_min = np.nanmin(training_treelist.HT)
    ht_max = np.nanmax(training_treelist.HT)

    generated_ht = []
    for _ in np.arange(len(where_to_add)):
        random_ht = -1
        while not (ht_min <= random_ht <= ht_max):
            random_ht = np.random.normal(mu, std)
        generated_ht.append(random_ht)

    generated_ht_gdf = gpd.GeoDataFrame(
        pd.DataFrame({"HT": generated_ht}),
        geometry=loc_in_grid,
        crs=cell.crs,
    )
    generated_ht_gdf["x"] = generated_ht_gdf.geometry.x
    generated_ht_gdf["y"] = generated_ht_gdf.geometry.y
    generated_ht_gdf["MAX_CROWN_RADIUS"] = np.quantile(
        als_trees_in_boundary.MAX_CROWN_RADIUS, 0.25
    )

    print(f"new_trees_gdf before adding synthetic trees: {new_trees_gdf.shape}")
    new_trees_gdf = pd.concat([new_trees_gdf, generated_ht_gdf[new_trees_gdf.columns]])
    print(f"new_trees_gdf after adding synthetic trees: {new_trees_gdf.shape}")


# ============================================================
# PREDICT DBH FROM TREE HEIGHT
# ============================================================
# Fits both an exponential and a linear HT->DBH model on the training
# treelist and uses whichever has the better R^2 to predict DBH for
# every "new" tree (ALS-detected + any synthetic ones).

training_treelist_bound = training_treelist.sjoin(boundary)

if not add_trees:
    # If we didn't add synthetic trees, the "new trees" are just the ALS trees
    new_trees_gdf = als_trees_in_boundary.dropna(subset=["HT"])

new_tree_ht = np.array(new_trees_gdf.HT).reshape(-1, 1)


def exponential_func(x, a, b):
    """Simple exponential model: y = a * exp(b * x)"""
    return a * np.exp(b * x)


params, covariance = curve_fit(
    exponential_func,
    training_treelist_bound.HT,
    training_treelist_bound.DIA,
    p0=[1, 0.5],
)
a_fit, b_fit = params

y_pred_exp = exponential_func(training_treelist_bound.HT, a_fit, b_fit)
r2_exp = r2_score(training_treelist_bound.DIA, y_pred_exp)

X_ht = np.array(training_treelist_bound.HT).reshape(-1, 1)
Y_dia = np.array(training_treelist_bound.DIA).reshape(-1, 1)
mod = LinearRegression()
mod.fit(X_ht, Y_dia)
y_predicted = mod.predict(X_ht)
r2_lin = r2_score(training_treelist_bound.DIA, y_predicted)

# Choose the better-fitting model to predict DBH for all new trees
if r2_exp > r2_lin:
    dbh = exponential_func(new_tree_ht, a_fit, b_fit)
else:
    dbh = mod.predict(np.array(new_tree_ht).reshape(np.array(new_tree_ht).shape[0], 1))

new_trees_gdf = new_trees_gdf.copy()
new_trees_gdf["DIA"] = dbh

# Add x/y coordinate columns needed for downstream spatial lookups
training_treelist_bound = training_treelist_bound.copy()
training_treelist_bound["x_point"] = training_treelist_bound.geometry.x
training_treelist_bound["y_point"] = training_treelist_bound.geometry.y
new_trees_gdf["x_point"] = new_trees_gdf.x
new_trees_gdf["y_point"] = new_trees_gdf.y

# Drop rows missing the core modelling columns
training_treelist_no_na = training_treelist_bound.dropna(
    subset=["HT", "DIA", "SPCD", "x_point", "y_point"]
)
print(f"new_trees_gdf shape after DBH prediction: {new_trees_gdf.shape}")


# ============================================================
# ATTACH SLOPE & ASPECT TO TRAINING TREES
# ============================================================

tif = rasterio.open(project_folder + "data/dtm.tif")
meta = tif.meta
origin_x = meta["transform"][2]  # top-left x
origin_y = meta["transform"][5]  # top-left y
res = meta["transform"][0]       # pixel size

training_treelist_no_na = training_treelist_no_na.copy()
s_vals, a_vals = lookup_slope_aspect(
    training_treelist_no_na.geometry.x.values,
    training_treelist_no_na.geometry.y.values,
    slope_map, aspect_map, origin_x, origin_y, res,
)
training_treelist_no_na["slope"] = s_vals
training_treelist_no_na["aspect"] = a_vals


# ============================================================
# FILTER NEW TREES TO DTM EXTENT, THEN ADD SLOPE & ASPECT
# ============================================================

min_x = origin_x
max_x = min_x + (meta["width"] * res)
max_y = origin_y
min_y = max_y - (meta["height"] * res)

in_extent = (
    (new_trees_gdf.x > min_x) &
    (new_trees_gdf.x < max_x) &
    (new_trees_gdf.y < max_y) &
    (new_trees_gdf.y > min_y)
)
new_trees_gdf = new_trees_gdf[in_extent].copy()

s_vals, a_vals = lookup_slope_aspect(
    new_trees_gdf.geometry.x.values,
    new_trees_gdf.geometry.y.values,
    slope_map, aspect_map, origin_x, origin_y, res,
)
new_trees_gdf["slope"] = s_vals
new_trees_gdf["aspect"] = a_vals
new_trees_gdf = new_trees_gdf.dropna()
new_trees_gdf["X"] = new_trees_gdf["x"]
new_trees_gdf["Y"] = new_trees_gdf["y"]

print(f"new_trees_gdf shape after slope/aspect filter: {new_trees_gdf.shape}")


# ============================================================
# SPECIES CLASSIFICATION (Random Forest)
# ============================================================
# Trains a RF classifier on HT/DIA/slope/aspect -> SPCD using the field
# (or FastFuels) training data, then predicts species for every new tree.

training_columns = ["HT", "DIA", "slope", "aspect"]
target = "SPCD"

X_feat = training_treelist_no_na[training_columns]
y_feat = training_treelist_no_na[[target]]
X_train, X_test, y_train, y_test = train_test_split(X_feat, y_feat, test_size=0.33)
y_train_1d = np.array(y_train).ravel()

x_predict = new_trees_gdf[training_columns]

clf = RandomForestClassifier(
    n_estimators=317,
    max_depth=23,
    min_samples_split=6,
    min_samples_leaf=1,
    max_features=None,
    bootstrap=True,
    criterion="log_loss",
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train_1d)

pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(f"Species classifier accuracy: {score:.3f}")

predicted_species = clf.predict(x_predict)


# ============================================================
# ASSIGN SPECIES AND ESTIMATE CROWN ATTRIBUTES
# ============================================================

# Add common/scientific names to training treelist for reference
training_treelist_species = training_treelist.merge(tree_species_code, on="SPCD")

new_trees_gdf = new_trees_gdf.copy()
new_trees_gdf["SPCD"] = predicted_species
new_trees_gdf["STATUSCD"] = 1  # assume all ALS-detected/new trees are live

# For each predicted species, estimate crown ratio (CR) and crown base
# height (CBH) from that species' training data via linear regression.
# If a species has no training data, fall back to simple HT-based rules.
chunks = []
for s in np.unique(new_trees_gdf.SPCD):
    new_trees_gdf_subset = new_trees_gdf[new_trees_gdf.SPCD == s].copy()
    subset_df = new_trees_gdf_subset.copy()

    if subset_df.shape[0] > 0:
        predict_x = np.array(new_trees_gdf_subset.DIA).reshape(-1, 1)
        est_crownratio = np.array(
            lin_reg(training_treelist_species, "DIA", "CR", predict_x)
        )
        new_trees_gdf_subset["CR"] = est_crownratio

        training_cbh = training_treelist_species[["DIA", "HT", "CBH"]].dropna()
        X_cbh = np.array(training_cbh[["DIA", "HT"]]).reshape(-1, 2)
        y_cbh = np.array(training_cbh["CBH"])

        predict_cbh_x = np.array(new_trees_gdf_subset[["DIA", "HT"]]).reshape(-1, 2)
        cbh_lin = LinearRegression().fit(X_cbh, y_cbh)
        new_trees_gdf_subset["CBH"] = cbh_lin.predict(predict_cbh_x)
    else:
        # Fallback for species not represented in the training data
        new_trees_gdf_subset["CR"] = new_trees_gdf_subset.HT / 10
        new_trees_gdf_subset["CBH"] = new_trees_gdf_subset.HT / 2

    chunks.append(new_trees_gdf_subset)

new_trees_with_extra = pd.concat(chunks, ignore_index=True)


# ============================================================
# FINAL CLEANUP AND OUTPUT
# ============================================================

new_trees_all = new_trees_with_extra.copy()
new_trees_all = new_trees_all[new_trees_all.DIA > 0]
new_trees_all = new_trees_all[new_trees_all.DIA < 1000]
new_trees_all["TREE_ID"] = np.arange(new_trees_all.shape[0])

# Add in Genus and Species (parsed from FIA scientific name)
new_trees_all = new_trees_all.merge(
    tree_species_code[["SPCD", "SCI_NAME"]], on="SPCD"
)
new_trees_all["Genus"] = new_trees_all.SCI_NAME.str.split(" ").str[0]
new_trees_all["Species"] = new_trees_all.SCI_NAME.str.split(" ").str[1]

# Reproject to Albers Equal Area (EPSG:5070) for consistent X/Y output
new_trees_all = new_trees_all.to_crs(5070)
new_trees_all["X"] = new_trees_all.geometry.x
new_trees_all["Y"] = new_trees_all.geometry.y

columns_of_interest = [
    "TREE_ID", "SPCD", "STATUSCD", "DIA", "HT", "CR",
    "X", "Y", "CBH", "MAX_CROWN_RADIUS",
]
als_modified_treelist = new_trees_all[columns_of_interest].copy()

# Cap crown ratio at 1.0 (can't exceed full tree height)
als_modified_treelist["CR"] = als_modified_treelist["CR"].clip(upper=1.0)

# Drop trees with no crown radius
als_modified_treelist = als_modified_treelist[
    als_modified_treelist["MAX_CROWN_RADIUS"] > 0
]

# Final spatial filter: keep only trees inside the site boundary
boundary_5070 = boundary.to_crs(5070)
bx = boundary_5070.bounds.iloc[0]

final = als_modified_treelist[
    (als_modified_treelist.X > bx.minx) &
    (als_modified_treelist.X < bx.maxx) &
    (als_modified_treelist.Y > bx.miny) &
    (als_modified_treelist.Y < bx.maxy)
]

final.to_csv(save_mod_treelist)
print(f"Saved {final.shape[0]} trees to {save_mod_treelist}")