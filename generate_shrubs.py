"""
PURPOSE
-------
1. Build a table of known plot locations ("plot_blk") tagged with terrain/
   canopy attributes (slope, aspect, canopy height model, canopy cover,
   tree density), either from:
       (A) real field plots (01_plot_identification.csv / 04_tree.csv), or
       (B) Intelimon TLS scan locations (a lat/long CSV joined to the site
           domain, with plot_blk names generated from site/plot/date).
2. Train a Random Forest classifier on that labelled data to predict which
   "plot_blk" (TLS scan) best matches every other grid cell in the site
   based on its terrain/canopy attributes.
3. For every grid cell, take the shrub point cloud recorded for its
   predicted plot_blk, scatter it (with a random rotation) at that cell's
   location, and build shrub polygons (buffered by convex-hull area).
4. (Field-plot path only) Merge in field-measured fuel loading/depth data.
5. Save the generated shrub polygons to disk.

Run top-to-bottom. Config comes from CLI args (sys.argv) -- see CONFIG below.
"""

import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# CONFIG
project_folder = sys.argv[1]     # folder containing data/cell_tree_density.geojson, data/ff_domain.geojson, rasters, etc.
plot_folder = sys.argv[2]        # field-plot CSV folder, or "NA" if using Intelimon plots instead
site_name = sys.argv[3]          # only used if plot_folder != "NA"
shrub_folder = sys.argv[4]       # folder of per-plot shrub point-cloud CSVs (<plot_blk>.csv)

intelimon_plots_csv = sys.argv[5]   # path to intelimonplots.csv, or "NA" to use field plots instead

# Output
save_shrubs_file = project_folder + "data/generated_shrubs.geojson"


# functions 
def get_raster_values(raster_file, geom):
    """
    Compute the mean raster value under each polygon in `geom`.

    Parameters
    ----------
    raster_file : str                    Path to a single-band raster (.tif)
    geom        : gpd.GeoDataFrame       Polygons to sample

    Returns
    -------
    list  Mean raster value per polygon (NaN-safe)
    """
    with rasterio.open(raster_file) as src:
        values = []
        for i in np.arange(geom.shape[0]):
            target_geom = [geom.geometry.iloc[i]]

            # Use nodata padding so sampling doesn't break outside raster bounds
            out_image, out_transform = mask(
                src,
                target_geom,
                crop=True,
                all_touched=True,
                nodata=-9999,  # fills areas outside the raster with nodata
                filled=False,  # forces out-of-bounds pixels to be filled, not error
            )
            values.append(np.nanmean(out_image[0, :, :]))

    return values


#   - intelimon_plots_csv == "NA": build plots_gdf from field data
#   - intelimon_plots_csv != "NA": build plots_gdf from Intelimon TLS locations

if intelimon_plots_csv == "NA":
    # field data
    plots_df = pd.read_csv(plot_folder + "01_plot_identification.csv")
    plots_df = plots_df[plots_df.site_name == site_name]
    plots_df = plots_df[plots_df.inventory_pre_post_fire_label == "Prefire"]

    field_trees = pd.read_csv(plot_folder + "04_tree.csv")
    field_trees = field_trees.merge(plots_df, on="inventory_id")
    field_trees_crs = np.unique(field_trees.plot_coord_srs)[0]

    # plot_blk identifies which TLS scan (.ptx) each field plot corresponds to
    plot_blk = plots_df.plot_blk
    plot_blk = [p.split(".ptx")[0] for p in plot_blk]

    plots_gdf = gpd.GeoDataFrame(
        plots_df,
        geometry=gpd.points_from_xy(plots_df.plot_coord_x, plots_df.plot_coord_y),
        crs=field_trees_crs,
    )
    plots_gdf = plots_gdf.to_crs(5070)

else:
    # Intelimon TLS plot locations
    intelimon_plots = pd.read_csv(intelimon_plots_csv)
    intelimon_plots_gdf = gpd.GeoDataFrame(
        intelimon_plots,
        geometry=gpd.points_from_xy(intelimon_plots.Longitude, intelimon_plots.Latitude),
        crs=4327,
    )
    intelimon_plots_gdf = intelimon_plots_gdf.to_crs(5070)

    domain_file = gpd.read_file(project_folder + "data/ff_domain.geojson")
    domain_file = domain_file.to_crs(5070)

    # Keep only Intelimon plots that actually fall inside the site domain
    plots_subset = intelimon_plots_gdf.sjoin(domain_file)
    plots_subset = plots_subset.drop(columns="index_right")

    plot_blk = pd.read_csv(project_folder + 'data/plot_blk_list.csv')
    plot_blk['site_name'] = [str.split(p, '_')[0] for p in plot_blk['plot_blk']]
    plot_blk['plot'] = [int(str.split(p, '_')[1]) for p in plot_blk['plot_blk']]
    plot_blk['acquisition_date'] = [str.split(p, '_')[2] for p in plot_blk['plot_blk']]

    plots_subset = plots_subset.merge(plot_blk, on = ['site_name', 'plot'])
    plots_subset['plot_coord_x'] = plots_subset.geometry.x
    plots_subset['plot_coord_y'] = plots_subset.geometry.y
    plots_subset.to_file(project_folder + 'data/intelimon_plots.geojson')
    plots_gdf = plots_subset[['plot_blk', 'plot_coord_x', 'plot_coord_y', 'geometry']].copy()


# get raster attributes for each cell

tree_density = gpd.read_file(project_folder + "data/cell_tree_density.geojson")  # 10 m resolution

slope_file = project_folder + "data/slope.tif"
aspect_file = project_folder + "data/aspect.tif"
chm_file = project_folder + "data/python_chm.tif"
canopy_cover_file = project_folder + "data/canopy_cover.tif"

with rasterio.open(slope_file) as src:
    raster_bounds = box(*src.bounds)

# Only keep grid cells that are fully covered by the rasters
overlapping_polygons = tree_density[tree_density.within(raster_bounds)]

data = overlapping_polygons.copy()
data["slope"] = get_raster_values(slope_file, overlapping_polygons)
data["aspect"] = get_raster_values(aspect_file, overlapping_polygons)
data["chm"] = get_raster_values(chm_file, overlapping_polygons)
data["canopy_cover"] = get_raster_values(canopy_cover_file, overlapping_polygons)

# Cells with no ALS-detected trees have NaN n_trees -- treat that as 0
data["n_trees"] = data["n_trees"].fillna(0)


# join grids to plots

full_data = plots_gdf.sjoin(data, how="right")

data_of_interest = list(data.columns)
data_of_interest.extend(["plot_blk", "plot_coord_x", "plot_coord_y"])

data = full_data[data_of_interest]

# Cells that match a known plot become training data; the rest need a
# plot_blk prediction
training_data = data.dropna(subset=["plot_blk"])
prediction_data = data[data["plot_blk"].isna()]


# Predict plot_blk for every grid cell based on its terrain/canopy attributes

training_columns = [
    "n_trees", "slope", "aspect", "chm", "canopy_cover",
    "plot_coord_x", "plot_coord_y",
]
target = "plot_blk"

X_feat = training_data[training_columns]
y_feat = training_data[[target]]

X_train, X_test, y_train, y_test = train_test_split(X_feat, y_feat, test_size=0.33)
y_train_1d = np.array(y_train).ravel()

x_predict = data[training_columns]

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train_1d)

pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
print(f"Plot classifier accuracy: {score}")

predicted_blk = clf.predict(x_predict)

data = data.copy()
data["predicted_blk"] = predicted_blk
data.to_file(project_folder + "data/predicted_blk.geojson")


# generate shrubs
# For each grid cell, load the shrub point cloud recorded for its
# predicted plot_blk, apply a random rotation, and re-center it on that
# cell's centroid.

generated_shrubs_x = []
generated_shrubs_y = []
generated_shrubs_z = []
generated_shrubs_hull = []

for i in np.arange(data.shape[0]):
    mid_x = data.geometry.iloc[i].centroid.x
    mid_y = data.geometry.iloc[i].centroid.y

    blk_name = data.predicted_blk.iloc[i]
    blk_name = blk_name.split(".ptx")[0]

    shrubs_location = pd.read_csv(shrub_folder + blk_name + ".csv")

    # Random rotation so shrub layouts aren't identically oriented everywhere
    theta = np.random.uniform(0, 2 * np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    x_new = shrubs_location.X * c - shrubs_location.Y * s
    y_new = shrubs_location.X * s + shrubs_location.Y * c

    generated_shrubs_x.extend(x_new + mid_x)
    generated_shrubs_y.extend(y_new + mid_y)
    generated_shrubs_z.extend(shrubs_location.Z)
    generated_shrubs_hull.extend(shrubs_location.convhull_area)

# Build point geometries, then buffer each point by its convex-hull "radius"
# to approximate the shrub's footprint as a polygon
shrub_df = pd.DataFrame({
    "id": np.arange(len(generated_shrubs_x)),
    "x": generated_shrubs_x,
    "y": generated_shrubs_y,
    "z": generated_shrubs_z,
    "hull": generated_shrubs_hull,
})

shrubs_gdf = gpd.GeoDataFrame(
    shrub_df,
    geometry=gpd.points_from_xy(shrub_df.x, shrub_df.y),
    crs=5070,
)
shrubs_gdf = shrubs_gdf.set_geometry(shrubs_gdf.buffer(shrubs_gdf.hull))


# Field data only - use field measurements of fuel load/depth to override the first layer of the generated shrubs

if intelimon_plots_csv == "NA":
    fuels_df = pd.read_csv(plot_folder + "plot_fueldepth.csv")
    fuel_load_df = pd.read_csv(plot_folder + "prefire_plot_fuel_loads.csv")

    field_fuel = fuels_df.merge(plots_df, on="inventory_id")
    field_fuel = field_fuel.merge(fuel_load_df, on="inventory_id")
    field_fuel["predicted_blk"] = field_fuel["plot_blk"]

    data = data.merge(field_fuel, on="predicted_blk")
    data.to_file(project_folder + "data/predicted_blk.geojson")


# output

shrubs_gdf = shrubs_gdf.sjoin(data)
shrubs_gdf.to_file(save_shrubs_file)
print(f"Saved {shrubs_gdf.shape[0]} shrubs to {save_shrubs_file}")