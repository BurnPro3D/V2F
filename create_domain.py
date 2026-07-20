import geopandas as gpd
import pandas as pd
from shapely import box
import sys

project_folder = sys.argv[1] 

#read in boundary file, make sure it is in 5070, and create a domain that is padded by 50 m in each direction
gdf = gpd.read_file(project_folder + 'data/boundary.geojson')
gdf = gdf.to_crs(5070)

total_bounds = gdf.total_bounds
total_bounds_pad = [total_bounds[0] - 50,
                    total_bounds[1] - 50,
                    total_bounds[2] + 50,
                    total_bounds[3] + 50]

polygon = box(*total_bounds_pad)


domain = gpd.GeoDataFrame(pd.DataFrame({'id':[1]}), geometry = [polygon], crs = 5070)
domain.to_file(project_folder + 'data/ff_domain.geojson')