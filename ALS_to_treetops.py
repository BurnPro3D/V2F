# First, we import all the libraries
import numpy as np
import pandas as pd
import rasterio
from skimage.feature import peak_local_max
import os
import pdal
import json
import subprocess
import geopandas as gpd
import sys
from rasterio.features import shapes


maindir = sys.argv[1] 
zmax = float(sys.argv[2]) 
target_resolution = int(sys.argv[3])  # e.g., 4 for 4m resolution
boundaryfile = maindir + 'data/ff_domain.geojson'

projection = 'EPSG:5070'

lazpath = maindir + 'data/als_outputs/'
combinefile = lazpath + 'combine.laz'


# check if the als_outputs directory exists
if not os.path.exists(lazpath):
    os.makedirs(lazpath)

# step 1: merge all the raw .laz files if not already combined
if not(os.path.isfile(combinefile)):
    
    merge_json =     {
        "pipeline": [
            maindir + 'data/als/*.laz',
            {
            "type": "filters.merge"
            },
            {
            "type": "writers.las",
            "filename": combinefile
            }
        ]
        }


    with open(maindir+'data/merge_json.json', 'w') as f:
        json.dump(merge_json, f)

    subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/merge_json.json' ], text=True)


#step 2: get the metadata from the laz files
jsonfile = {
  "pipeline": [
    combinefile,
    {
        "type": "filters.sort",
        "dimension": "X"
    }
  ]
}

print('getting metadata')
pipeline = pdal.Pipeline(json.dumps(jsonfile))
count = pipeline.execute()
# arrays = pipeline.arrays
metadata = pipeline.metadata
# log = pipeline.log
crs = (metadata['metadata']['readers.las']['spatialreference'])

minz = metadata['metadata']['readers.las']['minz']
maxz = metadata['metadata']['readers.las']['maxz']
# determine the limit beyond laz points it would be classified as noise
zlimit  = minz + ((maxz-minz) * .9)
print(zlimit)
if zlimit > zmax:
    zlimit = zmax
print('zlimit', zlimit)

#step 3: clip the laz file to the boundary
clipfile = lazpath + 'clip_combine.laz'
boundary = gpd.read_file(boundaryfile)
boundary = boundary.to_crs(crs)

clip_json = {
    "pipeline":[
       combinefile,
        {
            "type":"filters.reprojection",
            "in_srs":crs,
            "out_srs":crs
        },
        {
        "type": "filters.crop",
        "polygon": str(boundary.geometry.iloc[0])
        },
        
        {
        "type": "writers.las",
        "compression": "true",
        "minor_version": "2",
        "dataformat_id": "0",
        "filename": clipfile
        }

        ]
        }

with open(maindir+'data/clip_json.json', 'w') as f:
    json.dump(clip_json, f)

subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/clip_json.json' ])
print('Finished clipping laz file to boundary')


#step 4: filter the noise in clipped laz file
filterfile = lazpath + 'filter_clip_combine.laz'

filt_json = {
    "pipeline":[
       clipfile,
        {
        "type": "filters.outlier",
        "method": "statistical",
        "multiplier": 2.2,
        "mean_k": 12
        },
        {
          "type":"filters.range",
          "limits":"Classification![7:7]" # drop points classified as noise
        },
        {
        "type":"filters.range",
        "limits":f"Z[0:{zlimit}]" # drop points below ground or above noise
        },
        {
        "type": "writers.las",
        "compression": "true",
        "filename": filterfile
        }

        ]
        }

with open(maindir+'data/filt_json.json', 'w') as f:
    json.dump(filt_json, f)

subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/filt_json.json' ])
print('Finished filtering noise in laz file')

#step 5: Extract the DSM from the laz file

dsm_json = {
    "pipeline":[
       filterfile,
        {
            "type":"filters.reprojection",
            "in_srs":crs,
            "out_srs":projection #reproject to EPSG:5070 for raster outputs
        },
        {
            "type":"filters.range",
            "limits":"returnnumber[1:1]" # keep only first returns
        },

        {
            "type": "writers.gdal",
            "filename":maindir + "/data/dsm.tif",
            "output_type":"idw",
            "gdaldriver":"GTiff",
            "resolution": 0.5,
            "radius": 1

        }
    ]
}

with open(maindir+'data/dsm_json.json', 'w') as f:
    json.dump(dsm_json, f)

subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/dsm_json.json' ], text=True)

print('Finished extracting DSM from laz file')

#step 6: Extract the DTM from the laz file

dtm_json = {
    "pipeline":[
         filterfile,
         {
            "type":"filters.reprojection",
            "in_srs":crs,
            "out_srs":projection
        },
        {
            "type": "filters.assign",
            "value": [
              "ReturnNumber = 1 WHERE ReturnNumber < 1",
              "NumberOfReturns = 1 WHERE NumberOfReturns < 1"
            ]
          },
        {
          "type":"filters.assign",
          "assignment":"Classification[:]=0" #reset classification so that ground points can be classified
        },
        {
          "type":"filters.elm" # flag low noise points
        },
        {
          "type":"filters.outlier" # default statistical outlier filter
        },
        {
          "type":"filters.smrf", #classify ground points
          "ignore":"Classification[7:7]",
          "slope":0.2,
          "window":16,
          "threshold":0.45,
          "scalar":1.2
        },
        {
        "type":"filters.range",
        "limits":"Classification[2:2]" #keep only ground points
        },
        {
          "type": "writers.gdal",
          # "last":'true',
          "filename":maindir + "data/dtm.tif",
          "output_type":"all",
          "gdaldriver":"GTiff",
          "resolution": 0.5
        }
    ]
}


with open(maindir+'data/dtm_json.json', 'w') as f:
    json.dump(dtm_json, f)

subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/dtm_json.json' ], text=True)

print('Finished extracting DTM from laz file')

# step 7: build canopy height model
#calculate canopy height model
print('Calculating CHM')

def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    from scipy import interpolate

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image

#read in dtm and dsm
dtm_src = rasterio.open(maindir + "data/dtm.tif")
dsm_src = rasterio.open(maindir + "data/dsm.tif")
 
dtm_tif = dtm_src.read(1)

from rasterio.warp import reproject, Resampling
 
if dsm_src.crs != dtm_src.crs:
    raise ValueError(
        f"DSM CRS ({dsm_src.crs}) and DTM CRS ({dtm_src.crs}) differ -- "
        "reproject one onto the other's CRS before alignment."
    )
 
dsm_tif = np.full((dtm_src.height, dtm_src.width), np.nan, dtype='float32')

#make sure that dsm and dtm are aligned
reproject(
    source=rasterio.band(dsm_src, 1),
    destination=dsm_tif,
    src_transform=dsm_src.transform,
    src_crs=dsm_src.crs,
    src_nodata=-9999,
    dst_transform=dtm_src.transform,
    dst_crs=dtm_src.crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear,
)

# Interpolate missing pixels in the DTM and DSM
dtm_tif[dtm_tif <=0] = np.nan #assume domain is not below sea level

dtm_emptymask = np.isnan(dtm_tif)

dsm_emptymask = np.isnan(dsm_tif)

dtm_interp = interpolate_missing_pixels(dtm_tif, dtm_emptymask)
dsm_interp = interpolate_missing_pixels(dsm_tif, dsm_emptymask)

chm = dsm_interp - dtm_interp
chm[chm < 0] = 0

#save the chm
tif = rasterio.open(maindir + "data/dtm.tif")
meta = tif.meta
meta['count'] = 1

with rasterio.open(maindir + 'data/python_chm.tif', 'w',**meta) as dest:
    dest.write(chm, 1)


chm_file = maindir + 'data/python_chm.tif'
chm_src = rasterio.open(chm_file)
chm = chm_src.read(1)

import copy
canopy_yes_no = copy.deepcopy(chm)
canopy_yes_no[canopy_yes_no < 1 ] = 0
canopy_yes_no[canopy_yes_no >= 1 ] = 1

from skimage.measure import block_reduce
from rasterio.transform import Affine


native_resolution = chm_src.res[0]  # original resolution of the CHM
block_size = int(target_resolution / native_resolution)  

downsampled_canopy = block_reduce(canopy_yes_no, block_size=(block_size, block_size), func=(np.nansum))

tif = rasterio.open(chm_file)
meta = tif.meta
meta['width'] = downsampled_canopy.shape[1]
meta['height'] = downsampled_canopy.shape[0]
meta['transform'] = tif.transform * Affine.scale(block_size, block_size)

with rasterio.open(maindir + 'data/canopy_cover.tif', 'w',**meta) as dest:
    dest.write(downsampled_canopy/block_size * block_size, 1)


#step 8: detect individual tree tops from local maxima in chm

#get the tree tops from chm
transform = chm_src.transform  # Affine transform for georeferencing

# Normalize CHM 
chm_norm = (chm - np.nanmin(chm)) / (np.nanmax(chm) - np.nanmin(chm))

# Locate tree tops using local maxima filtering
tree_tops = peak_local_max(chm_norm, min_distance=3, 
                           threshold_abs=0.1, exclude_border=False)

# Convert pixel coordinates to geospatial coordinates and extract heights
tree_points = []
for row, col in tree_tops:
    x, y = transform * (int(col), int(row))  
    height = chm[row, col] 
    tree_points.append((height, y, x)) 

df = pd.DataFrame(tree_points, columns=["HT", "y", "x"])

# Let´s look at the df
df.head()


gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.x, df.y), crs = meta['crs'])
gdf = gdf.to_crs(5070)
gdf.to_file(maindir + 'data/treetops.geojson')

# step 9: Normalize the point cloud
normalized_file = lazpath + 'normalized.laz'

normalize_json = {
  "pipeline": [
    filterfile,
    
    {
      "type": "filters.hag_nn"
    },
    {
        "type":"filters.ferry",
        "dimensions":"HeightAboveGround=>Z"
    },
    {
      "type": "writers.las",
        "compression": "true",
        "filename": normalized_file
      }
    
  ]
}

with open(maindir+'data/normalize_json.json', 'w') as f:
    json.dump(normalize_json, f)

subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/normalize_json.json' ])
print('Finished normalizing the point cloud')

# step 10: Filter the normalized point cloud and get the number of returns and height of the returns

filt_normalized_file = lazpath + 'filt_normalized.laz'

filt_dbh_json = {
    "pipeline":[
       normalized_file,
        {
        "type": "filters.outlier",
        "method": "statistical",
        "multiplier": 2.2,
        "mean_k": 12
        },
        {
          "type":"filters.range",
          "limits":"Classification![7:7]"
        },
        {
            "type": "writers.gdal",
            "filename":maindir + "/data/num_returns.tif",
            "dimension":"NumberOfReturns",
            "data_type": "uint32_t",
            "output_type":"mean",
            "gdaldriver":"GTiff",
            "resolution": 0.5,
            "radius": 1

        },
        {
            "type": "writers.gdal",
            "filename":maindir + "/data/height_returns.tif",
            "output_type":"idw",
            "gdaldriver":"GTiff",
            "resolution": 0.5,
            "radius": 1

        }

        ]
        }

with open(maindir+'data/filt_dbh.json', 'w') as f:
    json.dump(filt_dbh_json, f)

subprocess.run(['pdal', '-v', '6', 'pipeline', f'{maindir}data/filt_dbh.json' ])
print('Finished filtering the point cloud')

# step 11: return density mask and vectorize it into crown polygons
raster = rasterio.open(maindir + '/data/num_returns.tif')
r = raster.read(1)

masked_raster = np.where(r != 9999, r, 0)
masked_raster = np.where(masked_raster > 1, 1, 0)
masked_raster = masked_raster.astype(np.uint8)

# out_meta = raster.meta.copy()
# out_meta['dtype'] = 'uint8'

with rasterio.open(maindir + '/data/num_returns_filter.tif', 'w', **raster.meta) as dst:
    dst.write_band(1, masked_raster.astype(np.uint8))



with rasterio.open(maindir + '/data/num_returns_filter.tif') as src:
    # Read the first band
    image = src.read(1)
    image = image.astype(np.uint8)
    
    # Extract geometries (generator)
    results = (
        {"properties": {"value": v}, "geometry": s}
        for s, v in shapes(image, transform=src.transform)
    )
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(list(results), crs=src.crs)

gdf = gdf[gdf['value'] == 1]

gdf['area'] = gdf.area

gdf.to_file(maindir + '/data/crown_radius.geojson')

# Step 12: merge with the treetops detected, estimate crown radius per tree
gdf['treecluster'] = np.arange(gdf.shape[0])
treetops = gpd.read_file(maindir + '/data/treetops.geojson')
gdf = gdf.to_crs(treetops.crs)

### calculate the crown radius through circle math
trees_in_cluster = gdf.sjoin(treetops)
treeclusters = np.unique(trees_in_cluster.treecluster)

radius_dict = {}


for t in treeclusters:
    temp_gdf = trees_in_cluster[trees_in_cluster.treecluster == t]

    num_trees = temp_gdf.shape[0]
    cluster_size = temp_gdf['area'].values[0]
    area_per_tree = cluster_size/num_trees
    radius = np.sqrt(area_per_tree/np.pi) # assume tree is circular
    radius_dict[t] = radius

trees_in_cluster['MAX_CROWN_RADIUS'] = trees_in_cluster['treecluster'].map(radius_dict)

tc = trees_in_cluster[['HT', 'y', 'x', 'MAX_CROWN_RADIUS']].reset_index()

tc_gdf = gpd.GeoDataFrame(tc, geometry = gpd.points_from_xy(tc.x, tc.y), crs = treetops.crs)
tc_gdf.to_file(maindir + '/data/treetops_radius.geojson')

# step 13: idntify crown polygons with no detected tree top inside of them = shrubs
joined = gdf.sjoin(treetops, how='left', predicate='intersects')
anti_join = joined.loc[joined['index_right'].isna()]
anti_join.to_file(maindir + '/data/als_shrubs.geojson')