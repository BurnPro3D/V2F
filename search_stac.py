

from pathlib import Path
import sys

from pystac.client import Client
import geopandas as gpd

project_folder = sys.argv[1]
# Load the area of interest

geojson_path = Path(project_folder + "data/ff_domain.geojson")
gdf = gpd.read_file(geojson_path)

# Create connection to STAC Catalog

stac_url = 'https://ndp-test.sdsc.edu/stac'
catalog = Client.open(stac_url)

# Search STAC for items interesecting with area of interest
search_results = catalog.search(
    bbox=gdf.total_bounds
)

# Display matching items
print('Catalog entries in area of interest:')
for item in search_results.items():
    print(item.id)

