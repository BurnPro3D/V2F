import time
import zipfile

import requests

import json
import geojson
import geopandas as gpd
import sys


project_folder = sys.argv[1]
site_name = sys.argv[2] 
save_mod_treelist = project_folder + sys.argv[3]
own_treelist = sys.argv[4].lower() == "true"  # whether to use the user-provided treelist (True) or the TreeMap PIM grid (False)
zip_path = project_folder + sys.argv[5]
API_KEY = sys.argv[6]

BASE = "https://api-v2-prod-nyvjyh5ywa-uw.a.run.app"
HEADERS = {"api-key": API_KEY}
res = float(sys.argv[7])  # horizontal resolution in meters 


def poll(url: str) -> dict:
    """GET `url` every 5 s until the resource is completed or failed."""
    while True:
        r = requests.get(url, headers=HEADERS).json()
        if r["status"] in ("completed", "failed"):
            assert r["status"] == "completed", r
            return r
        time.sleep(5)

# 1. Get Domain 

domain_file = gpd.read_file(project_folder + 'data/ff_domain.geojson')
domain_file = domain_file.to_crs(5070)
boundary = geojson.loads(domain_file.to_json())

boundary["pad_to_resolution"] = res

domain = requests.post(
    f"{BASE}/domains",
    headers=HEADERS,
    json=boundary,
).json()

print(domain)
domain_id = domain["id"]
grids = f"{BASE}/domains/{domain_id}/grids"
invs = f"{BASE}/domains/{domain_id}/inventories"


# 2. Road feature from OpenStreetMap (reused as a mask below).
road = requests.post(
    f"{BASE}/domains/{domain_id}/features/road/osm",
    headers=HEADERS, json={"name": "OSM roads"},
).json()
poll(f"{BASE}/domains/{domain_id}/features/{road['id']}")

# 3. Surface fuels: FBFM40 fuel-model grid -> per-model load lookup, with road
#    cells zeroed. 


fbfm = requests.post(
    f"{grids}/fbfm40/landfire",
    headers=HEADERS,
    json={
        "name": "FBFM40 fuel model (LANDFIRE 2024)",
        "version": "2024",
        "alignment": {"target": "domain", "resolution": res, "method": "nearest"},
    },
).json()
poll(f"{grids}/{fbfm['id']}")

surface = requests.post(
    f"{grids}/lookup/fbfm40",
    headers=HEADERS,
    json={
        "name": "Surface fuel loads (FBFM40 lookup, roads masked)",
        "source_grid_id": fbfm["id"],
        "source_band": "fbfm",
        "bands": ["fuel_load.1hr", "fuel_depth", "savr.1hr"],
        "modifications": [{
            "conditions": [{"source": "feature", "operator": "intersects",
                            "target": "cell", "feature_id": road["id"]}],
            "actions": [{"band": "fuel_load.1hr", "modifier": "replace", "value": 0},
                        {"band": "fuel_depth", "modifier": "replace", "value": 0}],
        }],
    },
).json()

if own_treelist:
    CSV_PATH = save_mod_treelist
    
    inventory = requests.post(
        f"{BASE}/domains/{domain_id}/inventories/tree/upload",
        headers=HEADERS,
        json={
            "name": "Field-cruise tree inventory",
            "format": "csv",
            "columns": {
                "x": "X",
                "y": "Y",
                "dbh": "DIA",
                "height": "HT",
                "crown_ratio": "CR",
                "fia_species_code": "SPCD",
                "fia_status_code" : "STATUSCD"
            },
        },
    ).json()
    inventory_id = inventory["inventory"]["id"]
    upload = inventory["upload"]
    
    # 2. PUT the file to the signed URL (no api-key — this goes to GCS).
    with open(CSV_PATH, "rb") as f:
        put = requests.put(
            upload["url"],
            data=f,
            headers={
                "Content-Type": upload["content_type"],
                "x-goog-content-length-range": f"0,{upload['max_size_bytes']}",
            },
        )
    put.raise_for_status()
    
    # 3. Poll the inventory until it is processed.
    poll(f"{invs}/{inventory['inventory']['id']}")
    
    
    tree_grid = requests.post(
        f"{grids}/voxelize/inventory/tree",
        headers=HEADERS,
        json={
            "name": "Canopy fuel voxels",
            "source_inventory_id": inventory["inventory"]["id"],
            "resolution": {"horizontal": res, "vertical": 1},
            "bands": ["bulk_density.foliage.live", "fuel_moisture.live", "savr.foliage"],
            "biomass_source": {
                "type": "allometry", "equations": "nsvb", "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
            "seed": 42,
        },
    ).json()

else:
    # 4. Canopy fuels: TreeMap PIM grid -> tree inventory (trees near roads
    #    removed) -> 3-D voxel grid of bulk density / moisture / SAVR.
    pim = requests.post(
        f"{grids}/pim/treemap", headers=HEADERS, json={"name": "TreeMap PIM grid"},
    ).json()
    poll(f"{grids}/{pim['id']}")
    
    inventory = requests.post(
        f"{invs}/tree/pim",
        headers=HEADERS,
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
    poll(f"{invs}/{inventory['id']}")
    
    tree_grid = requests.post(
        f"{grids}/voxelize/inventory/tree",
        headers=HEADERS,
        json={
            "name": "Canopy fuel voxels",
            "source_inventory_id": inventory["id"],
            "resolution": {"horizontal": res, "vertical": 1},
            "bands": ["bulk_density.foliage.live", "fuel_moisture.live", "savr.foliage"],
            "biomass_source": {
                "type": "allometry", "equations": "nsvb", "components": ["foliage"],
                "component_states": {"foliage": {"live": 1.0, "dead": 0.0}},
            },
            "seed": 42,
        },
    ).json()
    
    
    export_treelist = requests.post(
        f"{invs}/{inventory['id']}/exports/csv",
        headers=HEADERS,
        json={
            "name": "Tree inventory export",
            "tags": [
                "trees"
            ],
        },
    ).json()
    
    export = poll(f"{BASE}/exports/{export_treelist['id']}")
    
    signed_url = export["signed_url"]
    output_file = project_folder + "data/fftl.csv"
    
    response = requests.get(signed_url)
    
    # Raise an exception for HTTP error codes (e.g., 403 Expired or 404 Not Found)
    response.raise_for_status()
    
    with open(output_file, "wb") as f:
        f.write(response.content)
    
    print(f"Successfully downloaded to {output_file}")

# 5. Terrain (3DEP elevation) and a uniform surface-moisture grid.
topography = requests.post(
    f"{grids}/topography/3dep",
    headers=HEADERS,
    json={
        "name": "Topography (3DEP 10 m)",
        "source_resolution": 10,
        "bands": ["elevation"],
        "alignment": {"target": "domain", "resolution": res},
    },
).json()
moisture = requests.post(
    f"{grids}/uniform",
    headers=HEADERS,
    json={
        "name": "Surface fuel moisture (uniform 6%)",
        "resolution": res,
        "bands": [{"key": "fuel_moisture.1hr", "value": 6.0}],
    },
).json()

# Wait for the four remaining builds.
for grid in (surface, tree_grid, topography, moisture):
    poll(f"{grids}/{grid['id']}")


# 6. Bundle everything into QUIC-Fire .dat files. Each physical quantity is a
#    {grid_id, band} role; the fire grid defaults to 2 m horizontal cells
#    (QUIC-Fire's recommended size) and 1 m vertical layers.
export = requests.post(
    f"{grids}/exports/quicfire",
    headers=HEADERS,
    json={
        "name": f"{site_name} QUIC-Fire inputs",
        "alignment": {"target": 'domain', "dx": res, "dy": res, "CRS": "EPSG:5070"},
        "canopy_bulk_density": {"grid_id": tree_grid["id"], "band": "bulk_density.foliage.live"},
        "canopy_moisture": {"grid_id": tree_grid["id"], "band": "fuel_moisture.live"},
        "canopy_savr": {"grid_id": tree_grid["id"], "band": "savr.foliage"},
        "surface_fuel_load": {"grid_id": surface["id"], "band": "fuel_load.1hr"},
        "surface_fuel_depth": {"grid_id": surface["id"], "band": "fuel_depth"},
        "surface_moisture": {"grid_id": moisture["id"], "band": "fuel_moisture.1hr"},
        "surface_savr": {"grid_id": surface["id"], "band": "savr.1hr"},
        "topography": {"grid_id": topography["id"], "band": "elevation"},
    },
).json()

export = poll(f"{BASE}/exports/{export['id']}")

# 7. Download and unzip the QUIC-Fire input bundle.
with requests.get(export["signed_url"], stream=True) as r:
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)

with zipfile.ZipFile(zip_path) as z:
    z.extractall("quicfire_inputs")
    # print(z.namelist())
# -> ['treesrhof.dat', 'treesmoist.dat', 'treesfueldepth.dat', 'topo.dat',
#     'treesss.dat', 'metadata.json', 'domain.geojson']