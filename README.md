# V2F - Vegetation to Fuels


The setup for this script assumes that this is how the data is set up:

Main folder

-> data

--> als

inside the data folder, you have a burn unit boundary geojson (boundary.geojson)
inside the als folder, you have the .laz files from 3DEP downloaded


if you have run the updated IntELimon code, it was already run and outputs are saved to a folder (you can set the path)


To run this script, you also need to have a FastFuels v2 API key, available [here](https://beta-app-fastfuels-silvxlabs.web.app) 
run export FF_API_KEY="your-key-here" in the terminal to set the API key


To run run_als.sh, you will need to have PDAL installed - see [here](https://pdal.org/en/2.10.2/quickstart.html)


Usage: ./run_[als or code].sh [path/to/pipeline_config.json]


Config file description:

project_folder - this is a string (path), and points to the main folder

plot_folder - this is a string(path), and points to where the field data is stored. in this example, we have field data from the UCCA project. if there is no field data, "NA" is acceptable

site_name - this is a string, describes the name of the site in the field data

site - this is a string, describes the name of the site (could be different than what it is called in the field data)

als_treelist - string(relative path), describes what you want to name the treelist that you will derive from als

ee_project - string, this is the project name for access to the Google earth engine API

zipname - string, this describes the file name you want to export the zipped 3D fuels from FastFuels export

zmax - number (int), this is the height in meters that you will filter the raw point cloud

avg_tpa - number (int), or "None", this is the average TPA for the site (if known)

add_trees - string ("True" or "False"), this is a flag that is passed to the code that will tell it if you 
want to do the tree infilling or not

own_treelist - string ("True" or "False"), this is a flag that is passed to the code that will tell it if you want to bring a custom treelist to be voxelized in FastFuels

intelimon_plots_csv - string (path), this is the path to where the csv describing where intelimon plots are located, needed if you don't have field data

shrub_folder - string (path), this is a path to where the csv files describing the shrubs are stored (these files are the shrub csv generated from the intelimon code)

field_fuel_load_dat - string (path) or "NA", this is the path to where you want to save your new fuel loading dat file, path only used if there is field data

field_fuel_depth_dat - string (path) or "NA", this is the path to where you want to save your new fuel depth dat file, path only used if there is field data

There is an zipped example for St. Marks [here](https://drive.google.com/file/d/1aMfSuTSHzfyklssZkBnJwYbkIUxCOhZO/view?usp=drive_link) with all the starting data that is necessary to run this script.
