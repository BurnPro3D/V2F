#!/bin/bash
set -e   # stop immediately if any step below fails, instead of silently continuing

# ----------------------------------------------------------------------------
# Usage: ./run_pipeline.sh [path/to/pipeline_config.json]
# Defaults to pipeline_config.json in the same directory as this script if no
# argument is given.
#
# Requires `jq` (a command-line JSON processor)
# ----------------------------------------------------------------------------

if ! command -v jq &> /dev/null; then
    echo "Error: this script requires 'jq' to read the JSON config file." 
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_file="${1:-$script_dir/pipeline_config.json}"

if [[ ! -f "$config_file" ]]; then
    echo "Error: config file not found: $config_file" >&2
    exit 1
fi

# -r (raw output, no surrounding quotes) so values drop straight into bash variables cleanly.
project_folder=$(jq -r '.project_folder' "$config_file")
plot_folder=$(jq -r '.plot_folder' "$config_file")
site_name=$(jq -r '.site_name' "$config_file")
site=$(jq -r '.site' "$config_file")
als_treelist=$(jq -r '.als_treelist' "$config_file")
ee_project=$(jq -r '.ee_project' "$config_file")
zipname=$(jq -r '.zipname' "$config_file")
zmax=$(jq -r '.zmax' "$config_file")
avg_tpa=$(jq -r '.avg_tpa' "$config_file")
add_trees=$(jq -r '.add_trees' "$config_file")
own_treelist=$(jq -r '.own_treelist' "$config_file")
intelimon_plots_csv=$(jq -r '.intelimon_plots_csv' "$config_file")
shrub_folder=$(jq -r '.shrub_folder' "$config_file")
field_fuel_load_dat=$(jq -r '.field_fuel_load_dat' "$config_file")
field_fuel_depth_dat=$(jq -r '.field_fuel_depth_dat' "$config_file")
target_resolution=$(jq -r '.target_resolution' "$config_file")

if [[ -z "$FF_API_KEY" ]]; then
    echo "Error: FF_API_KEY environment variable is not set." >&2
    echo "Set it with: export FF_API_KEY=\"your-key-here\"" >&2
    exit 1
fi
ff_api_key="$FF_API_KEY"


# echo "Using config: $config_file"
# echo "  project_folder=$project_folder"
# echo "  site_name=$site"


# python create_treelist.py "$project_folder" "$plot_folder" "$site_name" "$als_treelist" "$avg_tpa" "$add_trees" "$ff_api_key"
# echo "Created treelist for site $site in $project_folder"
# python voxelize_ff.py "$project_folder" "$site_name" "$als_treelist" "$own_treelist" "$zipname" "$ff_api_key" "$target_resolution"
# echo "Voxelized trees for site $site in $project_folder"

# unzip -o "$project_folder/$zipname" -d "$project_folder"/data

# python generate_shrubs.py "$project_folder" "$plot_folder" "$site_name" "$shrub_folder" "$intelimon_plots_csv" 
# echo "Generated shrubs for site $site in $project_folder"

# if [[ "$plot_folder" != 'NA' ]]; then
#     python field_data_fuel_loading.py "$project_folder" "$plot_folder" "$site_name" "$ee_project" "$field_fuel_load_dat" "$field_fuel_depth_dat"
#     echo "Processed field data for site $site in $project_folder"
# else
#     echo "No field data provided, skipping field data processing."
# fi

# python voxelize_shrubs.py "$project_folder" "$site_name" "$site" "$field_fuel_load_dat" "$field_fuel_depth_dat"
# echo "Voxelized shrubs for site $site in $project_folder"

# #save adjusted files to rename them
# cp "$project_folder/data/treesfueldepth_adj.dat" "$project_folder/data/treesfueldepth.dat"
# cp "$project_folder/data/treesrhof_adj.dat" "$project_folder/data/treesrhof.dat"

python fill_dat_to_netcdf.py "$project_folder" "$site"
echo "Saved voxelized data to NetCDF for site $site in $project_folder"