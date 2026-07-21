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


echo "Using config: $config_file"
echo "  project_folder=$project_folder"
echo "  site_name=$site_name"


python create_domain.py "$project_folder"
echo "Created domain for site $site_name in $project_folder"

python search_stac.py "$project_folder"
echo "Searched STAC for $site_name data in $project_folder"

python ALS_to_treetops.py "$project_folder" "$zmax" "$target_resolution"
echo "Created treetops from ALS data for site $site_name in $project_folder"
 