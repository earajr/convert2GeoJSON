#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# onset
#python main.py --input_file /home/earajr/ECMWF_s2s/2025_testdata/forecast_status_20250601.nc --output_dir . --var_name onset_state --source 3day_onset --contour_thresholds -0.5 0.5 1.5 2.5 3.5 --contour_method category

python main.py --input_file /home/earajr/ECMWF_s2s/2025_testdata/onset_status_20250601.nc --output_dir . --var_name onset_day_of_year --source 3day_onset --contour_thresholds 24.5 29.5 34.5 39.5 44.5 49.5 54.5 59.5 64.5 69.5 74.5 79.5 84.5 89.5 94.5 99.5 104.5 109.5 114.5 119.5 124.5 129.5 134.5 139.5 144.5 149.5 154.5 159.5 164.5 169.5 174.5 179.5 184.5 189.5 194.5 199.5 204.5 209.5 214.5 219.5 224.5 229.5 --contour_method category

python main.py --input_file /home/earajr/ECMWF_s2s/2025_testdata/forecast_status_20250601.nc --output_dir . --var_name onset_day_of_year --source 3day_onset --contour_thresholds 24.5 29.5 34.5 39.5 44.5 49.5 54.5 59.5 64.5 69.5 74.5 79.5 84.5 89.5 94.5 99.5 104.5 109.5 114.5 119.5 124.5 129.5 134.5 139.5 144.5 149.5 154.5 159.5 164.5 169.5 174.5 179.5 184.5 189.5 194.5 199.5 204.5 209.5 214.5 219.5 224.5 229.5 --contour_method category


