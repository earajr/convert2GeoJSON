#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# RoA_config.yml can be found in the convert2GeoJSON directory and controls the region for which RoA geojsons will be generated. This is currently set to Africa with a buffer of 500 km. Pregenerated masks are available for all African countries in the convert2GeoJSON/masks directory. They have a 500 km buffer as standard.

# This example generates RoA contours which match the CRR standard contour levels, this is not neccessarily the optimum metyhod but is a good starting point given the planned use for FASTA
python main.py --input_file ../testdata/RoA/ROA-2025-06-22T15\:00\:00.nc --output_dir ../testdata_output/RoA/. --var_name posterior_mean --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --parallel

