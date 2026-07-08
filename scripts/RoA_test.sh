#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# CRR_config.yml can be found in the convert2GeoJSON directory and controls the region for which CRR geojsons will be generated. This is currently set to Africa with a buffer of 500 km. Pregenerated masks are available for all African countries in the convert2GeoJSON/masks directory. They have a 500 km buffer as standard.

# Standard contour levels provided here to allow for reproduction of standard CRR plots as they are produced using the NWCPY visualisation library
python main.py --input_file /home/earajr/temp/roa-retrieval/20260327/RoA_202603270000.nc --output_dir ../testdata_output/RoA/. --var_name precipitation --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --smooth --sigma 1 --simplify --parallel

#python main.py --input_file /home/earajr/temp/roa-extrapolation/20260327/RoA_202603270315_015.nc --output_dir ../testdata_output/RoA/. --var_name precipitation --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --smooth --sigma 1 --simplify --parallel

