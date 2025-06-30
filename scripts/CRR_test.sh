#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# CRR_config.yml can be found in the convert2GeoJSON directory and controls the region for which CRR geojsons will be generated. This is currently set to Africa with a buffer of 500 km. Pregenerated masks are available for all African countries in the convert2GeoJSON/masks directory. They have a 500 km buffer as standard.

# Accumulated flash area example must use a file with the AFA tag as the input file and a var_name of accumulated_flash_area
python main.py --input_file ../testdata/CRR/S_NWC_CRR_MSG3_Africa-VISIR_20240317T163000Z.nc --output_dir ../testdata_output/CRR/. --var_name crr_intensity --source CRR --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --smooth --sigma 1 --simplify --parallel

