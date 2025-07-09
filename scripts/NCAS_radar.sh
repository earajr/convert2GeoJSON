#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# Contour levels provided here show the reflectivity between -20.0 and 60.0 
python main.py --input_file ../testdata/NCAS_xband1/ncas-mobile-x-band-radar-1_lyneham_20230813-110423_SUR_v1.nc --output_dir ../testdata_output/NCAS_xband1/. --level 1000  --var_name dBZ --source NCASradar --contour_thresholds -20.0 -10.0 0.0 10.0 20.0 30.0 40.0 50.0 60.0 70.0 --smooth --sigma 1 --parallel

