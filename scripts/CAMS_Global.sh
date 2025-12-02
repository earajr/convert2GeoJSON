#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# SO2

# SO2 concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name so2_conc --source CAMS_global --contour_thresholds 0.0 0.177 0.316 0.562 1.0 1.77 3.16 5.62 10.0 17.7 31.6 56.2 100.0 177.0 316.0 562.0 1000.0 --simplify

# Volcanic SO2
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name VSO2 --source CAMS_global --contour_thresholds 0.0 0.00000000000177 0.00000000000316 0.00000000000562 0.00000000001 0.0000000000177 0.0000000000316 0.0000000000562 0.0000000001 0.000000000177 0.000000000316 0.000000000562 0.000000001 0.00000000177 0.00000000316 0.00000000562 0.00000001 0.0000000177 0.0000000316 0.0000000562 0.0000001 0.000000177 0.000000316 --simplify

# Volcanic SO2 concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name VSO2_conc --source CAMS_global --contour_thresholds 0.0 0.00177 0.00316 0.00562 0.01 0.0177 0.0316 0.0562 0.1 0.177 0.316 0.562 1.0 1.77 3.16 5.62 10.0 --simplify

# Carbon monoxide
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name co --source CAMS_global --contour_thresholds 0.0 0.00000001 0.0000000177 0.0000000316 0.0000000562 0.0000001 0.000000177 0.000000316 0.000000562 0.000001 0.00000177 0.00000316 0.00000562 0.00001 --simplify

# Carbon monoxide concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name co_conc --source CAMS_global --contour_thresholds 0.0 10.0 17.7 31.6 56.2 100.0 177.0 316.0 562.0 1000.0 1770.0 3160.0 5620.0 10000.0 --simplify

# NO2
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name no2 --source CAMS_global --contour_thresholds 0.0 0.0000000001 0.000000000177 0.000000000316 0.000000000562 0.000000001 0.00000000177 0.00000000316 0.00000000562 0.00000001 0.0000000177 0.0000000316 0.0000000562 0.0000001 --simplify

# NO2 concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name no2_conc --source CAMS_global --contour_thresholds 0.0 0.177 0.316 0.562 1.0 1.77 3.16 5.62 10.0 17.7 31.6 56.2 100.0 177.0 316.0 562.0 1000.0 --simplify

# NO
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name no --source CAMS_global --contour_thresholds 0.0 0.00000000001 0.0000000000177 0.0000000000316 0.0000000000562 0.0000000001 0.000000000177 0.000000000316 0.000000000562 0.000000001 0.00000000177 0.00000000316 0.00000000562 0.00000001 0.0000000177 0.0000000316 0.0000000562 0.0000001 --simplify

# NO concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name no_conc --source CAMS_global --contour_thresholds 0.0 0.00177 0.00316 0.00562 0.01 0.0177 0.0316 0.0562 0.1 0.177 0.316 0.562 1.0 1.77 3.16 5.62 10.0 17.7 31.6 56.2 100.0 177.0 562.0 --simplify

# NOx
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name nox --source CAMS_global --contour_thresholds 0.0 0.0000000001 0.000000000177 0.000000000316 0.000000000562 0.000000001 0.00000000177 0.00000000316 0.00000000562 0.00000001 0.0000000177 0.0000000316 0.0000000562 0.0000001 --simplify

# NOx concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name nox_conc --source CAMS_global --contour_thresholds 0.0 0.177 0.316 0.562 1.0 1.77 3.16 5.62 10.0 17.7 31.6 56.2 100.0 177.0 316.0 562.0 1000.0 --simplify

## Ozone
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name go3 --source CAMS_global --contour_thresholds 0.0 0.00000002 0.00000004 0.00000006 0.00000008 0.0000001 0.000000120 0.000000140 0.00000016 0.00000018 0.0000002 0.00000022 0.00000024 0.00000026 0.00000028 0.0000003 0.00000032 0.00000034 0.00000036 0.00000038 0.0000004 0.00000042 0.00000044 0.00000046 0.00000048 0.0000005 0.000001 --simplify

# Ozone concentration
#python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_1000.nc --output_dir ../testdata_output/CAMS_Global/. --var_name go3_conc --source CAMS_global --contour_thresholds 0.0 20.0 40.0 60.0 80.0 100.0 120.0 140.0 160.0 180.0 200.0 220.0 240.0 260.0 280.0 300.0 320.0 340.0 360.0 380.0 400.0 420.0 440.0 460.0 480.0 500.0 1000.0 --simplify

python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/CAMS_global/CAMS_Global_Europe_20250814_00_0_72_column.nc --output_dir ../testdata_output/CAMS_Global/. --var_name pm2p5 --source CAMS_global_2d --contour_thresholds 0.0 0.00000000000000177 0.00000000000000316 0.00000000000000562 0.00000000000001 0.0000000000000177 0.0000000000000316 0.0000000000000562 0.0000000000001 0.000000000000177 0.000000000000316 0.000000000000562 0.000000000001 0.00000000000177 0.00000000000316 0.00000000000562 0.00000000001 0.0000000000177 0.0000000000316 0.0000000000562 0.0000000001 0.000000000177 0.000000000316 0.000000000562 0.000000001 0.00000000177 0.00000000316 0.00000000562 0.00000001 --simplify


#pm1
#pm2p5
#pm10
#tcco
#tcno2
#tc_no
#gtco3
#tcso2
#tc_VSO2


#CAMS_global_2d
