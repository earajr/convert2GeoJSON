#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

# MTG_LI_config.yml can be found in the convert2GeoJSON directory and controls the accumulation period in minutes and the subset region for which lightning will geojsons will be generated. This is currently set to 5 minutes and Africa. Pregenerated masks are available for all African countries in the convert2GeoJSON/masks directory. They have a 50 km buffer as standard.

# Accumulated flash area example must use a file with the AFA tag as the input file and a var_name of accumulated_flash_area
python main.py --input_file ../testdata/MTG_LI/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AFA--FD--CHK-BODY---NC4E_C_EUMT_20250615192651_L2PF_OPE_20250615192600_20250615192630_N__O_0117_0013.nc --output_dir ../testdata_output/MTG_LI/. --var_name accumulated_flash_area --source MTG_LI_ACC --contour_thresholds 0.5 1.5 --smooth --sigma 1 --simplify --parallel

# Accumulated flash example must use a file with the AF tag as the input file and var_name of flash_accumulation
#python main.py --input_file ../testdata/MTG_LI/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AF--FD--CHK-BODY---NC4E_C_EUMT_20250615192651_L2PF_OPE_20250615192600_20250615192630_N__O_0117_0013.nc --output_dir ../testdata_output/MTG_LI/. --var_name flash_accumulation --source MTG_LI_ACC --contour_thresholds 1.0 5.0 10.0 50.0 100.0 500.0 --smooth --sigma 1 --simplify --parallel

# Accumulated flash radiance example must use a file with the AFR tag as the input file and var_name of flash_radiance
#python main.py --input_file ../testdata/MTG_LI/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AFR--FD--CHK-BODY---NC4E_C_EUMT_20250615192651_L2PF_OPE_20250615192600_20250615192630_N__O_0117_0013.nc --output_dir ../testdata_output/MTG_LI/. --var_name flash_radiance --source MTG_LI_ACC --contour_thresholds 500.0 1000.0 5000.0 10000.0 50000.0 --smooth --sigma 1 --simplify --parallel
