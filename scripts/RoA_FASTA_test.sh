#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

cd ../convert2GeoJSON

for hh in "00" "01" "02" "03" "04" "05" "06"
do
    for min in "00" "15" "30" "45"
    do
        #python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/RoA_FASTA/roa-retrieval/20260626/RoA_202606260000.nc --output_dir ../testdata_output/RoA_FASTA/. --var_name precipitation --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --smooth --sigma 1 --simplify --parallel

        python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/RoA_FASTA/roa-retrieval/20260706/RoA_20260706${hh}${min}.nc --output_dir ../testdata_output/RoA_FASTA/. --var_name precipitation --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --simplify --parallel --smooth --sigma 1 

        #python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/RoA_FASTA/roa-retrieval/20260626/RoA_202606260000.nc --output_dir ../testdata_output/RoA_FASTA/. --var_name precipitation --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --smooth --sigma 1 --simplify

#       for extrap in "015" "030" "045" "060" "075" "090" "105" "120"
#	do
#	    python main.py --input_file /home/earajr/convert2GEOJSON/convert2GeoJSON/testdata/RoA_FASTA/roa-extrapolation/20260626/RoA_20260626${hh}${min}_${extrap}.nc --output_dir ../testdata_output/RoA_FASTA/. --var_name precipitation --source RoA --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0 --simplify --parallel
#        done
    done
done


