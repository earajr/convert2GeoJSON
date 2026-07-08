#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

#This set the output directory to the the processed folder and todays date.
script_dir="/home/earajr/convert2GEOJSON"
job_list=${script_dir}"/geojson_job_list.txt"
indir="/home/force-nwr/nwr/uk/data/$(date +'%Y%m%d'00)"
#output_dir="/home/force-nwr/nwr/uk/processed-wrf-data/$(date +'%Y%m%d'00)"
output_dir="/home/earajr/convert2GEOJSON/convert2GeoJSON/scripts/$(date +'%Y%m%d'00)"

mkdir $output_dir

if [ -f ${job_list} ] ;
then
    rm -rf ${job_list}
fi
touch ${job_list}

declare -a vars=("uvmet10")
declare -a var_source=("WRF2d")

for ((i=0; i<${#vars[@]}; i++))
do
    for fil in ${indir}/wrfout_d0*
    do
        if [ ${var_source[i]} == "WRF2d" ]
        then
            thresh="-50 -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 15 20 25 30 35 40 45 50"
		
            echo "/home/force-nwr/micromamba/envs/convert2GeoJSON/bin/python3 ${script_dir}/convert2GeoJSON/convert2GeoJSON/main.py --input_file ${fil} --output_dir ${output_dir} --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh} --simplify"
        fi
    done
done

