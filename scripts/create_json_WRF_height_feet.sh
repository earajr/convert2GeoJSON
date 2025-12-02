#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

script_dir="/home/earajr/convert2GEOJSON"
job_list=${script_dir}"/convert2GeoJSON/scripts/geojson_job_list.txt"
indir="/home/force-nwr/nwr/uk/data/2025071300"
output_dir="/home/earajr/convert2GEOJSON/convert2GeoJSON/testdata_output/WRF/2025071300"

mkdir -p $output_dir

if [ -f ${job_list} ] ;
then
    rm -rf ${job_list}
fi
touch ${job_list}

declare -a vars=("tc")
declare -a var_source=("WRF3dh")
declare -a lev_units=("feet")
declare -a levs=("5000" "10000" "20000" "30000")

for ((i=0; i<${#vars[@]}; i++))
do
    echo ${vars[i]}
    echo ${var_source[i]}
    echo ${lev_units[i]}
    for fil in ${indir}/wrfout_d0*
    do
        if [ ${var_source[i]} == "WRF3dh" ]
        then
           for lev in "${levs[@]}"
           do
	       if [ ${vars[i]} == "rh" ];
               then
                   thresh="0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105"
               elif [ ${vars[i]} == "tc" ];
               then
                   thresh="-40 -38 -36 -34 -32 -30 -28 -26 -24 -22 -20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 50"
               elif [ ${vars[i]} == "dbz" ];
               then
                   thresh="0.0 10.0 20.0 30.0 40.0 50.0 60.0 200.0"
               fi

	       echo "/home/force-nwr/micromamba/envs/convert2GeoJSON/bin/python3 ${script_dir}/convert2GeoJSON/convert2GeoJSON/main.py --input_file ${fil} --output_dir ${output_dir} --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh} --level_units ${lev_units[i]} --level ${lev} --simplify" >> ${job_list}
           done
       fi
   done
done

#ulimit -s unlimited
#
#parallel --sshloginfile /home/force-nwr/scripts/wrf-to-geojson/machines.txt -j 50 < ${job_list}
#
#parallel --sshloginfile machines.txt --sshdelay 0.1 -j 120 < ${job_list}

