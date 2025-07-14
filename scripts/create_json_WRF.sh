#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

#This set the output directory to the the processed folder and todays date.
script_dir="/home/force-nwr/scripts/wrf-to-geojson"
job_list=${script_dir}"/geojson_job_list.txt"
indir="/home/force-nwr/nwr/uk/data/$(date +'%Y%m%d'00)"
output_dir="/home/force-nwr/nwr/uk/processed-wrf-data/$(date +'%Y%m%d'00)"

mkdir $output_dir

if [ -f ${job_list} ] ;
then
    rm -rf ${job_list}
fi
touch ${job_list}

declare -a vars=("rh" "tc" "mdbz" "dbz" "T2" "rh2")
declare -a var_source=("WRF3dp" "WRF3dp" "WRF2d" "WRF3dp" "WRF2d" "WRF2d")
declare -a levs=("925" "900" "850" "800" "750" "700" "600" "500" "350" "200")

for ((i=0; i<${#vars[@]}; i++))
do
    for fil in ${indir}/wrfout_d0*
    do
	if [ ${var_source[i]} == "WRF3dp" ]
        then
	    for lev in "${levs[@]}"
            do
	        if [ ${vars[i]} == "rh" ];
		then
		    thresh="0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105"
		elif [ ${vars[i]} == "tc" ];
		then
		    thresh="-20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 50"
		elif [ ${vars[i]} == "dbz" ];
		then
		    thresh="0.0 10.0 20.0 30.0 40.0 50.0 60.0 200.0"
                fi

		echo "/home/force-nwr/micromamba/envs/convert2GeoJSON/bin/python3 ${script_dir}/convert2GeoJSON/convert2GeoJSON/main.py --input_file ${fil} --output_dir ${output_dir} --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh} --level ${lev} --simplify" >> ${job_list}

            done
        elif [ ${var_source[i]} == "WRF2d" ]
        then
	    if [ ${vars[i]} == "mdbz" ];
	    then
	        thresh="0.0 10.0 20.0 30.0 40.0 50.0 60.0 200.0"
	    elif [ ${vars[i]} == "rh2" ];
	    then
                thresh="0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105"
	    elif [ ${vars[i]} == "T2" ];
            then
		thresh="-20 -18 -16 -14 -12 -10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 50"
	    fi
		
            echo "/home/force-nwr/micromamba/envs/convert2GeoJSON/bin/python3 ${script_dir}/convert2GeoJSON/convert2GeoJSON/main.py --input_file ${fil} --output_dir ${output_dir} --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh} --simplify" >> ${job_list}
        fi
    done
done

ulimit -s unlimited

parallel --sshloginfile /home/force-nwr/scripts/wrf-to-geojson/machines.txt -j 50 < ${job_list}

#parallel --sshloginfile machines.txt --sshdelay 0.1 -j 120 < ${job_list}

