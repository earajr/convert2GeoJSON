#!/bin/bash

source /home/earajr/anaconda3/etc/profile.d/conda.sh
conda activate convert2geojson

script_dir="/home/earajr/convert2GEOJSON"
job_list=${script_dir}"/geojson_job_list.txt"

indir=$1

if [ -f ${job_list} ] ;
then
    rm -rf ${job_list}
fi

touch ${job_list}

declare -a vars=("rh" "tc" "mdbz" "dbz" "T2" "rh2")
declare -a var_source=("WRF3dp" "WRF3dp" "WRF2d" "WRF3dp" "WRF2d" "WRF2d")
declare -a levs=("925" "900" "850" "800" "750" "700")

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

                echo "python convert2GEOJSON.py --input_file ${fil} --output_dir . --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh} --level ${lev}" >> ${job_list}
#                python convert2GEOJSON.py --input_file ${fil} --output_dir . --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh} --level ${lev}

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
		
            echo "python convert2GEOJSON.py --input_file ${fil} --output_dir . --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh}" >> ${job_list}
#            python convert2GEOJSON.py --input_file ${fil} --output_dir . --var_name ${vars[i]} --source ${var_source[i]} --contour_thresholds ${thresh}
        fi
    done
done

#parallel -j 20 < ${job_list}

#python convert2GEOJSON.py --input_file /home/force-nwr/nwr/uk/data/2024041000/wrfout_d02_2024-04-10_15\:00\:00 . --var_name dBZ --source NCASradar --contour_thresholds -20.0 -10.0 0.0 10.0 20.0 30.0 40.0 50.0 60.0 200.0 --level 2000 --smooth
