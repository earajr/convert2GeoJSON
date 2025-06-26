#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON_new

cd ../convert2GeoJSON

# create job file
job_list="../scripts/job_list.txt"
if [ -f ${job_list} ]
then
   rm -rf ${job_list}
fi
touch ${job_list}

# Loop over files
for fil in ../testdata/WRF/cf_compliant/*
do
    # Loop over heigh levels (feet)
    for lev in 1000 2000 3000 4000 5000 10000 15000 20000 25000 30000 35000
    do
        echo "python main.py --input_file ${fil} --output_dir ../testdata_output/WRF_cf_compliant --var_name tc --source WRFhybridz --level ${lev} --level_units ft --contour_start -60.0 --contour_stop 40.0 --interval 2.0" >> ${job_list}
	echo "python main.py --input_file ${fil} --output_dir ../testdata_output/WRF_cf_compliant --var_name rh --source WRFhybridz --level ${lev} --level_units ft --contour_start 0.0 --contour_stop 100.0 --interval 5.0" >> ${job_list}

    done

    # Loop over pressure levels
    for lev in 1000 925 850 700 600 500 400 300 200
    do
        echo "python main.py --input_file ${fil} --output_dir ../testdata_output/WRF_cf_compliant --var_name tc --source WRFhybridp --level ${lev} --level_units hPa --contour_start -60.0 --contour_stop 40.0 --interval 2.0" >> ${job_list}
	echo "python main.py --input_file ${fil} --output_dir ../testdata_output/WRF_cf_compliant --var_name tc --source WRFhybridp --level ${lev} --level_units hPa --contour_start 0.0 --contour_stop 100.0 --interval 5.0" >> ${job_list}
    done
done

