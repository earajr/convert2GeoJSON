#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate convert2GeoJSON

start_date="20250201"
end_date="20250831"

current="$start_date"

while [ "$current" -le "$end_date" ]; do
    echo "$current"  # replace with your command

    out_dir="/home/earajr/ECMWF_s2s/2025_Ghana_onset/"${current}
    
    if [ ! -d ${out_dir} ]
    then
        mkdir -p ${out_dir} 
    fi
    
    python /home/earajr/ECMWF_s2s/convert2GeoJSON/convert2GeoJSON/main.py --input_file /home/earajr/ECMWF_s2s/3DAY_2025_testdata/onset_status_${current}.nc --output_dir ${out_dir} --var_name onset_state --source 3day_onset --contour_thresholds -0.5 0.5 1.5 2.5 3.5 --contour_names 1 2 3 --contour_method category
    python /home/earajr/ECMWF_s2s/convert2GeoJSON/convert2GeoJSON/main.py --input_file /home/earajr/ECMWF_s2s/3DAY_2025_testdata/onset_status_${current}.nc --output_dir ${out_dir} --var_name rain_days_ago --source 3day_onset --contour_thresholds -0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 --contour_names 1 2 3 4 5 6 7 8 9 10 --contour_method category
    python /home/earajr/ECMWF_s2s/convert2GeoJSON/convert2GeoJSON/main.py --input_file /home/earajr/ECMWF_s2s/3DAY_2025_testdata/onset_status_${current}.nc --output_dir ${out_dir} --var_name onset_day_of_year --source 3day_onset --contour_thresholds 24.5 29.5 34.5 39.5 44.5 49.5 54.5 59.5 64.5 69.5 74.5 79.5 84.5 89.5 94.5 99.5 104.5 109.5 114.5 119.5 124.5 129.5 134.5 139.5 144.5 149.5 154.5 159.5 164.5 169.5 174.5 179.5 184.5 189.5 194.5 199.5 204.5 209.5 214.5 219.5 224.5 229.5 --contour_names 25_29 30_34 35_39 40_44 45_49 50_54 55_59 60_64 65_69 70_74 75_79 80_84 85_89 90_94 95_99 100_104 105_109 110_114 115_119 120_124 125_129 130_134 135_139 140_144 145_149 150_154 155_159 160_164 165_169 170_174 175_179 180_184 185_189 190_194 195_199 200_204 205_209 210_214 215_219 220_224 225_229 --contour_method category

    python /home/earajr/ECMWF_s2s/convert2GeoJSON/convert2GeoJSON/main.py --input_file /home/earajr/ECMWF_s2s/3DAY_2025_testdata/forecast_status_${current}.nc --output_dir ${out_dir} --var_name onset_state --source 3day_onset --contour_thresholds -0.5 0.5 1.5 2.5 3.5 --contour_names 1 2 3 --contour_method category
    python /home/earajr/ECMWF_s2s/convert2GeoJSON/convert2GeoJSON/main.py --input_file /home/earajr/ECMWF_s2s/3DAY_2025_testdata/forecast_status_${current}.nc --output_dir ${out_dir} --var_name rain_days_ago --source 3day_onset --contour_thresholds -0.5 0.5 1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5 9.5 10.5 --contour_names 1 2 3 4 5 6 7 8 9 10 --contour_method category
    python /home/earajr/ECMWF_s2s/convert2GeoJSON/convert2GeoJSON/main.py --input_file /home/earajr/ECMWF_s2s/3DAY_2025_testdata/forecast_status_${current}.nc --output_dir ${out_dir} --var_name onset_day_of_year --source 3day_onset --contour_thresholds 24.5 29.5 34.5 39.5 44.5 49.5 54.5 59.5 64.5 69.5 74.5 79.5 84.5 89.5 94.5 99.5 104.5 109.5 114.5 119.5 124.5 129.5 134.5 139.5 144.5 149.5 154.5 159.5 164.5 169.5 174.5 179.5 184.5 189.5 194.5 199.5 204.5 209.5 214.5 219.5 224.5 229.5 --contour_names 25_29 30_34 35_39 40_44 45_49 50_54 55_59 60_64 65_69 70_74 75_79 80_84 85_89 90_94 95_99 100_104 105_109 110_114 115_119 120_124 125_129 130_134 135_139 140_144 145_149 150_154 155_159 160_164 165_169 170_174 175_179 180_184 185_189 190_194 195_199 200_204 205_209 210_214 215_219 220_224 225_229 --contour_method category
    
    current=$(date -d "$current + 1 day" +"%Y%m%d")
done


## onset
##python main.py --input_file /home/earajr/ECMWF_s2s/2025_testdata/forecast_status_20250601.nc --output_dir . --var_name onset_state --source 3day_onset --contour_thresholds -0.5 0.5 1.5 2.5 3.5 --contour_method category
#
#python main.py --input_file /home/earajr/ECMWF_s2s/2025_testdata/onset_status_20250601.nc --output_dir . --var_name onset_day_of_year --source 3day_onset --contour_thresholds 24.5 29.5 34.5 39.5 44.5 49.5 54.5 59.5 64.5 69.5 74.5 79.5 84.5 89.5 94.5 99.5 104.5 109.5 114.5 119.5 124.5 129.5 134.5 139.5 144.5 149.5 154.5 159.5 164.5 169.5 174.5 179.5 184.5 189.5 194.5 199.5 204.5 209.5 214.5 219.5 224.5 229.5 --contour_method category
#
#python main.py --input_file /home/earajr/ECMWF_s2s/2025_testdata/forecast_status_20250601.nc --output_dir . --var_name onset_day_of_year --source 3day_onset --contour_thresholds 24.5 29.5 34.5 39.5 44.5 49.5 54.5 59.5 64.5 69.5 74.5 79.5 84.5 89.5 94.5 99.5 104.5 109.5 114.5 119.5 124.5 129.5 134.5 139.5 144.5 149.5 154.5 159.5 164.5 169.5 174.5 179.5 184.5 189.5 194.5 199.5 204.5 209.5 214.5 219.5 224.5 229.5 --contour_method category


