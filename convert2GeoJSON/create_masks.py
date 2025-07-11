import read_data
from utils import get_or_create_region_mask
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Code to allow for pregeneration of masks for different readers, there is a need to have the approriate SOURCE, file, variable etc input here to read some key features
# however once one file has been read then the mask generation happens in parallel (based on the list of mask in the masks directory).
# Masks like this should only be generated for datasets where the shape, size and position of the input data is fixed e.g. geostationary satellite retrievals. 

# CRR
#SOURCE = "CRR"
#input_file = "../../FASTA/S_NWC_CRR_MSG3_Africa-VISIR_20240317T163000Z.nc"
#VAR_NAME = "crr_intensity"
#contour_dict = {"contour_method":"standard", "contour_flag":True, "contours_set":True, "interval_flag":False, "contour_thresholds":[0.2, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 200.0], "colormap":"viridis"} # Mostly irrelevant but allows the read data function to work for the first file
#level_dict = {}
#parallel_dict = {"max_workers":10}

## MTG_LI
#SOURCE = "MTG_LI_ACC"
#input_file = "../../testdata/MTG_LI/W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AFA--FD--CHK-BODY---NC4E_C_EUMT_20250615231622_L2PF_OPE_20250615231530_20250615231600_N__O_0140_0012.nc"
#VAR_NAME = "accumulated_flash_area"
#contour_dict = {"contour_method":"standard", "contour_flag":True, "contours_set":True, "interval_flag":False, "contour_thresholds":[0.5, 1.5], "colormap":"viridis"} # Mostly irrelevant but allows the read data function to work for the first file
#level_dict = {}
#parallel_dict = {"max_workers":10}

# RoA
SOURCE = "RoA"
input_file = "../../testdata/RoA/ROA-2025-06-22T15:00:00.nc"
VAR_NAME = "posterior_mean"
contour_dict = {"contour_method":"standard", "contour_flag":True, "contours_set":True, "interval_flag":False, "contour_thresholds":[0.2, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 200.0], "colormap":"viridis"} # Mostly irrelevant but allows the read data function to work for the first file
level_dict = {}
parallel_dict = {"max_workers":10}

data = read_data.read_datafile(SOURCE, input_file, VAR_NAME, contour_dict, level_dict, parallel_dict["max_workers"])

lat = data['entry000']['lat']
lon = data['entry000']['lon']

buffer_dist = 500

def generate_mask(args):
    region_name, lat, lon, SOURCE = args
    try:
        mask = get_or_create_region_mask(region_name, lat, lon, SOURCE,  buffer_km=buffer_dist)
        return region_name, True
    except Exception as e:
        print(f"❌ Failed to create mask for {region_name}: {e}")
        return region_name, False

def main():

    with open("masks/mask_list.txt", "r") as f:
        region_list = [line.strip() for line in f if line.strip()]

    task_args = [(region, lat, lon, SOURCE) for region in region_list]

    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(generate_mask, task_args), total=len(task_args)))

    print("\nSummary:")
    for region, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {region}")

if __name__ == "__main__":
    main()

