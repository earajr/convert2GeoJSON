import numpy as np
import utils
import contouring_par

def read_datafile(source, file_path, var, contour_dict, level_dict, max_workers):
    """
    Decide on which reader to use based on source variable and extract data
 
    Parameters
    ----------
    source : str
        The source of the data (which model, observation type etc)
    file_path : str
        path to the input file.
    var : str
        Name of variable to be read
 
    Returns
    -------
    data_dict: dictionary
        Dictionary of variable values, latitudes, longitudes and metadata
 
    """
    data_readers = {
        'WRF2d': read_data_from_wrf2d,
        'WRF3dp': read_data_from_wrf3dp,
        'WRF3dh': read_data_from_wrf3dh,
        'WRFhybridp': read_data_from_hybrid_vert_wrf_p,
        'WRFhybridz': read_data_from_hybrid_vert_wrf_z,
        'HYSPLIT': read_data_from_HYSPLIT,
        'CRR': read_data_from_CRR,
        'NCASradar': read_data_from_NCASradar,
        'MTG_LI_ACC': read_data_from_MTG_LI_ACC,
        'UM3dh': read_UM_data_h
        # Add more mappings as needed
    }

    if source not in data_readers:
        raise ValueError(f"Unknown data source: {source}")

    data_reader_func = data_readers[source]

    if source == 'CRR':
        return data_reader_func(file_path, var, contour_dict, level_dict, max_workers=max_workers)
    else:
        return data_reader_func(file_path, var, contour_dict, level_dict)

def read_data_from_wrf2d(file_path, var, contour_dict, level_dict):
    """
    Get data from WRF netcdf input file
 
    Parameters
    ----------
    file_path : str
        Input file path
    var : str
        NetCDF variable to extract
    contour_dict: dictionary
        Information about the contour levels that have (or haven't) been supplied as arguments
 
    Returns
    -------
    data_dict: dictionary
        Dictionary of variable values, latitudes, longitudes and metadata
 
    """
    from netCDF4 import Dataset
    from wrf import (getvar, ALL_TIMES, latlon_coords, extract_times, extract_global_attrs, is_standard_wrf_var)

    # Read input data
    wrf_in = Dataset(file_path, "r")

    times = extract_times(wrf_in, ALL_TIMES)
    num_times = len(times)

    data_dict = {}

    for i in np.arange(0, num_times, 1):

        if var == "T2":
            data = getvar(wrf_in, var, timeidx=i)-273.15
            units = "degC"
        else:
            data = getvar(wrf_in, var, timeidx=i)
            if i == 0:
                if is_standard_wrf_var(wrf_in, var):
                    units = wrf_in.variables[var].units
                else:
                    units = data.units

        if i == 0:
            lat, lon = latlon_coords(data)
            grid_id = str(int(extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']))
            sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')['SIMULATION_START_DATE']
            sim_start_time = sim_start_time.replace('_', 'T')
            valid_time = str(extract_times(wrf_in, ALL_TIMES)[0])[0:22]
            dx = float(extract_global_attrs(wrf_in, 'DX')['DX'])
            dx_units = "m"
            level_type = "Single"
            if dx < 20000:
                if dx >= 1000:
                    rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1000.0)
                else:
                    rec_sigma = 5.0
            else:
                rec_sigma = 1.5

        # Max and min values in data
        max_int_data = np.ceil(np.nanmax(data))
        min_int_data = np.floor(np.nanmin(data))

       # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict

def read_data_from_wrf3dp(file_path, var, contour_dict, level_dict):
    """
    Get data from WRF netcdf input file
 
    Parameters
    ----------
    file_path : str
        Input file path
    var : str
        NetCDF variable to extract
    contour_dict : dictionary
        Dictionary that provides information on contours
    level_dict : dictionary
        Dictionary that provides information on pressure level
 
    Returns
    -------
    list
        List comprising of data, latitudes, longitudes and file
        metadata
 
    """
    from netCDF4 import Dataset
    from wrf import (getvar, ALL_TIMES, latlon_coords, extract_times, extract_global_attrs, is_standard_wrf_var, interplevel)

    # Read input data
    wrf_in = Dataset(file_path, "r")

    times = extract_times(wrf_in, ALL_TIMES)
    num_times = len(times)

    data_dict = {}

    for i in np.arange(0, num_times, 1):

        data_temp_3d = getvar(wrf_in, var, timeidx=i)
        p = getvar(wrf_in, "pressure", timeidx=i)

        data = interplevel(data_temp_3d, p, level_dict["level_id"])

        if i == 0:
            if is_standard_wrf_var(wrf_in, var):
                units = wrf_in.variables[var].units
            else:
                units = data.units

        if i == 0:
            lat, lon = latlon_coords(data)
            grid_id = str(int(extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']))
            sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')['SIMULATION_START_DATE']
            sim_start_time = sim_start_time.replace('_', 'T')
            valid_time = str(extract_times(wrf_in, ALL_TIMES)[0])[0:22]
            dx = float(extract_global_attrs(wrf_in, 'DX')['DX'])
            dx_units = "m"
            level_type = "P"+level_dict["level_id"]
            if dx < 20000:
                if dx >= 1000:
                    rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1000.0)
                else:
                    rec_sigma = 5.0
            else:
                rec_sigma = 1.5

        # Max and min values in data
        max_int_data = np.ceil(np.nanmax(data))
        min_int_data = np.floor(np.nanmin(data))

        # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)
        
        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict


def read_data_from_wrf3dh(file_path, var, contour_dict, level_dict):
    """
    Get data from WRF netcdf input file
 
    Parameters
    ----------
    input_file : str
        Input file path
    nc_var : str
        NetCDF variable to extract
    contour_dict : dictionary
        Dictionary providing information on contour levels
    level_dict : dictionary
        Dictionary providing information on height level
 
    Returns
    -------
    list
        List comprising of data, latitudes, longitudes and file
        metadata
 
    """
    from netCDF4 import Dataset
    from wrf import (getvar, ALL_TIMES, latlon_coords, extract_times, extract_global_attrs, is_standard_wrf_var, interplevel)

    # Read input data
    wrf_in = Dataset(file_path, "r")

    times = extract_times(wrf_in, ALL_TIMES)
    num_times = len(times)

    data_dict = {}

    for i in np.arange(0, num_times, 1):

        data_temp_3d = getvar(wrf_in, var, timeidx=i)
        z = getvar(wrf_in, "z", timeidx=i, units="m")

        data = interplevel(data_temp_3d, z, level_dict["level_id"])

        if i == 0:
            if is_standard_wrf_var(wrf_in, var):
                units = wrf_in.variables[var].units
            else:
                units = data.units

        if i == 0:
            lat, lon = latlon_coords(data)
            grid_id = str(int(extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID']))
            sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')['SIMULATION_START_DATE']
            sim_start_time = sim_start_time.replace('_', 'T')
            valid_time = str(extract_times(wrf_in, ALL_TIMES)[0])[0:22]
            dx = float(extract_global_attrs(wrf_in, 'DX')['DX'])
            dx_units = "m"
            level_type = "H"+level_dict["level_id"]
            if dx < 20000:
                if dx >= 1000:
                    rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1000.0)
                else:
                    rec_sigma = 5.0
            else:
                rec_sigma = 1.5

       # Max and min values in data
        max_int_data = np.ceil(np.nanmax(data))
        min_int_data = np.floor(np.nanmin(data))

        # Define LEVELS and THRESHOLDS (not actual max min values, just to set approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict

def read_data_from_hybrid_vert_wrf_z(file_path, var, contour_dict, level_dict):
    """
    Get data from FORCE hybrid pressure level WRF netcdf input file
    WORK IN PROGRESS
 
    Parameters
    ----------
    input_file : str
        Input file path
    nc_var : str
        NetCDF variable to extract
    contour_dict : dictionary
        Dictionary providing information on contour levels
    level_dict : dictionary
        Dictionary providing information on height level
 
    Returns
    -------
    list
        List comprising of data, latitudes, longitudes and file
        metadata
 
    """
    from netCDF4 import Dataset, num2date
    from wrf import interplevel
    import cftime

    if level_dict["units"] == "ft" or "feet":
        level_m = float(level_dict["level_id"])*0.3048
    else:
        level_m = float(level_dict["level_id"])
        level_dict["units"] == "m"

    # Read input data
    wrf_in = Dataset(file_path, "r")
    times = wrf_in.variables["time"]
    dtimes = num2date(times[:], times.units)
    num_times = np.shape(dtimes)[0]

    data_dict = {}

    lat = wrf_in.variables["latitude"][:]
    lon = wrf_in.variables["longitude"][:]

    data_temp_3d = wrf_in.variables[var][:]
    z = wrf_in.variables['z'][:]

    data_temp_level = interplevel(data_temp_3d, z, level_m)

    for i in np.arange(0, num_times, 1):

        data = data_temp_level[i].values

        if i == 0:
            grid_id = "unknown"
            sim_start_time = "unknown"
            sim_start_time = sim_start_time.replace('_', 'T')
            level_type = "H"+level_dict["level_id"]
            units = wrf_in.variables[var].units
        
            lat_diff = np.abs((lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)] - lat[int(np.shape(lat)[0]/2)+1,int(np.shape(lat)[1]/2)]) * 110.948)
            lon_diff = np.abs((lon[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)] - lon[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)+1]) * 110.948 * np.cos(np.deg2rad(lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)])))

            dx = float(round(((lat_diff + lon_diff)/2.0)*10.0)/10.0)
            dx_units = "km"

            if dx < 20000:
                if dx >= 1000:
                    rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1000.0)
                else:
                    rec_sigma = 5.0
            else:
                rec_sigma = 1.5

        valid_time = dtimes[i].isoformat()

       # Max and min values in data
        max_int_data = np.ceil(np.nanmax(data))
        min_int_data = np.floor(np.nanmin(data))

        # Define LEVELS and THRESHOLDS (not actual max min values, just to set approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

        entry_name = f"entry{i:03d}"

        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'level_units': level_dict["units"], 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : int(dx), 'grid_units': dx_units}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict

def read_data_from_hybrid_vert_wrf_p(file_path, var, contour_dict, level_dict):
    """
    Get data from FORCE hybrid pressure level WRF netcdf input file
    WORK IN PROGRESS

    Parameters
    ----------
    input_file : str
        Input file path
    nc_var : str
        NetCDF variable to extract
    contour_dict : dictionary
        Dictionary providing information on contour levels
    level_dict : dictionary
        Dictionary providing information on height level

    Returns
    -------
    list
        List comprising of data, latitudes, longitudes and file
        metadata

    """
    from netCDF4 import Dataset, num2date
    from wrf import interplevel
    import cftime

    # Read input data
    wrf_in = Dataset(file_path, "r")
    times = wrf_in.variables["time"]
    dtimes = num2date(times[:], times.units)
    num_times = np.shape(dtimes)[0]

    data_dict = {}

    a = wrf_in.variables["a"][:]
    a_reshaped = a.reshape(1,-1,1,1)
    b = wrf_in.variables["b"][:]
    b_reshaped = b.reshape(1,-1,1,1)
    ps = wrf_in.variables["ps"][:]
    ps_reshaped = ps.reshape(ps.shape[0],1,ps.shape[1],ps.shape[2])

    lat = wrf_in.variables["latitude"][:]
    lon = wrf_in.variables["longitude"][:]

    data_temp_3d = wrf_in.variables[var][:]
    pres = a_reshaped + (b_reshaped*ps_reshaped)

    data_temp_level = interplevel(data_temp_3d, pres, level_dict["level_id"])

    for i in np.arange(0, num_times, 1):

        data = data_temp_level[i].values

        if i == 0:
            grid_id = "unknown"
            sim_start_time = "unknown"
            sim_start_time = sim_start_time.replace('_', 'T')
            level_type = "P"+level_dict["level_id"]
            units = wrf_in.variables[var].units

            lat_diff = np.abs((lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)] - lat[int(np.shape(lat)[0]/2)+1,int(np.shape(lat)[1]/2)]) * 110.948)
            lon_diff = np.abs((lon[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)] - lon[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)+1]) * 110.948 * np.cos(np.deg2rad(lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)])))

            dx = float(round(((lat_diff + lon_diff)/2.0)*10.0)/10.0)
            dx_units = "km"

            if dx < 20000:
                if dx >= 1000:
                    rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1000.0)
                else:
                    rec_sigma = 5.0
            else:
                rec_sigma = 1.5

        valid_time = dtimes[i].isoformat()

       # Max and min values in data
        max_int_data = np.ceil(np.nanmax(data))
        min_int_data = np.floor(np.nanmin(data))

        # Define LEVELS and THRESHOLDS (not actual max min values, just to set approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

        entry_name = f"entry{i:03d}"

        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'level_units': "hPa", 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : int(dx), 'grid_units': dx_units}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict


def read_data_from_HYSPLIT(file_path, var, contour_dict, level_dict):
    """
    Get concentration data from HYSPLIT netcdf input file
 
    Parameters
    ----------
    file_path : str
        Input file path
    var : str
        NetCDF variable to extract
    contour_dict: dictionary
        Information about the contour levels that have (or haven't) been supplied as arguments
    level_dict : dictionary
        Information about the level
 
    Returns
    -------
    data_dict: dictionary
        Dictionary of variable values, latitudes, longitudes and metadata
 
    """

    from netCDF4 import Dataset
    from netCDF4 import num2date
    from datetime import datetime

    # create data dictionary 
    data_dict = {}

    # Read input data
    HYSPLIT_in= Dataset(file_path, "r")

    # Read times
    times = HYSPLIT_in.variables["time"][:]
    time_units = HYSPLIT_in.variables["time"].units
    times_str = [ num2date(time, units=time_units).strftime('%Y-%m-%d_%H:%M:%S') for time in times ]

    # Read levels
    levels = HYSPLIT_in.variables["levels"][:]
    level_units = HYSPLIT_in.variables["levels"].units
    levels_str = []
    for i, lev in enumerate(levels):
        if i == 0:
            levels_str.append("0_"+str(lev))
        else:
            levels_str.append(str(levels[i-1])+"_"+str(lev))

    # Read origin lat, lons, levels and times
    olats = HYSPLIT_in.variables["olat"][:].tolist()
    olons = HYSPLIT_in.variables["olon"][:].tolist()
    olvls = HYSPLIT_in.variables["olvl"][:].tolist()
    otims_temp = HYSPLIT_in.variables["otim"][:].tolist()
    otims = [datetime.strptime(str(time_int), "%y%m%d%H%M").strftime('%Y-%m-%d_%H:%M:%S') for time_int in otims_temp]

    # Calculate start of simulation
    sim_start_time = min([datetime.strptime(str(time_int), "%y%m%d%H%M").strftime('%Y-%m-%d_%H:%M:%S') for time_int in otims_temp])

    # Read in data 
    data_temp4d = HYSPLIT_in.variables[var][:,:,:,:]

    lat_1d = HYSPLIT_in.variables['latitude'][:]
    lon_1d = HYSPLIT_in.variables['longitude'][:]

    dx = float(np.abs(lat_1d[1] - lat_1d[0]))
    dx_units = "degrees"
    if dx < 0.2:
        if dx >= 0.01:
            rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/0.01)
        else:
            rec_sigma = 5.0
    else:
        rec_sigma = 1.5

    lat, lon = np.meshgrid(lat_1d, lon_1d)
    lat = lat.transpose()
    lon = lon.transpose()

    # Max and min values in data
    max_int_data = np.nanmax(data_temp4d)
    min_int_data = np.nanmin(data_temp4d)

    # Define LEVELS and THRESHOLDS (not actual max min values, just to set approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    count = 0

    for i in np.arange(0, len(times), 1):
        for j in np.arange(0, len(levels), 1):

            level_type = "H"+levels_str[j]
            valid_time = times_str[i]
            units = "undefined"

            entry_name = f"entry{count:03d}"

            data = data_temp4d[i,j,:,:]

            data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'origin_lats' : olats, 'origin_lons' : olons, 'origin_levels' : olvls, 'origin_times' : otims}}

            count = count + 1

    HYSPLIT_in.close()

    return data_dict

def read_data_from_CRR(file_path, var, contour_dict, level_dict, max_workers):
    """
    Get concentration data from CRR netcdf input file
 
    Parameters
    ----------
    file_path : str
        Input file path
    var : str
        NetCDF variable to extract
    contour_dict: dictionary
        Information about the contour levels that have (or haven't) been supplied as arguments
    level_dict : dictionary
        Information about the level
 
    Returns
    -------
    data_dict: dictionary
        Dictionary of variable values, latitudes, longitudes and metadata
 
    """

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    from scipy.spatial import ConvexHull
    from skimage import measure
    from shapely.geometry import mapping
    import yaml
    import os

    default_region = "Africa"
    default_buffer = 500

    try:
        # Read the CRR_config.yml file to retrieve region to be processed.
        with open('CRR_config.yml', 'r') as file:
            config = yaml.safe_load(file)
            region = config['CRR_reader_config']['region']
            buffer = config['CRR_reader_config']['buffer']
    except:
        region = default_region
        buffer = default_buffer

    # create data dictionary 
    data_dict = {}

    # Read input data
    CRR_in= Dataset(file_path, "r")

    if CRR_in.product_name == "CRR":
        # Read times
        nominal_product_time = datetime.strptime(CRR_in.nominal_product_time, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d_%H:%M:%S")
        time_coverage_start = datetime.strptime(CRR_in.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d_%H:%M:%S")
        time_coverage_end = datetime.strptime(CRR_in.time_coverage_end, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d_%H:%M:%S")
    elif CRR_in.product_name == "EXIM":
        #Read times
        nominal_product_time = (datetime.strptime(CRR_in.nominal_product_time.split("_")[0], "%Y-%m-%dT%H:%M:%SZ")+timedelta(minutes=int(CRR_in.nominal_product_time.split("_")[1]))).strftime("%Y-%m-%d_%H:%M:%S")
        time_coverage_start = (datetime.strptime(CRR_in.time_coverage_start.split("_")[0], "%Y-%m-%dT%H:%M:%SZ")+timedelta(minutes=int(CRR_in.time_coverage_start.split("_")[1]))).strftime("%Y-%m-%d_%H:%M:%S")
        time_coverage_end = (datetime.strptime(CRR_in.time_coverage_end.split("_")[0], "%Y-%m-%dT%H:%M:%SZ")+timedelta(minutes=int(CRR_in.time_coverage_end.split("_")[1]))).strftime("%Y-%m-%d_%H:%M:%S")

    # Read in lat and lon values
    if "lat" in CRR_in.variables.keys():
        lat = CRR_in.variables["lat"][:]
    if "lon" in CRR_in.variables.keys():
        lon = CRR_in.variables["lon"][:]
    if lat is None or lon is None:
        raise ValueError("Could not get latitude/longitude from input file")
    
    # Get or create mask

    mask = utils.get_or_create_region_mask(region, lat, lon, "CRR", buffer_km=buffer)

    # Read in data
    data = CRR_in.variables[var][:,:]
    data = np.where(mask, data, 0)

    # Should be no need to loop over times or levels as CRR data files are always 1 per satellite image on a single level.
    try:
        units = CRR_in.variables[var].units
    except:
        units = "undefined"

    satellite_id = CRR_in.satellite_identifier
    region_id = CRR_in.region_id
    level_type = "Single"
    dx = float(CRR_in.spatial_resolution)
    dx_units = "km"

    if dx < 20:
        if dx >= 1:
            rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1.0)
        else:
            rec_sigma = 5.0
    else:
        rec_sigma = 1.5

    # Max and min values in data
    max_int_data = np.ceil(np.nanmax(data))
    min_int_data = np.floor(np.nanmin(data))

    # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    try:
        # Check if any mandatory satellite data is missing
        crr_conditions = CRR_in.variables["crr_conditions"][:,:]
        idx = np.where((crr_conditions > 8900) & (crr_conditions < 9020), 1, 0)
        missing_sat_data = any([len(i)>0 for i in idx])
    except IndexError as e:
        missing_sat_data = False

    if missing_sat_data:
        missing_data_feature = contouring_par.create_missing_data_feature(idx, lat, lon, max_workers)
    else:
        missing_data_feature = None

    data[data>50.0] = 0.0

    entry_name = "entry000" #each CRR file only contains a single time/level so there is only ever 1 entry
    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': satellite_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : nominal_product_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units}}
    # Add missing data, if available:
    if missing_data_feature:
        data_dict[entry_name]['metadata']['missing_data'] = missing_data_feature

    # Close wrf_in file
    CRR_in.close()

    return data_dict

def read_data_from_NCASradar(file_path, var, contour_dict, level_dict):
    """
    Get horizontal data from NCAS mobile radars where data has been written to cfradial format
 
    Parameters
    ----------
    file_path : str
        Input file path
    var : str
        NetCDF variable to extract (dBZ is expected)
    contour_dict: dictionary
        Information about the contour levels that have (or haven't) been supplied as arguments
    level_dict : dictionary
        Information about the level
 
    Returns
    -------
    data_dict: dictionary
        Dictionary of variable values, latitudes, longitudes and metadata
 
    """

    import pyart
    from datetime import datetime
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt

    # create data dictionary
    data_dict = {}

    # Read input data
    # Might be that it is possible to add multiple files here in the future. A network of radars could be mapped onto the same grid allowing for a combined radar product (An overarching radar reader might be needed if the radar files are of a differnt file type (.h5 for example))
    radar_in= Dataset(file_path, "r")

    # Read times
    time_coverage_start = datetime.strptime(radar_in.time_coverage_start, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d_%H:%M:%S")
    time_coverage_end = datetime.strptime(radar_in.time_coverage_end, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d_%H:%M:%S")

    # Read in radar location
    site_lat = float(radar_in.variables["latitude"][:])
    site_lon = float(radar_in.variables["longitude"][:])
    site_alt = float(radar_in.variables["altitude"][:])
    units = radar_in.variables[var].units

    level_type = "H"+level_dict["level_id"]

    level_alt = float(level_dict["level_id"]) - site_alt

    level_alt_base = level_alt - 100.0 # These are just filler values at the moment need to check with Lindsey, Neely et al to see if a 200 m deep level is reasonable. 
    level_alt_top = level_alt + 100.0

    # Read range of radar
    max_range = np.nanmax(radar_in.variables["range"][:])
    mag_limit = 10**(np.ceil(np.log10(np.abs(max_range))))
    mag_limit_defecit_percent = np.floor((mag_limit-max_range)/10**(np.ceil(np.log10(np.abs(max_range)))-2))
    mag_limit_defecit = mag_limit*(mag_limit_defecit_percent/100.0)
    max_range_rounded = mag_limit-mag_limit_defecit

    dx = (max_range_rounded*2.0)/301.0
    dx_units = "m"

    rec_sigma = 1.0

    radar = pyart.io.read(file_path)

    grid_id = "pyart_gridded"
    grid = pyart.map.grid_from_radars((radar,), grid_shape=(1,301,301), grid_limits=((level_alt_base, level_alt_top), ((-1 * max_range_rounded), max_range_rounded), ((-1 * max_range_rounded), max_range_rounded)), gridding_algo = "map_gates_to_grid")

    lat = grid.get_point_longitude_latitude()[1]
    lon = grid.get_point_longitude_latitude()[0]

    data = grid.fields[var]["data"][0,:,:]

    entry_name = "entry000" # as the level is supplied on calling the function only a single 2d field is created
    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : time_coverage_start, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, "site_lat" : site_lat, "site_lon" : site_lon }}

    # Close radar_in file
    radar_in.close()

    return data_dict

def process_single_MTG_LI_file_with_args(args):
    """
    Worker function to read data from MTG)LI files. Due to frequency of file production (1 every 30 seconds) many files a needed to be read
    for a single accumulation period. Readin the files in parallel allows for some speedup.

    Parameters
    ----------
    fil : str
        Input file path
    var : str
        NetCDF variable to extract
    mask: array
        region of interest mask so that only information that is required is gathered.

    Returns
    -------
    result: array
        data from a single file to be combined with other files being read in parallel.

    """

    from satpy import Scene
    import numpy as np

    fil, var, mask = args

    scn = Scene(filenames=[fil], reader="li_l2_nc")
    scn.load([var])

    scn_values = np.where(mask, scn[var].values, 0)
    
    if var == "accumulated_flash_area":
        valid_mask = np.isfinite(scn_values) & (scn_values > 0)
        result = np.zeros_like(scn_values)
        result[valid_mask] = 1
    else:
        valid_mask = np.isfinite(scn_values)
        result = np.zeros_like(scn_values)
        result[valid_mask] = scn_values[valid_mask]

    return result

def read_data_from_MTG_LI_ACC(file_path, var, contour_dict, level_dict):
    """
    Get horizontal data from Meteosat Third Generation Lightning Imager accumulated/gridded products.
 
    Parameters
    ----------
    file_path : string
        path of file to be read in (latest of the accumulation period).
    var : str
        NetCDF variable to extract ("flash_radiance", "flash_accumulation", "accumulated_flash_area")
    contour_dict: dictionary
        Information about the contour levels that have (or haven't) been supplied as arguments
    level_dict : dictionary
        Information about the level
 
    Returns
    -------
    data_dict: dictionary
        Dictionary of variable values, latitudes, longitudes and metadata
 
    """

    from satpy.scene import Scene
    from satpy import find_files_and_readers
    import yaml
    import os
    import datetime
    import geopandas as gpd
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Polygon, MultiPolygon
    from matplotlib.path import Path
    import matplotlib.pyplot as plt
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import ThreadPoolExecutor

    default_region = "Africa"
    default_buffer = 500

    try:
        # Read the MTG_LI_config.yml file to retrieve region to be processed.
        with open('MTG_LI_config.yml', 'r') as file:
            config = yaml.safe_load(file)
            region = config['MTG_LI_reader_config']['region']
            buffer = config['MTG_LI_reader_config']['buffer']
    except:
        region = default_region
        buffer = default_buffer

    # create data dictionary
    data_dict = {}

    # Read the MTG_LI_config.yml file to retrieve the accumulation period in minutes and the limits of the box to be processed. A box that is
    with open('MTG_LI_config.yml', 'r') as file:
        config = yaml.safe_load(file)

    accum_mins = int(config['MTG_LI_reader_config']['accum_mins'])
    region = config['MTG_LI_reader_config']['region']

    # Using the file name work out the start and end times
    file_basename = os.path.basename(file_path)
    diri = os.path.dirname(file_path)

    latest_datetime = file_basename.split("_")[8]

    end_time = datetime.datetime(int(latest_datetime[0:4]), int(latest_datetime[4:6]), int(latest_datetime[6:8]), int(latest_datetime[8:10]), int(latest_datetime[10:12]), int(latest_datetime[12:14])) - datetime.timedelta(seconds=1)
    time_coverage_end = end_time.strftime("%Y-%m-%d_%H:%M:%S")
    start_time = datetime.datetime(int(latest_datetime[0:4]), int(latest_datetime[4:6]), int(latest_datetime[6:8]), int(latest_datetime[8:10]), int(latest_datetime[10:12]), int(latest_datetime[12:14])) - datetime.timedelta(minutes=int(accum_mins)) + datetime.timedelta(seconds=1)
    time_coverage_start = start_time.strftime("%Y-%m-%d_%H:%M:%S")

    if var == "accumulated_flash_area":
        var_str = "-AFA--"
    if var == "flash_accumulation":
        var_str = "-AF--"
    if var == "flash_radiance":
        var_str = "-AFR--"
        
    my_files_temp = find_files_and_readers(base_dir=diri, start_time=start_time, end_time=end_time, reader="li_l2_nc")["li_l2_nc"]
    my_files = []
    for fil in my_files_temp:
        if var_str in fil:
           my_files.append(fil)
 
    expected_num_files = accum_mins * 2
    if len(my_files) == expected_num_files:
        print("All expected files are present!")
    else:
        print("Warning: The number of files read does not match the expected number.")

    # Gather important information from the first listed file to allow proper processing of the rest
    initial_scn = Scene(filenames=[my_files[0]], reader="li_l2_nc")
    initial_scn.load([var])
    lat_lon = np.array(initial_scn[var].attrs['area'].get_lonlats())
    satellite_id = initial_scn[var].attrs['grid_mapping']
    full_lat = lat_lon[1]
    full_lon = lat_lon[0]

    lat_diff = np.abs((full_lat[int(np.shape(full_lat)[0]/2),int(np.shape(full_lat)[1]/2)] - full_lat[int(np.shape(full_lat)[0]/2),int(np.shape(full_lat)[1]/2)+1]) * 110.948)
    lon_diff = np.abs((full_lon[int(np.shape(full_lat)[0]/2),int(np.shape(full_lat)[1]/2)] - full_lon[int(np.shape(full_lat)[0]/2)+1,int(np.shape(full_lat)[1]/2)]) * 110.948 * np.cos(np.deg2rad(full_lat[int(np.shape(full_lat)[0]/2),int(np.shape(full_lat)[1]/2)])))

    dx = round(((lat_diff + lon_diff)/2.0)*10.0)/10.0
    dx_units = "km"

    if dx < 20:
        if dx >= 1:
            rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1.0)
        else:
            rec_sigma = 5.0
    else:
        rec_sigma = 1.5

    data = np.zeros_like(full_lat)
    mask = utils.get_or_create_region_mask(region, full_lat, full_lon, "MTG_LI", buffer_km=buffer)

    # Launch parallel processing to read all the files in the my_files list
    with ThreadPoolExecutor(max_workers=3) as executor:
        args_list = [(f, var, mask) for f in my_files]
        results = list(executor.map(process_single_MTG_LI_file_with_args, args_list))

    # Depending on the variable being read accumulate the read variable properly
    if var == "accumulated_flash_area":
        for result in results:
            valid_mask = np.isfinite(result) & (result > 0)
            data[valid_mask] = 1
            units = "no units"
    else:
        for result in results:
            valid_mask = np.isfinite(result)
            data[valid_mask] += result[valid_mask]
            if var == "flash_radiance":
                units = "mW.m-2.sr-1"
            elif var == "flash_accumulation":
                units = "flashes/pixel"

    lat = full_lat
    lon = full_lon

    entry_name = "entry000"

    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type' : 'Single', 'grid_id': satellite_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : time_coverage_end, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units}}

    return data_dict

def read_UM_data_h(file_path, var, contour_dict, level_dict):
    """
    Use cfpython to read UM files and interpolate to a specific height level
    WORK IN PROGRESS
 
    Parameters
    ----------
    file_path : str
        Input file path
    var : str
        variable to extract (if reading UM data then this is the stash code)
    contour_dict : dictionary
        Dictionary that provides information on contours
    level_dict : dictionary
        Dictionary that provides information on pressure level
 
    Returns
    -------
    list
        List comprising of data, latitudes, longitudes and file
        metadata
 
    """

    import cf
    import numpy as np
    from wrf import interplevel
    import iris.coord_systems as cs
    import iris.analysis.cartography as cart

    # create data dictionary
    data_dict = {}

    f = cf.read(file_path)

    for i, field in enumerate(f):
        try:
            std_name = field.properties()["standard_name"]
        except:
            std_name = ""
        field_var = [identity for identity in field.identities() if "ncvar%" in identity][0].split("%")[1]
        if std_name == var or field_var == var:
            index = i

    var_data = f[index]
    units = "kg/kg"
    stash_code = var_data.get_property("stash_code")
    um_stash_source = var_data.get_property("um_stash_source")

    try:
        long_name = var_data.get_property("long_name")
    except:
        long_name = ""

    try:
        standard_name = var_data.get_property("standard_name")
    except:
        standard_name = ""

    level_type = "H"+str(level_dict["level_id"])

    rotated_pole = False
    hybrid_height = False

    try:
        dim_coord_keys = list(var_data.dimension_coordinates().keys())
    except:
        dim_coord_keys = []

    for dim_coord_key in dim_coord_keys:
        dim_coord = var_data.dimension_coordinates()[dim_coord_key]
        dim_coord_str = str(var_data.dimension_coordinates()[dim_coord_key])

        if ((dim_coord.properties().get("standard_name") or dim_coord.properties().get("long_name")) == ("grid_latitude" or "grid_longitude")) and rotated_pole == False:
            rotated_pole = True
            print("Rotated pole identified. Trying to extract rotated pole information from coordinate references.")

            try:
                coord_ref_keys = list(var_data.coordinate_references().keys())

                for coord_ref_key in coord_ref_keys:
                    coord = var_data.coordinate_references()[coord_ref_key]
                    coord_str = str(var_data.coordinate_references()[coord_ref_key])
                    # Identify if data has been produced on a rotated lat lon grid and read in the required details for generating true lat and lon values
                    if coord_str.split(":")[1] == 'rotated_latitude_longitude':
                        grid_pole_lon = var_data.coordinate_references()[coord_ref_key]['grid_north_pole_longitude']
                        grid_pole_lat = var_data.coordinate_references()[coord_ref_key]['grid_north_pole_latitude']
                        n_pole_grid_lon = var_data.coordinate_references()[coord_ref_key]['north_pole_grid_longitude']

            except:
                print(f"Error: Unable to find rotated pole coordinate references.")
                sys.exit(1)

        if ((dim_coord.properties().get("standard_name") or dim_coord.properties().get("long_name")) == ("latitude" or "lat")):
            latitude = field.dimension_coordinate(dim_coord_key).array

        if ((dim_coord.properties().get("standard_name") or dim_coord.properties().get("long_name")) == ("longitude" or "lon")):
            longitude = field.dimension_coordinate(dim_coord_key).array

        if ((dim_coord.properties().get("standard_name") or dim_coord.properties().get("long_name")) == ("grid_latitude")):
            grid_latitude = field.dimension_coordinate(dim_coord_key).array

        if ((dim_coord.properties().get("standard_name") or dim_coord.properties().get("long_name")) == ("grid_longitude")):
            grid_longitude = field.dimension_coordinate(dim_coord_key).array

        if ((dim_coord.properties().get("standard_name") or coord.properties().get("long_name")) == "atmosphere_hybrid_height_coordinate" or (dim_coord.properties().get("standard_name") or coord.properties().get("long_name")) == "model_level_number"):

            hybrid_height = True
            coord_ref_keys = list(var_data.coordinate_references().keys())

            if coord_ref_keys:

                for coord_ref_key in coord_ref_keys:
                    coord_ref = var_data.coordinate_references()[coord_ref_key]
                    coord_ref_str = str(var_data.coordinate_references()[coord_ref_key])
                       # Identify if data has been produced on a hybrid height vertical coordinate and read in the required details for generating true lat and lon values
                    if coord_ref_str.split(":")[1] == 'atmosphere_hybrid_height_coordinate' or coord_ref_str.split(":")[1] == 'model_level_number':

                        print("Hybrid height coordinate reference identified. Trying to extract hybrid height information form coordinate references.")
                        coord_conversion = coord_ref.coordinate_conversion

                        a_key = coord_conversion.get_domain_ancillary('a')
                        b_key = coord_conversion.get_domain_ancillary('b')
                        orog_key = coord_conversion.get_domain_ancillary('orog')

                        a = field.constructs[a_key].array
                        b = field.constructs[b_key].array

                        orog = field.constructs[orog_key].array

                        break

            else:
                # Extract auxiliary coordinates: hybrid height coefficients a and b
                a, b, orog = None, None, None
                for aux_id, aux in var_data.auxiliary_coordinates().items():
                    long_name = aux.properties().get("long_name", "").lower()
                    standard_name = aux.properties().get("standard_name", "").lower()
                    if long_name == "height based hybrid coeffient a" or standard_name == "atmosphere_hybrid_height_coordinate":
                        a = var_data.construct(aux_id).array
                    elif long_name == "height based hybrid coeffient b" or long_name == "sigma":
                        b = var_data.construct(aux_id).array
                    elif long_name == "surface_altitude" or standard_name == "surface_altitude":
                        orog = var_data.construct(aux_id).array

                if a is None or b is None:
                    raise ValueError("Could not find auxiliary coordinates for a or b.")

                if orog is None:
                    print("The orography data is missing and needs to be read from an additional file")


# At this stage there should be a lat and lon array (or a grid lat and lon and approriate aux values to convert to the real grid lat and lon).


    if rotated_pole:
        x, y = np.meshgrid(grid_longitude, grid_latitude)

        lon_temp, lat_temp = cart.unrotate_pole(x, y, grid_pole_lon, grid_pole_lat)

    else:

        lat_1d = var_data.dimension_coordinate('latitude').array
        lon_1d = var_data.dimension_coordinate('longitude').array

        if np.any(lon_1d > 180):
            lon_1d = (lon_1d - 180) % 360 - 180
            sort_order = np.argsort(lon_1d)
            lon_1d = lon_1d[sort_order]

            var_data = var_data[:, :, :, sort_order]


        lat_temp, lon_temp = np.meshgrid(lat_1d, lon_1d)
        lat_temp = lat_temp.transpose()
        lon_temp = lon_temp.transpose()

        # THIS NEEDS EDITING - THE WAY TO DO IT IS TO IGNORE ALL THESE OPTIONS AND JUST REGRID FROM THE MASTER, A CHECK SHOULD BE PERFORMED TO IDENTIFY IF A TEMP OROG FILE IS ALREADY PRESENT AND MAKE SURE THAT THE LAT LON VALUES ARE THE SAME AS IN THE REGRIDDED FILE. iF NOT THEN THE EXISTING TEMP SHOULD BE DELETED AND REPLACED WITH A NEWLY GENERATED OROG FILE.
        if orog is None:
            print("identify grid that is being used")
            if len(longitude) == 96 and len(latitude) == 72:
                grid = "N48"
                orog_fil = "./UM_orog_files/N48_orog"
            if len(longitude) == 192 and len(latitude) == 144:
                grid = "N96"
#                orog_fil = "./UM_orog_files/N96_orog"
                orog_fil = "./cfpython_testdata/33_orography.nc"
            if len(longitude) == 432 and len(latitude) == 324:
                grid = "N216"
                orog_fil = "./UM_orog_files/N216_orog"
            if len(longitude) == 640 and len(latitude) == 480:
                grid = "N320"
                orog_fil = "./UM_orog_files/N320_orog"
            if len(longitude) == 1024 and len(latitude) == 768:
                grid = "N512"
                orog_fil = "./UM_orog_files/N512_orog"
            if len(longitude) == 1536 and len(latitude) == 1152:
                grid = "N768"
                orog_fil = "./UM_orog_files/N768_orog"
            if len(longitude) == 2560 and len(latitude) == 1920:
                grid = "N1280"
                orog_fil = "./UM_orog_files/N1280_orog"

            grid_id = grid

            if orog_fil:
                orog_f = cf.read(orog_fil)

                for i, field in enumerate(orog_f):
                    stash_code = field.properties()["um_stash_source"]
                    if stash_code == "m01s00i033":
                        index = i
                        break

                orog_data = orog_f[index]
                orog = orog_data.data.array
            else:
                print("The approriate orog file is not available, need to generate a orog field from either master orog file or appropriately spaced global")

    # calculate dx
    dx = np.abs(lat_1d[int(len(lat_1d)/2)] - lat_1d[int((len(lat_1d)/2))+1])
    dx_units = "degrees"

    rec_sigma = 1.5

    if hybrid_height and a.any() and b.any() and orog.any():
        a_reshaped = a.reshape(-1, 1, 1)  # Shape (85, 1, 1)
        b_reshaped = b.reshape(-1, 1, 1)  # Shape (85, 1, 1)

        # Calculate 3D height field
        z = cf.Data(a_reshaped + b_reshaped * orog, units="m")


    import yaml
    import os

    try:
        if sort_order > 0:
            z = z[ :, :, sort_order]
        else:
            z = z
    except:
        z = z

    if os.path.exists("UM_zoom_config.yml"):
        with open("UM_zoom_config.yml", "r") as f:
            config = yaml.safe_load(f).get("UM_zoom_config", None)
        ll_lat = config.get("ll_lat")
        ll_lon = config.get("ll_lon")
        ur_lat = config.get("ur_lat")
        ur_lon = config.get("ur_lon")

        lat_mask = (lat_temp >= ll_lat) & (lat_temp <= ur_lat)
        lon_mask = (lon_temp >= ll_lon) & (lon_temp <= ur_lon)
        combined_mask = lat_mask & lon_mask

        lat_indices = np.where(np.any(combined_mask, axis=1))[0]
        lon_indices = np.where(np.any(combined_mask, axis=0))[0]

        subset_var_data = var_data[:, :, lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]
        subset_lat_temp = lat_temp[lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]
        subset_lon_temp = lon_temp[lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]
        subset_z = z[:,lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]

        var_data = subset_var_data
        lat = subset_lat_temp
        lon = subset_lon_temp
        z = subset_z


    # Interpolate onto height levels

    var_array = var_data.data.array
    z_array = z.data.array

    # Get all dimension coordinates and their corresponding domain axes
    dim_coords = var_data.dimension_coordinates()
    data_axes = var_data.get_data_axes()

    # Loop through dimension coordinates to find X and Y axes
    for coord_key, coord in dim_coords.items():
        # Check the axis attribute
        axis = coord.get_property("axis", None)
        axis_key = var_data.domain_axis(coord_key, key=True)
        index = data_axes.index(axis_key)

        if axis == "T":
            time_index = index
        if axis == "X":
            x_index = index
        if axis == "Y":
            y_index = index

    time = var_data.dimension_coordinate("time").datetime_array
    time_str = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time]

    data_all = np.zeros((np.shape(var_array)[time_index], 1, np.shape(var_array)[y_index], np.shape(var_array)[x_index]), dtype=float)

    for i in np.arange(0, np.shape(var_array)[time_index], 1):
        data_all[i,0,:,:] = interplevel(var_array[i,:,:,:], z_array[:,:,:], level_dict["level_id"])

    # Max and min values in data
    max_int_data = np.ceil(np.nanmax(data))
    min_int_data = np.floor(np.nanmin(data))

    # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    # Add additional frame around where data is present and populate with lat and lon values using finite difference approach
    
    for i in np.arange(0, np.shape(var_array)[time_index], 1):

        data = data_all[i,0,:,:]

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'valid_time': time_str[i], 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units}}

    return data_dict


