import numpy as np
import utils

def read_datafile(source, file_path, var, contour_dict, level_dict):
    """
#   Decide on which reader to use based on source variable and extract data
#
#   Parameters
#   ----------
#   source : str
#       The source of the data (which model, observation type etc)
#   file_path : str
#       path to the input file.
#   var : str
#       Name of variable to be read
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
    """
    data_readers = {
        'WRF2d': read_data_from_wrf2d,
        'WRF3dp': read_data_from_wrf3dp,
        'WRF3dh': read_data_from_wrf3dh,
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

    return data_reader_func(file_path, var, contour_dict, level_dict)

def read_data_from_wrf2d(file_path, var, contour_dict, level_dict):
    """
#   Get data from WRF netcdf input file
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       NetCDF variable to extract
#   contour_dict: dictionary
#       Information about the contour levels that have (or haven't) been supplied as arguments
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
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
            data_temp = getvar(wrf_in, var, timeidx=i)-273.15
            units = "degC"
        else:
            data_temp = getvar(wrf_in, var, timeidx=i)
            if i == 0:
                if is_standard_wrf_var(wrf_in, var):
                    units = wrf_in.variables[var].units
                else:
                    units = data_temp.units

        if i == 0:
            lat_temp, lon_temp = latlon_coords(data_temp)
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
        max_int_data = np.ceil(np.nanmax(data_temp))
        min_int_data = np.floor(np.nanmin(data_temp))

       # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

        '''
        # Add additional frame around where data is present and populate with lat and lon values using finite difference approach

        lat = np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2))
        lat[1:-1,1:-1] = lat_temp
        lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
        lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
        lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
        lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])

        lon = np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2))
        lon[1:-1,1:-1] = lon_temp
        lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
        lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
        lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
        lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])

        # Fill variables in empty frame around data with the current lowest value in the dataset
        data_min = np.amin(data_temp)
        edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
        edge_min = np.amin(edge_list)
        data_step = THRESHOLDS[1]-THRESHOLDS[0]
        data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
        data[ data == 0.0 ] =  edge_min - data_step
        data[1:-1,1:-1] = data_temp
        '''

        lat = lat_temp
        lon = lon_temp
        data = data_temp

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict

def read_data_from_wrf3dp(file_path, var, contour_dict, level_dict):
    """
#   Get data from WRF netcdf input file
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       NetCDF variable to extract
#   contour_dict : dictionary
#       Dictionary that provides information on contours
#   level_dict : dictionary
#       Dictionary that provides information on pressure level
#
#   Returns
#   -------
#   list
#       List comprising of data, latitudes, longitudes and file
#       metadata
#
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

        data_temp = interplevel(data_temp_3d, p, level_dict["level_id"])

        if i == 0:
            if is_standard_wrf_var(wrf_in, var):
                units = wrf_in.variables[var].units
            else:
                units = data_temp.units

        if i == 0:
            lat_temp, lon_temp = latlon_coords(data_temp)
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
        max_int_data = np.ceil(np.nanmax(data_temp))
        min_int_data = np.floor(np.nanmin(data_temp))

        # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)
        
        '''
        # Add additional frame around where data is present and populate with lat and lon values using finite difference approach

        lat = np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2))
        lat[1:-1,1:-1] = lat_temp
        lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
        lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
        lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
        lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])

        lon = np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2))
        lon[1:-1,1:-1] = lon_temp
        lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
        lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
        lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
        lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])

        # Fill variables in empty frame around data with the current lowest value in the dataset
        data_min = np.amin(data_temp)
        edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
        edge_min = np.amin(edge_list)
        data_step = THRESHOLDS[1]-THRESHOLDS[0]
        data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
        data[ data == 0.0 ] =  edge_min - data_step
        data[1:-1,1:-1] = data_temp
        '''

        lat = lat_temp
        lon = lon_temp
        data = data_temp

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict


def read_data_from_wrf3dh(file_path, var, contour_dict, level_dict):
    """
#   Get data from WRF netcdf input file
#
#   Parameters
#   ----------
#   input_file : str
#       Input file path
#   nc_var : str
#       NetCDF variable to extract
#   contour_dict : dictionary
#       Dictionary providing information on contour levels
#   level_dict : dictionary
#       Dictionary providing information on height level
#
#   Returns
#   -------
#   list
#       List comprising of data, latitudes, longitudes and file
#       metadata
#
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

        data_temp = interplevel(data_temp_3d, z, level_dict["level_id"])

        if i == 0:
            if is_standard_wrf_var(wrf_in, var):
                units = wrf_in.variables[var].units
            else:
                units = data_temp.units

        if i == 0:
            lat_temp, lon_temp = latlon_coords(data_temp)
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
        max_int_data = np.ceil(np.nanmax(data_temp))
        min_int_data = np.floor(np.nanmin(data_temp))

        # Define LEVELS and THRESHOLDS (not actual max min values, just to set approriate values for data to be added to extra frame around data.
        CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

        # Add additional frame around where data is present and populate with lat and lon values using finite difference approach

        '''
        lat = np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2))
        lat[1:-1,1:-1] = lat_temp
        lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
        lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
        lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
        lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])

        lon = np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2))
        lon[1:-1,1:-1] = lon_temp
        lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
        lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
        lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
        lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])

        # Fill variables in empty frame around data with the current lowest value in the dataset
        data_min = np.amin(data_temp)
        edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
        edge_min = np.amin(edge_list)
        data_step = THRESHOLDS[1]-THRESHOLDS[0]
        data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
        data[ data == 0.0 ] =  edge_min - data_step
        data[1:-1,1:-1] = data_temp
        '''

        lat = lat_temp
        lon = lon_temp
        data = data_temp

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}

    # Close wrf_in file
    wrf_in.close()

    return data_dict

def read_data_from_HYSPLIT(file_path, var, contour_dict, level_dict):
    """
#   Get concentration data from HYSPLIT netcdf input file
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       NetCDF variable to extract
#   contour_dict: dictionary
#       Information about the contour levels that have (or haven't) been supplied as arguments
#   level_dict : dictionary
#       Information about the level
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
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

    lat_temp, lon_temp = np.meshgrid(lat_1d, lon_1d)
    lat_temp = lat_temp.transpose()
    lon_temp = lon_temp.transpose()

    # Max and min values in data
    max_int_data = np.nanmax(data_temp4d)
    min_int_data = np.nanmin(data_temp4d)

    # Define LEVELS and THRESHOLDS (not actual max min values, just to set approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    '''
    # Add additional frame around where data is present and populate with lat and lon values using finite difference approach
    lat = np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2))
    lat[1:-1,1:-1] = lat_temp
    lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
    lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
    lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
    lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])

    lon = np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2))
    lon[1:-1,1:-1] = lon_temp
    lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
    lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
    lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
    lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])
    '''

    lat = lat_temp
    lon = lon_temp

    count = 0

    for i in np.arange(0, len(times), 1):
        for j in np.arange(0, len(levels), 1):

            level_type = "H"+levels_str[j]
            valid_time = times_str[i]
            units = "undefined"

            entry_name = f"entry{count:03d}"

            data_temp = data_temp4d[i,j,:,:]

            '''
            # Fill variables in empty frame around data with the current lowest value in the dataset
            data_min = np.amin(data_temp)
            edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
            edge_min = np.amin(edge_list)
            data_step = THRESHOLDS[1]-THRESHOLDS[0]
            data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
            data[ data == 0.0 ] =  edge_min - data_step
            data[1:-1,1:-1] = data_temp
            '''

            data = data_temp

            data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : rec_sigma, 'origin_lats' : olats, 'origin_lons' : olons, 'origin_levels' : olvls, 'origin_times' : otims}}

            count = count + 1

    HYSPLIT_in.close()

    return data_dict

def read_data_from_CRR_backup(file_path, var, contour_dict, level_dict):
    """
#   Get concentration data from CRR netcdf input file
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       NetCDF variable to extract
#   contour_dict: dictionary
#       Information about the contour levels that have (or haven't) been supplied as arguments
#   level_dict : dictionary
#       Information about the level
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
    """

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    from scipy.spatial import ConvexHull
    from skimage import measure

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
        lat_temp = CRR_in.variables["lat"][:]
    if "lon" in CRR_in.variables.keys():
        lon_temp = CRR_in.variables["lon"][:]
    if lat_temp is None or lon_temp is None:
        raise ValueError("Could not get latitude/longitude from input file")
    
    
#    # update lat values:
#    lat = np.ma.array(
#        np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2)),
#        mask=False
#    )
#    lat[1:-1,1:-1] = lat_temp
#    lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
#    lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
#    lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
#    lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])
    lat_temp.mask[lat_temp < -90] = True

    lat = lat_temp

    # update lon values:
#    lon = np.ma.array(
#        np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2)),
#        mask=False
#    )
#    lon[1:-1,1:-1] = lon_temp
#    lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
#    lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
#    lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
#    lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])
    lon_temp.mask[lon_temp < -180] = True

    lon = lon_temp

    # Read in data
    data_temp = CRR_in.variables[var][:,:]

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
    max_int_data = np.ceil(np.nanmax(data_temp))
    min_int_data = np.floor(np.nanmin(data_temp))

    # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    try:
        # Check if any mandatory satellite data is missing
        crr_conditions = CRR_in.variables["crr_conditions"][:,:]
        idx_temp = np.where((crr_conditions > 8900) & (crr_conditions < 9020), 1, 0)
        # Fill in empty frame around data with zeros
        idx = np.zeros((np.shape(idx_temp)[0]+2, np.shape(idx_temp)[1]+2))
        idx[1:-1,1:-1] = idx_temp
        missing_contours = measure.find_contours(idx, level=0.5)
        missing_sat_data = any([len(i)>0 for i in idx])
    except IndexError as e:
        missing_sat_data = False

    if missing_sat_data:
        coords_list = []
        for i, ctr in enumerate(missing_contours):
            coords, out_of_bounds = get_contour_coords(ctr, lat, lon)
            coords_s = simplify_coords(coords, 0.25)
            coords_list.append(coords_s)
        missing_data_feature = create_feature("missing_data", coords_list)
    else:
        missing_data_feature = None

#    # Fill variables in empty frame around data with the current lowest value in the dataset
#    data_min = np.amin(data_temp)
#    edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
#    edge_min = np.amin(edge_list)
#    data_step = THRESHOLDS[1]-THRESHOLDS[0]
#    data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
#    data[ data == 0.0 ] =  edge_min - data_step
#    data[1:-1,1:-1] = data_temp
#    data[data>50.0] = 0.0

    data = data_temp
    data[data>50.0] = 0.0

    entry_name = "entry000" #each CRR file only contains a single time/level so there is only ever 1 entry
    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': satellite_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : nominal_product_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}
    # Add missing data, if available:
    if missing_data_feature:
        data_dict[entry_name]['metadata']['missing_data'] = missing_data_feature

    # Close wrf_in file
    CRR_in.close()

    return data_dict

def read_data_from_CRR(file_path, var, contour_dict, level_dict):
    """
#   Get concentration data from CRR netcdf input file
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       NetCDF variable to extract
#   contour_dict: dictionary
#       Information about the contour levels that have (or haven't) been supplied as arguments
#   level_dict : dictionary
#       Information about the level
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
    """

    from netCDF4 import Dataset
    from datetime import datetime, timedelta
    from scipy.spatial import ConvexHull
    from skimage import measure

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
        lat_temp = CRR_in.variables["lat"][:]
    if "lon" in CRR_in.variables.keys():
        lon_temp = CRR_in.variables["lon"][:]
    if lat_temp is None or lon_temp is None:
        raise ValueError("Could not get latitude/longitude from input file")

    # update lat values:
#    lat = np.ma.array(
#        np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2)),
#        mask=False
#    )
#    lat[1:-1,1:-1] = lat_temp
#    lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
#    lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
#    lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
#    lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])
    
#    lat_temp.mask[lat_temp < -90] = True
    lat = lat_temp

    # update lon values:
#    lon = np.ma.array(
#        np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2)),
#        mask=False
#    )
#    lon[1:-1,1:-1] = lon_temp
#    lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
#    lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
#    lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
#    lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])
#    lon.mask[lon < -180] = True

#    lon_temp.mask[lon_temp < -180] = True
    lon = lon_temp

    # Read in data
    data_temp = CRR_in.variables[var][:,:]

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
    max_int_data = np.ceil(np.nanmax(data_temp))
    min_int_data = np.floor(np.nanmin(data_temp))

    # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    missing_data_feature = None

#    # Fill variables in empty frame around data with the current lowest value in the dataset
#    data_min = np.amin(data_temp)
#    edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
#    edge_min = np.amin(edge_list)
#    data_step = THRESHOLDS[1]-THRESHOLDS[0]
#    data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
#    data[ data == 0.0 ] =  edge_min - data_step
#    data[1:-1,1:-1] = data_temp
#    data[data>50.0] = 0.0

    data = data_temp
    data[data>50.0] = 0.0


    entry_name = "entry000" #each CRR file only contains a single time/level so there is only ever 1 entry
    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': satellite_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : nominal_product_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}
    # Add missing data, if available:
    if missing_data_feature:
        data_dict[entry_name]['metadata']['missing_data'] = missing_data_feature

    # Close wrf_in file
    CRR_in.close()

    return data_dict


def read_data_from_NCASradar(file_path, var, contour_dict, level_dict):
    """
#   Get horizontal data from NCAS mobile radars where data has been written to cfradial format
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       NetCDF variable to extract (dBZ is expected)
#   contour_dict: dictionary
#       Information about the contour levels that have (or haven't) been supplied as arguments
#   level_dict : dictionary
#       Information about the level
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
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
    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : time_coverage_start, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma), "site_lat" : site_lat, "site_lon" : site_lon }}

    # Close radar_in file
    radar_in.close()

    return data_dict


def read_data_from_MTG_LI_ACC(file_path, var, contour_dict, level_dict):
    '''
#   Get horizontal data from Meteosat Third Generation Lightning Imager accumulated/gridded products.
#
#   Parameters
#   ----------
#   file_path : string
#       path of file to be read in (latest of the accumulation period).
#   var : str
#       NetCDF variable to extract ("flash_radiance", "flash_accumulation", "accumulated_flash_area")
#   contour_dict: dictionary
#       Information about the contour levels that have (or haven't) been supplied as arguments
#   level_dict : dictionary
#       Information about the level
#
#   Returns
#   -------
#   data_dict: dictionary
#       Dictionary of variable values, latitudes, longitudes and metadata
#
    '''

    from satpy.scene import Scene
    from satpy import find_files_and_readers
    import yaml
    import os
    import datetime

    # create data dictionary
    data_dict = {}

# Read the MTG_LI_config.yml file to retrieve the accumulation period in minutes and the limits of the box to be processed. A box that is
    with open('MTG_LI_config.yml', 'r') as file:
        config = yaml.safe_load(file)

    accum_mins = int(config['MTG_LI_reader_config']['accum_mins'])
    ll_lat = float(config['MTG_LI_reader_config']['ll_lat'])
    ll_lon = float(config['MTG_LI_reader_config']['ll_lon'])
    ur_lat = float(config['MTG_LI_reader_config']['ur_lat'])
    ur_lon = float(config['MTG_LI_reader_config']['ur_lon'])

# using the file name work out the start and end times
    file_basename = os.path.basename(file_path)
    diri = os.path.dirname(file_path)

    latest_datetime = file_basename.split("_")[8]

    end_time = datetime.datetime(int(latest_datetime[0:4]), int(latest_datetime[4:6]), int(latest_datetime[6:8]), int(latest_datetime[8:10]), int(latest_datetime[10:12]), int(latest_datetime[12:14])) - datetime.timedelta(seconds=1)
    time_coverage_end = end_time.strftime("%Y-%m-%d_%H:%M:%S")
    start_time = datetime.datetime(int(latest_datetime[0:4]), int(latest_datetime[4:6]), int(latest_datetime[6:8]), int(latest_datetime[8:10]), int(latest_datetime[10:12]), int(latest_datetime[12:14])) - datetime.timedelta(minutes=int(accum_mins)) + datetime.timedelta(seconds=1)
    time_coverage_start = start_time.strftime("%Y-%m-%d_%H:%M:%S")

    my_files = find_files_and_readers(base_dir=diri, start_time=start_time, end_time=end_time, reader="li_l2_nc")["li_l2_nc"]
    expected_num_files = accum_mins * 2
    if len(my_files) == expected_num_files:
        print("All expected files are present!")
    else:
        print("Warning: The number of files read does not match the expected number.")


    for i, fil in enumerate(my_files):
        scn = Scene(filenames=[fil], reader="li_l2_nc")
        scn.load([var])

        if i == 0:
            satellite_id = scn[var].attrs['grid_mapping']
            # get full file lat and lon values
            lat_lon = np.array(scn[var].attrs['area'].get_lonlats())
            full_lat = lat_lon[1]
            full_lon = lat_lon[0]

            for j in np.arange(0, np.shape(full_lat)[0], 1):
                lat_slice = full_lat[:,j]
                if np.all(np.isinf(lat_slice)):
                    lat_slice = np.linspace(85.0, -85.0, len(lat_slice), endpoint=True)
                    full_lat[:,j] = lat_slice
                else:
                    valid_indices = np.where(~np.isinf(lat_slice))[0]
                    inf_indices = np.where(np.isinf(lat_slice))[0]

                    lat_slice[0:np.min(valid_indices)] = np.linspace(85.0, lat_slice[np.min(valid_indices)], len(lat_slice[0:np.min(valid_indices)]), endpoint=False)
                    start_val = lat_slice[np.max(valid_indices)] + (lat_slice[np.max(valid_indices)] - lat_slice[np.max(valid_indices)-1])
                    lat_slice[np.max(valid_indices)+1:np.max(inf_indices)+1] = np.linspace(start_val, -85.0, len(lat_slice[np.max(valid_indices)+1:np.max(inf_indices)+1]), endpoint = True)
                    full_lat[:,j] = lat_slice

            for j in np.arange(0, np.shape(full_lat)[1], 1):
                lon_slice = full_lon[j,:]
                if np.all(np.isinf(lon_slice)):
                    lon_slice = np.linspace(-85.0, 85.0, len(lon_slice), endpoint=True)
                    full_lon[j,:] = lon_slice
                else:
                    valid_indices = np.where(~np.isinf(lon_slice))[0]
                    inf_indices = np.where(np.isinf(lon_slice))[0]

                    lon_slice[0:np.min(valid_indices)] = np.linspace(-85.0, lon_slice[np.min(valid_indices)], len(lon_slice[0:np.min(valid_indices)]), endpoint=False)
                    start_val = lon_slice[np.max(valid_indices)] + (lon_slice[np.max(valid_indices)] - lon_slice[np.max(valid_indices)-1])
                    lon_slice[np.max(valid_indices)+1:np.max(inf_indices)+1] = np.linspace(start_val, 85.0, len(lon_slice[np.max(valid_indices)+1:np.max(inf_indices)+1]), endpoint = True)
                    full_lon[j,:] = lon_slice

            # find indices of ll and ur corners
            ll_index1, ll_index2 = find_indices(ll_lat, ll_lon, full_lat, full_lon)
            ur_index1, ur_index2 = find_indices(ur_lat, ur_lon, full_lat, full_lon)
            # calculate index ranges
            index1_range = np.abs(ur_index1 - ll_index1) + 1
            index2_range = np.abs(ur_index2 - ll_index2) + 1
            # create flash data and subset lat lon arrays
            data = np.zeros((index1_range, index2_range))
            lat = full_lat[ur_index1:ll_index1+1, ll_index2:ur_index2+1]
            lon = full_lon[ur_index1:ll_index1+1, ll_index2:ur_index2+1]

            lat_diff = np.abs((lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)] - lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)+1]) * 110.948)
            lon_diff = np.abs((lon[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)] - lon[int(np.shape(lat)[0]/2)+1,int(np.shape(lat)[1]/2)]) * 110.948 * np.cos(np.deg2rad(lat[int(np.shape(lat)[0]/2),int(np.shape(lat)[1]/2)])))

            dx = round(((lat_diff + lon_diff)/2.0)*10.0)/10.0
            dx_units = "km"

            if dx < 20:
                if dx >= 1:
                    rec_sigma = 5.0+(9.0/38.0) - (9/38.0)*(dx/1.0)
                else:
                    rec_sigma = 5.0
            else:
                rec_sigma = 1.5

        scn_values = scn[var].values[ur_index1:ll_index1+1,ll_index2:ur_index2+1]
        if var == "accumulated_flash_area":
            valid_mask = np.isfinite(scn_values) & (scn_values > 0)
            data[valid_mask] = 1
            units = "no units"
        else:
            valid_mask = np.isfinite(scn_values)
            data[valid_mask] += scn_values[valid_mask]
            if var == "flash_radiance":
                units = "mW.m-2.sr-1"
            elif var == "flash_accumulation":
                units = "flashes/pixel"

    entry_name = "entry000"

    data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type' : 'Single', 'grid_id': satellite_id, 'time_coverage_start': time_coverage_start, 'time_coverage_end': time_coverage_end, 'nominal_product_time' : time_coverage_end, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma) }}

    return data_dict

def read_UM_data_h(file_path, var, contour_dict, level_dict):
    """
#   Use cfpython to read UM files and interpolate to a specific height level
#
#   Parameters
#   ----------
#   file_path : str
#       Input file path
#   var : str
#       variable to extract (if reading UM data then this is the stash code)
#   contour_dict : dictionary
#       Dictionary that provides information on contours
#   level_dict : dictionary
#       Dictionary that provides information on pressure level
#
#   Returns
#   -------
#   list
#       List comprising of data, latitudes, longitudes and file
#       metadata
#
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
        lat_temp = subset_lat_temp
        lon_temp = subset_lon_temp
        z = subset_z

        print(np.shape(var_data))
        print(np.shape(lat_temp))
        print(np.shape(lon_temp))
        print(np.shape(z))

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

    data_temp = np.zeros((np.shape(var_array)[time_index], 1, np.shape(var_array)[y_index], np.shape(var_array)[x_index]), dtype=float)

    for i in np.arange(0, np.shape(var_array)[time_index], 1):
        data_temp[i,0,:,:] = interplevel(var_array[i,:,:,:], z_array[:,:,:], level_dict["level_id"])

    # Max and min values in data
    max_int_data = np.ceil(np.nanmax(data_temp))
    min_int_data = np.floor(np.nanmin(data_temp))

    # Define LEVELS and THRESHOLDS (not actual max min values, just to et approriate values for data to be added to extra frame around data.
    CONTOURS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)

    # Add additional frame around where data is present and populate with lat and lon values using finite difference approach
    
    lat = lat_temp
    lon = lon_temp
    '''
    lat = np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2))
    lat[1:-1,1:-1] = lat_temp
    lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])/2
    lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])/2
    lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])/2
    lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])/2

    lon = np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2))
    lon[1:-1,1:-1] = lon_temp
    lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])/2
    lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])/2
    lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])/2
    lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])/2
    '''

    for i in np.arange(0, np.shape(var_array)[time_index], 1):
        '''
        # Fill variables in empty frame around data with the current lowest value in the dataset
        data_min = np.amin(data_temp)
        edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
        edge_min = np.amin(edge_list)
        data_step = THRESHOLDS[1]-THRESHOLDS[0]
        data = np.zeros((np.shape(data_temp)[2]+2, np.shape(data_temp)[3]+2))
        data[ data == 0.0 ] =  edge_min - data_step
        data[1:-1,1:-1] = data_temp[i,0,:,:]
        '''

        data = data_temp

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'valid_time': time_str[i], 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}

    return data_dict


