import numpy as np

def input_args():
    """
#    Parse and validate input arguments
#
#    Returns
#    -------
#    list
#        List of strings representing input_file, output_dir
#
#    """

    import os
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input_file', type=str, help="Path to inputfile")
    parser.add_argument('--output_dir', type=str, help="Path to output directory")
    parser.add_argument('--var_name', type=str, help="Name of variable to be read")
    parser.add_argument('--source', type=str, help="Source of the data e.g. CRR, WRF2d, WRF3dp, ...")
    parser.add_argument('--level', type=str, help="Level identifier for 3d data, e.g. for WRF3dp 800 for 800 hPa, all heights are in m")
    parser.add_argument('--level_units', type=str, help="Level units identifier, make sure that units make sense for levels e.g. m or ft for z or hPa for pressure levels")
    parser.add_argument('--contour_start', type=numeric_type, help='Start value for contours')
    parser.add_argument('--contour_stop', type=numeric_type, help='Stop value for contours')
    parser.add_argument('--interval', type=numeric_type, help='Level interval')
    parser.add_argument('--smooth', action='store_true', help='A flag to activate smoothing')
    parser.add_argument('--sigma', type=numeric_type, help='User defined sigma value to overrule suggested sigma')
    parser.add_argument('--colormap', type=str, help="Name of matplotlib colormap to generate colours for contours")
    parser.add_argument('--contour_thresholds', nargs="+", help="Define contours explicitly e.g --contour_thresholds 0.2 1.0 2.0 3.0 5.0 7.0 10.0 15.0 20.0 30.0 50.0 200.0")
    parser.add_argument('--contour_names', nargs="+", help="Define contour names explicitly e.g.--contour_names 0.2_1.0 1.0_2.0 2.0_3.0 3.0_5.0 5.0_7.0 7.0_10.0 10.0_15.0 15.0_20.0 20.0_30.0 30.0_50.0 50.0+")
    parser.add_argument('--colors', nargs="+", help="Define hex code colors explicitly, can only be used if contours have been explicitly set and must have 1 fewer values than the --contours argument (remove leading # from hex codes e.g. --colors 1e50d2 0294fe 00d2fc 448622 02c403 6dec02 fefe02 fdc102 ff5000 b31b26 7f7f7f")
    parser.add_argument('--contour_method', type=str, help='Choice of contour creation method e.g. standard or pixel')
    parser.add_argument('--parallel', action='store_true', help='A flag to parallel processing using tiling of data')
    parser.add_argument('--num_workers', type=numeric_type, help='Number of workers processing tiles in parallel')
    parser.add_argument('--simplify', action='store_true', help='A flag to denote whether polygons are to be simplified once created')
    parser.add_argument('--tolerance', type=numeric_type, help='Value to denote the level of polygon simplification default value of 0.005')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        err_msg = "Input file {0} does not exist\n"
        err_msg = err_msg.format(args.input_file)
        raise ValueError(err_msg)

    # Check if output directory exists
    if not os.path.exists(args.output_dir):
        err_msg = "Output directory {0} does not exist\n"
        err_msg = err_msg.format(args.output_dir)
        raise ValueError(err_msg)

    # Check if variable argument exists
    if not args.var_name:
        err_msg = "No --var_name argument. A variable to be read from the input file needs to be supplied\n"
        err_msg = err_msg.format(args.output_dir)
        raise ValueError(err_msg)
    # Check if data source argument exists
    if not args.source:
        err_msg = "No --source argument. The file source type needs to be supplied\n"
        err_msg = err_msg.format(args.output_dir)
        raise ValueError(err_msg)

    # Create and populate contour dictionary 
    contour_dict = {}
    #If contour method has been selected then use whatever method is selected else use standard
    if args.contour_method:
        if args.contour_method in ["standard", "pixel"]:
#            contour_dict["contour_method"] = args.contour_method  # Currently only a single contouring method is available
            contour_dict["contour_method"] = "standard"
        else:
           err_msg = "Unrecognised --contour_method argument. A method that is recognised needs to be supplied e.g. standard or pixel.\n"
           err_msg = err_msg.format(args.contour_method)
           raise ValueError(err_msg)
    else:
        contour_dict["contour_method"] = "standard"
    #If contour thresholds have been explicitly defined add them (and approriate flags) to the contour dictionary
    if args.contour_thresholds:
        contour_dict["contour_flag"]=True
        contour_dict["contours_set"]=True
        contour_dict["interval_flag"]=False
        contour_dict["contour_thresholds"] = [numeric_type(lev) for lev in args.contour_thresholds]
        # If contour names have been supplied add these to 
        if args.contour_names:
            contour_dict["contour_names"] = args.contour_names
    else:
        contour_dict["contours_set"]=False
        # If contour start and stop points have not been defined return fill values for start and stop and approriate flag value
        if any(arg is None for arg in [args.contour_start, args.contour_stop]):
            contour_dict["contour_flag"]=False
            contour_dict["contour_start"] = 0
            contour_dict["contour_stop"] = 0
            # If contour interval has been set add it (and flag) to contour_dictionary
            if not args.interval:
                contour_dict["interval_flag"] = False
                contour_dict["interval"] = 0
            else:
                contour_dict["interval_flag"]=True
                contour_dict["interval"] = args.interval
        # If contour start and stop points have been defined add to contour_dict
        else:
            contour_dict["contour_flag"]=True
            contour_dict["contour_start"] = args.contour_start
            contour_dict["contour_stop"] = args.contour_stop
            if not args.interval:
                contour_dict["interval_flag"] = False
                contour_dict["interval"] = 0
            else:
                contour_dict["interval_flag"]=True
                contour_dict["interval"] = args.interval

    # Create and populate parallelisation dictionary
    parallel_dict = {}
    #If contour method has been selected then use whatever method is selected else use standard
    if args.parallel:
        parallel_dict["parallel_flag"] = True
        if args.num_workers:
            parallel_dict["max_workers"] = args.num_workers
        else:
            print("No value has been selected for num_workers, a defualt value of 10 will be used.")
            parallel_dict["max_workers"] = 10

    else:
        parallel_dict["max_workers"] = 0 # Not actually 0 but the data will be processed in serial without tiling.

    # Create and populate simpification dictionay
    simplify_dict = {}
   
    if args.simplify:
        simplify_dict["simplify_flag"] = True
        if args.tolerance:
            simplify_dict["tolerance"] = args.tolerance
        else:
            print("No value has been selected for simplification tolerance, a defualt value of 0.005 will be used.")
            simplify_dict["tolerance"] = 0.005

    else:
        simplify_dict["tolerance"] = 0.0

    # If colors have been explicitly defined add the palette to the contour dictionary
    if args.colors:
        contour_dict["color_pal"] = args.colors
    else:
        # If the colormap has been defined and is valid add to the contour dictionary, if not set to standard colormap of viridis
        if args.colormap:
            try:
                plt.get_cmap(args.colormap)
                contour_dict["colormap"] = args.colormap
            except ValueError:
                contour_dict["colormap"] = "viridis"
        else:
            contour_dict["colormap"] = "viridis"

    # Define smoothing dictionary, if --smooth and --sigma value supplied add to dictionary (standard sigma value 
    smooth_dict = {}
    if args.smooth:
        smooth_dict["smooth_flag"]=True
        if args.sigma:
            smooth_dict["sigma_override"]=True
            smooth_dict["sigma"] = args.sigma
        else:
            smooth_dict["sigma_override"]=False
    else:
        smooth_dict["smooth_flag"]=False
        smooth_dict["sigma_override"] = False

    # Define level dictionary
    level_dict = {}
    if args.level:
        level_dict["level_id"] = args.level
    if args.level_units:
        level_dict["units"] = args.level_units

    return args.input_file, args.output_dir, args.var_name, args.source, smooth_dict, contour_dict, level_dict, parallel_dict, simplify_dict


def numeric_type(value):
    '''
    When passed a variable attempt to return a valid numeric type

    Parameters
    ----------
    value: variable for which valid numeric type is being returned

    Returns
    -------
    int or float
    '''

    import argparse

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid numeric value: {value}")

def simplify_coords(coords, tolerance):
    """
    Reduce the complexity of polygons, not routinely used for actual contour polygons but can be useful for large polygons of missing data (e.g. CRR)

    Parameters
    ----------
    coords : list
        List of coordinates that are to be simplified
    tolerance: float
        the tolerance
    """
    from shapely.geometry import Polygon

    poly = Polygon(coords)
    poly_s = poly.simplify(tolerance = tolerance)
    simplified_coords = [ list(pair) for pair in poly_s.boundary.coords[:] ]

    return simplified_coords

def find_indices(target_lat, target_lon, lat2d, lon2d):
    '''
    Find the indices of the closest grid point to the required lat and lon point

    Parameters
    ----------
    target_lat: float
        The target latitude
    target_lon: float
        The tartget longitude
    lat2d: array
        2d array of latitude values to search
    lon2d: array
        2d array of longitude values to search

    Returns
    -------
    index1 : integer
        first index of the lat-lon arrays that minimises the distance to the target lat lons
    index2 : integer
        second index of the lat-lon arrays that minimises the distance to the target lat lons
    '''

    index1 = int(np.shape(lat2d)[0] / 2.0)
    index2 = int(np.shape(lat2d)[1] / 2.0)

    dist = np.sqrt((lat2d[index1, index2] - target_lat)**2 + (lon2d[index1, index2] - target_lon)**2)

    while True:
        temp_index1 = index1
        temp_index2 = index2
        temp_dist = dist

        # Check 3x3 neighborhood
        for i in np.arange(-1, 2, 1):
            for j in np.arange(-1, 2, 1):
                new_index1 = temp_index1 + i
                new_index2 = temp_index2 + j

                # Check if the new indices are within bounds
                if 0 <= new_index1 < lat2d.shape[0] and 0 <= new_index2 < lat2d.shape[1]:
                    # Calculate the distance for each neighboring point
                    neighbor_dist = np.sqrt((lat2d[new_index1, new_index2] - target_lat)**2 +
                                            (lon2d[new_index1, new_index2] - target_lon)**2)

                    # Update if a closer point is found
                    if neighbor_dist < dist:
                        dist = neighbor_dist
                        index1 = new_index1
                        index2 = new_index2

        # If the indices haven't changed, break the loop
        if temp_index1 == index1 and temp_index2 == index2:
            print(f"Minimum distance found at index1 = {index1}, index2 = {index2}")
            break

    return index1, index2

def filter_var_data(data, gauss_sigma):
    """
    Use a Gaussian filter to smooth the data

    Parameters
    ----------
    var : array
        data
    gauss_sigma : float
        Standard deviation for Gaussian kernel

    Returns
    -------
    array
        Smoothed data

    """
    from scipy.ndimage import gaussian_filter

    truncate = gauss_sigma+1.0

    if np.isnan(data).any():
        dataU = data
        dataV = dataU.copy()
        dataV[np.isnan(dataU)]=0
        dataVV=gaussian_filter(dataV, sigma=gauss_sigma)

        dataW=0*dataU.copy()+1
        dataW[np.isnan(dataU)]=0
        dataWW=gaussian_filter(dataW, sigma=gauss_sigma,truncate=truncate)

        data[1:-1,1:-1] = dataVV[1:-1,1:-1]/dataWW[1:-1,1:-1]
    else:
        data[1:-1,1:-1] = gaussian_filter(data[1:-1,1:-1], sigma=gauss_sigma)

    return data

def generate_contours(contour_dict, max_val, min_val):
    """
#    Create list of contours and thresholds 
#
#    Parameters
#    ----------
#    contour_dict : dictionary
#        Containing flags, start, stop and interval information for contours and thresholds
#    max_val : int
#        The maximum value of the read data
#    min_val : int
#        The minimum value of the read data
#    Returns
#    -------
#    2 x lists
#        List of strings representing contour labels
#        List of integers or floats of threshold values for contour generation
#
    """

# Currently this is method only works for linearly spaced contours, work needs to be done to allow for logarithmic scales too.

# Create contours and thresholds list to be returned
    contours = []
    thresholds = []

# Check contents of contour_dict to determing start, stop and interval values.
# If values have not been supplied the values of start, stop and interval will be determined by the range of data being converted.
    if contour_dict["contour_flag"]:
        if not contour_dict["contours_set"]:
            if contour_dict["interval_flag"]: # All contour start, stop and interval have been supplied
                start = contour_dict["contour_start"]
                stop = contour_dict["contour_stop"]
                interval = contour_dict["interval"]
            else: # Level start and stop have been supplied; an approriate interval will be selected (regardless of range)
                val_range = float(contour_dict["contour_stop"])-float(contour_dict["contour_start"])
                order_of_magnitude = np.ceil(np.log10(val_range))
                val_range_scaled = val_range/10**(order_of_magnitude-1)
                interval = (np.floor(val_range_scaled)/2.0)*10**(order_of_magnitude-2)

                start = (np.round(contour_dict["contour_start"]/interval)*interval)-interval
                stop = (np.round(contour_dict["contour_stop"]/interval)*interval)+interval

    else:
        if contour_dict["interval_flag"]: # Level start and stop have not been supplied but an interval value has.
            interval = contour_dict["interval"]

            start = (np.round(min_val/interval)*interval)-interval
            stop = (np.round(max_val/interval)*interval)+interval
        else: # No contour information has been supplied.
            val_range = float(max_val)-float(min_val)
            order_of_magnitude = np.ceil(np.log10(val_range))
            val_range_scaled = val_range/10**(order_of_magnitude-1)
            interval = (np.floor(val_range_scaled)/2.0)*10**(order_of_magnitude-2)

            start = (np.round(min_val/interval)*interval)-interval
            stop = (np.round(max_val/interval)*interval)+interval

    # Generate the lists based on the start, stop and interval values.

    if not contour_dict["contours_set"]:
        for i in np.arange(start, stop+interval, interval):
            contour_name = f"{i}_{i + interval}"
            threshold_value = float(i)
            thresholds.append(float(threshold_value))
            if i != stop:
                contours.append(contour_name)
    else:
        if "contour_names" in contour_dict:
            contours = contour_dict["contour_names"]
            thresholds = contour_dict["contour_thresholds"]
        else:
            for i in np.arange(0, len(contour_dict["contour_thresholds"])-1, 1):
                contour_name = f"{contour_dict['contour_thresholds'][i]}_{contour_dict['contour_thresholds'][i+1]}"
                contours.append(contour_name)
            thresholds = contour_dict["contour_thresholds"]

    return contours, thresholds

def create_color_palette(contours,colormap):
    """
    Create a hexcode pallete for the contours and given matplotlib colorpalette

    Parameters
    ----------
    contours : List
        Levels produced from contour start, stop and interval, (doesn't have to be contours just a list of the correct length)
    colormap : str
        Matplotlib colormap (already checked and replaced with viridis if user defined option not present.

    Returns
    -------
    hex_palette : dict
        Dictionary of hexcode colours linked to the contour index derived from contours
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    cmap = plt.get_cmap(colormap)

    colors = [cmap(i / (len(contours))) for i in range(len(contours))]

    hex_colors = [to_hex(color) for color in colors]

    hex_palette = dict(zip(range(len(hex_colors)), hex_colors))

    return hex_palette

def get_or_create_region_mask(region_name, lat, lon, reader_id, buffer_km=50, mask_dir="masks"):
    """
    Returns a cached or newly generated boolean mask for a given region.

    Parameters:
        region_name (str): Unique name for region, e.g., "Africa"
        lat (ndarray): 2D latitude array
        lon (ndarray): 2D longitude array
        buffer_km (float): Buffer around region in km
        mask_dir (str): Directory to store/read cached masks

    Returns:
        mask (ndarray): Boolean array of same shape as lat/lon
    """

    import os
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import transform
    import pyproj
    import matplotlib.pyplot as plt


    os.makedirs(mask_dir, exist_ok=True)
    mask_path = os.path.join(mask_dir, f"{reader_id}_{region_name}_buffer{buffer_km}km.npy")

    if os.path.exists(mask_path):
        print(f"✅ Using cached mask from: {mask_path}")
        return np.load(mask_path)

    print(f"⚙️  Creating new mask for region: {region_name}")

    shp = gpd.read_file("data/ne_110m_admin_0_countries.shp")

    region = shp[shp["CONTINENT"].str.lower() == region_name.lower()]
    if region.empty:
        region = shp[shp["NAME"].str.lower() == region_name.lower()]

    if region.empty:

        shp = gpd.read_file("data/10m_cultural/ne_10m_admin_0_countries.shp")

        region = shp[shp["CONTINENT"].str.lower() == region_name.lower()]
        if region.empty:
            region = shp[shp["NAME"].str.lower() == region_name.lower()]

    if region.empty:

        raise ValueError(f"Region '{region_name}' not found in continent or country list.")

    # Combine geometries (handles MultiPolygon) and apply buffer
    region_poly = region.unary_union
    to_m = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    to_deg = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    buffered = transform(to_m, region_poly).buffer(buffer_km * 1000)
    buffered_latlon = transform(to_deg, buffered)

    # Flatten 2D lat/lon arrays to list of points
    flat_lat = lat.ravel()
    flat_lon = lon.ravel()
    mask_flat = np.array([
        buffered_latlon.contains(Point(x, y)) if np.isfinite(x) and np.isfinite(y) else False
        for x, y in zip(flat_lon, flat_lat)
    ])

    # Reshape back to 2D mask
    mask = mask_flat.reshape(lat.shape)

    np.save(mask_path, mask)
    print(f"💾 Mask saved to: {mask_path}")
    return mask

def add_points(feature_collection, lats, lons, properties=None):
    """
    Add points to an existing feature collection
    
    Parameters
    ----------
    feature_collection : dict
        Feature collection in same structure as geojson
    lat : list
        List of latitudes (floats)
    lon : list
        List of longitude (floats)
    properties : dict
        Dictionary containing properties to be included for the point feature

    Returns
    -------
    dictionary of the input feature collection but with the points added

    """

    for lat, lon in zip(lats, lons):
        point_feature = { "type": "Feature", "geometry" : {"type" : "Point", "coordinates" : [lon, lat]}, "properties" : properties}
        feature_collection["features"].append(point_feature)

    return feature_collection
