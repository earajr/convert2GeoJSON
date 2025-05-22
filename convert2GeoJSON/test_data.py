import datetime
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_test_data(shape=(350, 350), value_range=(0, 50.0), entry_name="entry000", seed=42):
    np.random.seed(seed)

    # 1. Create base field: gradient + noise
    y = np.linspace(0, 1, shape[0])
    x = np.linspace(0, 1, shape[1])
    X, Y = np.meshgrid(x, y)

    base = (X + Y) / 2 * (value_range[1] - value_range[0])
    noise = np.random.rand(*shape) * 5
    values = base + noise

    # 2. Add high-value islands (e.g., storms or peaks)
    for _ in range(3):
        cx, cy = np.random.randint(50, 250, size=2)
        radius = np.random.randint(10, 30)
        blob = np.zeros(shape)
        Y_grid, X_grid = np.ogrid[:shape[0], :shape[1]]
        mask = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2 <= radius ** 2
        blob[mask] = value_range[1] * 2.0  # slightly exceed max
        blob = gaussian_filter(blob, sigma=5)
        values += blob

    # 3. Add low-value pockets (e.g., depressions)
    for _ in range(3):
        cx, cy = np.random.randint(50, 250, size=2)
        radius = np.random.randint(10, 30)
        blob = np.zeros(shape)
        Y_grid, X_grid = np.ogrid[:shape[0], :shape[1]]
        mask = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2 <= radius ** 2
        blob[mask] = -value_range[1] * 0.8
        blob = gaussian_filter(blob, sigma=5)
        values += blob

    # 4. Clip values to range
    values_temp = np.clip(values, value_range[0], value_range[1])

    # 5. Create curvilinear lat/lon arrays (simulate map projection warp)
    lat_temp = np.linspace(-10, 10, shape[0])[:, None] + 0.05 * np.sin(2 * np.pi * X)
    lon_temp = np.linspace(100, 120, shape[1])[None, :] + 0.05 * np.cos(2 * np.pi * Y)

#    # Add additional frame around where data is present and populate with lat and lon values using finite difference approach
#
#    lat = np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2))
#    lat[1:-1,1:-1] = lat_temp
#    lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
#    lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
#    lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
#    lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])
#
#    lon = np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2))
#    lon[1:-1,1:-1] = lon_temp
#    lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
#    lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
#    lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
#    lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])
#
#    # Fill variables in empty frame around data with the current lowest value in the dataset
#    edge_list = list(values_temp[0,:])+list(values_temp[-1,:])+list(values_temp[:,0])+list(values_temp[:,-1])
#    edge_min = np.amin(edge_list)
#    values_step = 0.5
#    values = np.zeros((np.shape(values_temp)[0]+2, np.shape(values_temp)[1]+2))
#    values[ values == 0.0 ] =  edge_min - values_step
#    values[1:-1,1:-1] = values_temp

    values = values_temp
    lat = lat_temp
    lon = lon_temp

    # 5. Fill metadata with dummy values
    now = datetime.datetime.utcnow()
    metadata = {
        'varname': "synthetic_var",
        'level_type': "surface",
        'grid_id': "grid_001",
        'sim_start_time': now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        'valid_time': (now + datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        'units': "unitless",
        'grid_spacing': 1.0,
        'grid_units': "degrees",
        'sigma': 1.0
    }

    # 6. Build dictionary
    data_dict = {
        entry_name: {
            'values': values,
            'lat': lat,
            'lon': lon,
            'metadata': metadata
        }
    }

    return data_dict

