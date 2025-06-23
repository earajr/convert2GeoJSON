import datetime
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_test_data(shape=(350, 350), value_range=(0, 50.0), entry_name="entry000", seed=42):
    """
    Generates place holder test data with sufficient complexity to act as a test dataset for testing other components.

    Parameters
    ----------
    shape : tuple
        Shape of the test data being generated, default value of 350x350
    value_range: tuple
        Approximate max and min values of the test data being generated
    entry_name: string
        Standard entry name to spoof data being read in from a file
    seed: integer
        Seed to be used for random number generation.

    Returns
    -------
    data_dict : dict
        Dictionary of randomly generated data and metadata to spoof real data read from a file.
    """

    np.random.seed(seed)

    # Create base field: gradient + noise
    y = np.linspace(0, 1, shape[0])
    x = np.linspace(0, 1, shape[1])
    X, Y = np.meshgrid(x, y)

    base = (X + Y) / 2 * (value_range[1] - value_range[0])
    noise = np.random.rand(*shape) * 10
    values = base + noise

    # Add high-value islands (e.g., storms or peaks)
    for _ in range(3):
        cx, cy = np.random.randint(50, 250, size=2)
        radius = np.random.randint(10, 30)
        blob = np.zeros(shape)
        Y_grid, X_grid = np.ogrid[:shape[0], :shape[1]]
        mask = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2 <= radius ** 2
        blob[mask] = value_range[1] * 2.0  # slightly exceed max
        blob = gaussian_filter(blob, sigma=5)
        values += blob

    # Add low-value pockets (e.g., depressions)
    for _ in range(3):
        cx, cy = np.random.randint(50, 250, size=2)
        radius = np.random.randint(10, 30)
        blob = np.zeros(shape)
        Y_grid, X_grid = np.ogrid[:shape[0], :shape[1]]
        mask = (X_grid - cx) ** 2 + (Y_grid - cy) ** 2 <= radius ** 2
        blob[mask] = -value_range[1] * 0.8
        blob = gaussian_filter(blob, sigma=5)
        values += blob

    # Clip values to range
    values_temp = np.clip(values, value_range[0], value_range[1])

    # Create curvilinear lat/lon arrays (simulate map projection warp)
    lat_temp = np.linspace(-10, 10, shape[0])[:, None] + 0.05 * np.sin(2 * np.pi * X)
    lon_temp = np.linspace(100, 120, shape[1])[None, :] + 0.05 * np.cos(2 * np.pi * Y)

    values = values_temp
    lat = lat_temp
    lon = lon_temp

    # Fill metadata with dummy values
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

    # Build dictionary
    data_dict = {
        entry_name: {
            'values': values,
            'lat': lat,
            'lon': lon,
            'metadata': metadata
        }
    }

    return data_dict

