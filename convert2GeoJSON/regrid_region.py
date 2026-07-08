import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import pickle
import matplotlib.tri as mtri

R_earth = 6371.0

# --- estimate local resolution ----------------------------------------------
def source_spacing(lons, lats, mask=None, nn=2, percentile=5):
    """
    Estimate target spacing (km) from nearest-neighbour distances.
    nn=2 => nearest neighbour excluding self (first NN is self).
    percentile: take this percentile of the distribution of NN distances.
    """
    pts_lon = np.asarray(lons).ravel()
    pts_lat = np.asarray(lats).ravel()
    if mask is not None:
        pts_lon = pts_lon[mask]
        pts_lat = pts_lat[mask]
    pts = np.column_stack([pts_lon, pts_lat])
    tree = cKDTree(np.deg2rad(pts))  # KDTree in radians is fine; we'll compute haversine exactly
    # query k nearest; include self at distance 0
    dists_rad, idx = tree.query(np.deg2rad(pts), k=nn)
    # convert angular to km: distance = rad * R_earth
    dists_km = dists_rad * R_earth
    # choose the 2nd column (nearest neighbor excluding self)
    nn_distances_km = dists_km[:, 1] if nn >= 2 else dists_km[:, 0]
    # protect against zeros/nans
    nn_distances_km = nn_distances_km[np.isfinite(nn_distances_km) & (nn_distances_km>0)]
    if len(nn_distances_km) == 0:
        raise RuntimeError("No valid nearest neighbour distances found.")
    return float(int(0.8*np.percentile(nn_distances_km, percentile))) # Safety factor included and value returned as float closest to whole integer number

# --- build rectangular target grid -----------------------------------------
def build_target_grid(limit_lons, limit_lats, spacing_km, lat_reference=None):
    """
    Build lon/lat vectors for rectangular grid given spacing in km.
    lat_reference: latitude to compute lon-degree scaling; if None use mid-lat.
    Returns lon2d, lat2d, (dlon_deg, dlat_deg)
    """
    lon_min, lon_max = float(np.min(limit_lons)), float(np.max(limit_lons))
    lat_min, lat_max = float(np.min(limit_lats)), float(np.max(limit_lats))

    mid_lat = lat_reference if lat_reference is not None else 0.5*(lat_min + lat_max)
    # degrees per km
    deg_per_km_lat = 1.0 / 111.0  # approx degrees latitude per km
    deg_per_km_lon = 1.0 / (111.0 * np.cos(np.deg2rad(mid_lat)))

    dlat = spacing_km * deg_per_km_lat
    dlon = spacing_km * deg_per_km_lon

    # number of grid points (ensure at least 2)
    ny = max(2, int(np.ceil((lat_max - lat_min) / dlat)) + 1)
    nx = max(2, int(np.ceil((lon_max - lon_min) / dlon)) + 1)

    lon_vec = np.linspace(lon_min, lon_min + (nx-1)*dlon, nx)
    lat_vec = np.linspace(lat_min, lat_min + (ny-1)*dlat, ny)
    lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)

    return lon2d, lat2d

def build_target_grid_from_points(lons, lats, spacing_km, lat_reference=None):
    """
    Build a rectangular lon/lat target grid based on an existing set
    of lon/lat points (e.g. MPAS points already filtered in projected space).

    Parameters
    ----------
    lons, lats : 1D arrays
        Longitude and latitude of source points.
    spacing_km : float
        Desired grid spacing in kilometers.
    lat_reference : float, optional
        Reference latitude for longitude scaling.
        If None, median latitude of input points is used.

    Returns
    -------
    lon2d, lat2d : 2D arrays
        Target grid coordinates.
    """

    lons = np.asarray(lons).ravel()
    lats = np.asarray(lats).ravel()

    # Remove any non-finite points (important for safety)
    good = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[good]
    lats = lats[good]

    if lons.size < 2:
        raise ValueError("Not enough points to build a target grid")

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()

    # Reference latitude for lon scaling
    ref_lat = lat_reference if lat_reference is not None else np.median(lats)

    # degrees per km
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * np.cos(np.deg2rad(ref_lat)))

    dlat = spacing_km * deg_per_km_lat
    dlon = spacing_km * deg_per_km_lon

    # Number of grid points
    ny = max(2, int(np.ceil((lat_max - lat_min) / dlat)) + 1)
    nx = max(2, int(np.ceil((lon_max - lon_min) / dlon)) + 1)

    lat_vec = lat_min + dlat * np.arange(ny)
    lon_vec = lon_min + dlon * np.arange(nx)

    lon2d, lat2d = np.meshgrid(lon_vec, lat_vec)

    return lon2d, lat2d

def build_or_load_triangulation(src_lons, src_lats, cache_key, cache_dir="tri_cache"):
    """
    Build matplotlib Triangulation from source points or load cached version.
    cache_key: string used to name the cache file; must uniquely identify source points (bbox + #points or md5).
    """
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(fname):
        with open(fname, "rb") as f:
            tri_data = pickle.load(f)
        triang = mtri.Triangulation(tri_data["x"], tri_data["y"], tri_data["triangles"])
        return triang

    # Build triangulation (Delaunay)
    x = np.asarray(src_lons).ravel()
    y = np.asarray(src_lats).ravel()
    triang = mtri.Triangulation(x, y)

    save = {"x": x, "y": y, "triangles": triang.triangles}
    with open(fname, "wb") as f:
        pickle.dump(save, f, protocol=pickle.HIGHEST_PROTOCOL)
    return triang

def triangulation_regrid_to_grid(src_lons, src_lats, src_field, lon2d, lat2d, cache_key, tri_cache_dir="tri_cache", fill_value=np.nan):
    """
    Interpolate src_field (1D) defined at src_lons/src_lats (1D) to target lon2d/lat2d using
    linear interpolation on a Delaunay triangulation. Returns gridded field (ny,nx).
    """
    # prepare arrays
    src_lons = np.asarray(src_lons).ravel()
    src_lats = np.asarray(src_lats).ravel()
    src_field = np.asarray(src_field).ravel()
    # build or load triangulation
    triang = build_or_load_triangulation(src_lons, src_lats, cache_key, cache_dir=tri_cache_dir)

    # Linear interpolator on triangulation
    lininterp = mtri.LinearTriInterpolator(triang, src_field)

    # Evaluate on target grid
    lon_flat = lon2d.ravel()
    lat_flat = lat2d.ravel()
    interp_vals = lininterp(lon_flat, lat_flat)  # returns masked array when outside convex hull
    # convert masked to fill_value
    if np.ma.is_masked(interp_vals):
        interp_vals = interp_vals.filled(fill_value)
    field_grid = interp_vals.reshape(lon2d.shape)
    return field_grid
