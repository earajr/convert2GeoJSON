import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from shapely.geometry import Polygon, mapping, MultiPolygon, LineString
from shapely.ops import polygonize
from shapely.geometry.polygon import orient
from concurrent.futures import ProcessPoolExecutor

def compute_tile_bounds(i, j, lat, lon):
    """
    Compute polygon corners for tile (i, j) with overlap.

    Parameters:
    ----------
    i & j:  Indices of data
    tile_size : Size of the tile on which data will be split for contouring and contour manipulation.
    overlap: Size of overlap to guarantee that erroneous polygons at tile edges can be removed at a later stage
    lat : 2d array of latitide points
    lon : 2d array of longitude points

    Returns:
    --------
    outer_bounds: List of (lon, lat) tuples for 4 corners of the outer tile (with overlap)
    inner_bounds: List of (lon, lat) tuples for 4 corners of the clipped tile (less overlap)
    """

    overlap = 5
    tile_size = 150

    ny, nx = lat.shape

    # Outer bounds with full overlap
    i_start = max(i - overlap, 0)
    i_end = min(i + tile_size + overlap, ny)
    j_start = max(j - overlap, 0)
    j_end = min(j + tile_size + overlap, nx)

    outer_corners = [
        (lon[i_start, j_start], lat[i_start, j_start]),  # Top-left
        (lon[i_start, j_end - 1], lat[i_start, j_end - 1]),  # Top-right
        (lon[i_end - 1, j_end - 1], lat[i_end - 1, j_end - 1]),  # Bottom-right
        (lon[i_end - 1, j_start], lat[i_end - 1, j_start])   # Bottom-left
    ]

    # Inner bounds for clipping
    i_start_clip = max(i - overlap + 7, 0)
    i_end_clip = min(i + tile_size + overlap - 7, ny)
    j_start_clip = max(j - overlap + 7, 0)
    j_end_clip = min(j + tile_size + overlap - 7, nx)

    inner_corners = [
        (lon[i_start_clip, j_start_clip], lat[i_start_clip, j_start_clip]),  # Top-left
        (lon[i_start_clip, j_end_clip - 1], lat[i_start_clip, j_end_clip - 1]),  # Top-right
        (lon[i_end_clip - 1, j_end_clip - 1], lat[i_end_clip - 1, j_end_clip - 1]),  # Bottom-right
        (lon[i_end_clip - 1, j_start_clip], lat[i_end_clip - 1, j_start_clip])   # Bottom-left
    ]

    return outer_corners, inner_corners

def safe_polygon_from_corners(corners):
    """
    Return a valid Polygon from corners if no NaNs are present.

    Parameters:
    ----------
    corners: coordinates of the corners which will be used to create the returned polygon

    Returns:
    --------
    Polygon: A polygon described by the corners passed to the function.
    """

    if any(np.isnan(coord).any() for coord in corners):
        return None  # or raise/skips based on your logic
    return Polygon(close_ring(corners))

def close_ring(coords):
    """
    Ensure that a coordinate ring is closed (that the first and last points are equal to each other.

    Parameters:
    ----------
    coords: coordinates of a polygon to be checked

    Returns:
    --------
    coords: coordinates of a polygon once checked
    """

    if coords[0] != coords[-1]:
        return coords + [coords[0]]
    return coords

def clip_polygons_to_bbox(polygons, bbox):
    """
    Clip the oversized tile regions to the exact region of interest to remove artifacts generated along tile edges

    Parameters:
    ----------
    polygons: List of polygons to be used to create geojson
    bbox: bounding box of the tile to be used to trim polygons to correct region

    Returns:
    --------
    clipped_polys: List of polygons after clipping
    """

    clipped_polys = []
    for poly in polygons:
        clipped = poly.intersection(bbox)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            if clipped.area > 0 and clipped.is_valid:
                clipped_polys.append(orient(clipped, sign=1.0))
        elif isinstance(clipped, MultiPolygon):
            for p in clipped.geoms:
                if p.area > 0 and p.is_valid:
                    clipped_polys.append(orient(p, sign=1.0))
    return clipped_polys

def remove_duplicate_holes(polygons):
    """
    When polygonize is used on contours polygons are created that are actually holes within the current layer.
    These polygons need to be removed to prevent erroneous regions of overlapping layers in the GeoJSON

    Parameters:
    -----------
    polygons: list of polygons created by polygonize from contours

    Returns:
    --------
    final_polys: polygons that are required to make the GeoJSON
    rejected_polys: Polygons that are rejected due to being duplicates of holes
    """

    final_polys = []
    rejected_polys = []
    hole_polys = []

    for poly in polygons:
        for interior in poly.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.is_valid and hole_poly.area > 0:
                hole_polys.append(hole_poly)

    for poly in polygons:
        is_duplicate = any(Polygon(poly.exterior).equals(hole) for hole in hole_polys)
        if not is_duplicate:
            final_polys.append(poly)
        else:
            rejected_polys.append(poly)

    return final_polys, rejected_polys


def get_contour_feature_data(var, lat, lon, thresholds):
    """
    Generate contours from data values and convert into raw polygons (still need to be trimmed)

    Parameters:
    -----------
    var: array of data values
    lat: 2d latitude array
    lon: 2d longitude array
    thresholds: Threshold values used to generate contours.

    Returns:
    --------
    raw_polygons: dictionary containing lists of raw polygons split by the threshold level.
    """

    if np.ma.isMaskedArray(lat):
        lat = lat.filled(np.nan)
    if np.ma.isMaskedArray(lon):
        lon = lon.filled(np.nan)
    if np.ma.isMaskedArray(var):
        var = var.filled(np.nan)

    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    data_flat = var.flatten()

    valid_mask = np.isfinite(lat_flat) & np.isfinite(lon_flat) & np.isfinite(data_flat)
    lat_flat = lat_flat[valid_mask]
    lon_flat = lon_flat[valid_mask]
    data_flat = data_flat[valid_mask]

    points = np.column_stack([lon_flat, lat_flat])
    unique_points, indices = np.unique(points, axis=0, return_index=True)

    lon_flat = lon_flat[indices]
    lat_flat = lat_flat[indices]
    data_flat = data_flat[indices]

    triang = tri.Triangulation(lon_flat, lat_flat)
    mask = np.any(np.isnan(data_flat[triang.triangles]), axis=1)
    triang.set_mask(mask)

    fig, ax = plt.subplots()
    cs = ax.tricontourf(triang, data_flat, levels=thresholds)
    plt.close(fig)

    raw_polygons = {}
    for i, collection in enumerate(cs.collections):
        # extract segments & polygonize as you have already done
        segments = []
        for path in collection.get_paths():
            vertices = path.vertices
            codes = path.codes

            if codes is not None:
                points = []
                for j, (vertex, code) in enumerate(zip(vertices, codes)):
                    if code == 1 and points:
                        segments.extend([LineString(points[k:k+2]) for k in range(len(points)-1)])
                        points = []
                    points.append(vertex)
                if len(points) > 1:
                    segments.extend([LineString(points[k:k+2]) for k in range(len(points)-1)])
            else:
                segments.extend([LineString(vertices[k:k+2]) for k in range(len(vertices)-1)])
        
        raw_polys = list(polygonize(segments))
        if raw_polys:
            final_polys, _ = remove_duplicate_holes(raw_polys)
            raw_polygons[i] = final_polys

    return raw_polygons

def tile_array(data, lat, lon):
    """
    generate tiles to be processed in parallel to speed up generation of polygons for GeoJSON creation

    Parameters:
    -----------
    data: array of data values
    lat: 2d latitude array
    lon: 2d longitude array

    Returns:
    --------
    tiles: list describing all the different tiles to be processed in parallel.
    """


    tile_size = 150
    overlap = 5

    tiles = []
    ny, nx = data.shape
    for i in range(0, ny, tile_size - overlap):
        for j in range(0, nx, tile_size - overlap):
            i_end = min(i + tile_size, ny)
            j_end = min(j + tile_size, nx)
            tiles.append((
                data[i:i_end, j:j_end],
                lat[i:i_end, j:j_end],
                lon[i:i_end, j:j_end],
                (i, j)
            ))
    return tiles

def process_tile(args):
    """
    Process the tiles in parallel to create required polygons for GeoJSON generation.

    Parameters:
    -----------
    var_tile: array of data for tile
    lat_tile: array of latitude values for tile
    lon_tile: array of longitude values for tile
    i & j: indices of tile
    thresholds: threshold levels for contouring
    tile_bounds_with_overlap: larger tile bounds to be processed
    tile_bounds_core: actual bounds of the tile required (used for clipping after polygon production)

    Returns:
    --------
    clipped_polygons: Polygons that have been created using get_contour_feature_data and clipped to the correct size using clip_polygons_to_bbox

    """

    var_tile, lat_tile, lon_tile, (i, j), thresholds, tile_bounds_with_overlap, tile_bounds_core = args

    tile_bbox_with_overlap = safe_polygon_from_corners(tile_bounds_with_overlap)
    tile_bbox_core = safe_polygon_from_corners(tile_bounds_core)

    if tile_bbox_with_overlap is None or tile_bbox_core is None:
        return None

    else:

        try:
            raw_polygons = get_contour_feature_data(var_tile, lat_tile, lon_tile, thresholds)

            if not raw_polygons:
                return None

            clipped_polygons = {}
            for level, polys in raw_polygons.items():
                clipped = clip_polygons_to_bbox(polys, tile_bbox_core)
                if clipped:
                    clipped_polygons[level] = clipped
        
            return clipped_polygons

        except Exception as e:
            print(f"Error processing tile ({i},{j}):", e)
            return None

def write_geojson(feature_collection, output_dir, input_file, entry, var_name, metadata):
    """
    Write GeoJSON data to output file

    Parameters
    ----------
    feature_collection : dict
        Feature collection in same structure as geojson.
    output_dir : str
        Output directory path
    input_file : str
        Name of input file
    entry : str
        The entry from the input file, a file containing multiple times or levels would have multiple entries.
    """
    import json
    import os

    if metadata["level_type"] == "Single":
        output_file = output_dir+"/"+os.path.splitext(os.path.basename(input_file))[0]+"_"+var_name+"_"+entry+".geojson"
    else:
        output_file = output_dir+"/"+os.path.splitext(os.path.basename(input_file))[0]+"_"+var_name+"_"+entry+"_"+metadata["level_type"]+".geojson"

    # Write geojson file

    with open(output_file, 'w') as f:
        json.dump(feature_collection, f)


def generate_geojson(var, lat, lon, contours, thresholds, metadata, hex_palette):
    """
    Main function in the contouring module, this is called from main and makes use of the other functions in the module to produce
    a GeoJSON.

    Parameters
    ----------
    var: data array
    lat: latitude array
    lon: longitude array
    contours: names of contour levels to be created
    thresholds: threshold values that describe the contour levels being created
    metadata: variety of metadata to properly descibe the data being processed
    hex_palette: standard pallette of hex colours to be included in the resultant GeoJSON
    
    Returns:
    --------
    feature_collection: feature collection dictionary in a format that can easily be exported as a GeoJSON.
    """

    tiles = [(var, lat, lon, (0,0))]

    args_list = []

    for v, la, lo, (i, j) in tiles:
        # Use overlap=2 for larger tile bounds, overlap=1 for core bounds
        tile_bounds_with_overlap, tile_bounds_core = compute_tile_bounds(i, j, lat=lat, lon=lon)

        args_list.append((v, la, lo, (i, j), thresholds, tile_bounds_with_overlap, tile_bounds_core))

    all_features = []
    with ProcessPoolExecutor(max_workers=1) as executor:
        for result in executor.map(process_tile, args_list):
            if result:
                # result is dict: level -> list of polygons
                for level, polys in result.items():
                    for poly in polys:
                        all_features.append({
                            "type": "Feature",
                            "geometry": mapping(poly),
                            "properties": {
                                "ObjectType": "data-contour",
                                "level": level,
                                "level_value": contours[level] if level < len(contours) else None,
                                "threshold": thresholds[level] if level < len(thresholds) else None
                            }
                        })

    feature_collection = {
        "type": "FeatureCollection",
        "properties": {
            "levels": {i: level for i, level in enumerate(contours)},
            "hex_palette": hex_palette
        },
        "features": all_features
    }

    return feature_collection
