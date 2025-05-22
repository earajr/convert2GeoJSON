import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from shapely.geometry import Polygon, mapping, MultiPolygon, LineString
from shapely.ops import unary_union
from shapely import coverage_union_all
from shapely.geometry.polygon import orient#
from shapely.strtree import STRtree
from shapely.ops import polygonize
import json
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
from collections import defaultdict

from shapely.geometry import Polygon

def separate_holes_from_shells(polygons):
    """
    From a list of polygons, separate those whose exteriors match holes of others.

    Returns:
        final_polys: list of polygons **without** duplicates (shells with holes assigned)
        rejected_polys: list of polygons that are duplicates (holes to pass downward)
    """
    hole_polys = []
    for poly in polygons:
        for interior in poly.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.is_valid and hole_poly.area > 0:
                hole_polys.append(hole_poly)

    final_polys = []
    rejected_polys = []

    for poly in polygons:
        is_duplicate = any(Polygon(poly.exterior).equals(hole) for hole in hole_polys)
        if is_duplicate:
            rejected_polys.append(poly)
#        else:
#            final_polys.append(poly)

    return final_polys, rejected_polys

def remove_duplicate_holes(polygons):
    """
    Remove any polygon whose exterior exactly matches any hole (interior ring) of another polygon,
    ignoring coordinate order and winding direction.
    """
    final_polys = []
    rejected_polys = []
    hole_polys = []

    # Step 1: collect all hole geometries as Polygon objects
    for poly in polygons:
        for interior in poly.interiors:
            hole_poly = Polygon(interior)
            if hole_poly.is_valid and hole_poly.area > 0:
                hole_polys.append(hole_poly)

    # Step 2: exclude any polygon whose exterior matches a hole
    for poly in polygons:
        is_duplicate = any(Polygon(poly.exterior).equals(hole) for hole in hole_polys)
        if not is_duplicate:
            final_polys.append(poly)
        else:
            rejected_polys.append(poly)

    return final_polys, rejected_polys

def get_contour_feature_data(var, lat, lon, contours, thresholds, metadata, hex_palette, contour_method):
    """
    Generate GeoJSON features from data
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

    n = len(cs.collections)

    # Step 1: collect raw polygons from each contour level
    raw_polygons = {}
    for i, collection in enumerate(cs.collections):
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
            final_polys, rejected_polys = remove_duplicate_holes(raw_polys)
            raw_polygons[i] = final_polys
#            if i > 0:
#                raw_polygons.setdefault(i - 1, []).extend(rejected_polys)



#    # Step 2: subtract overlapping areas to get clean levels
    sorted_levels = sorted(raw_polygons.keys())
    clean_polygons = {}

    for i, level in enumerate(sorted_levels):
        cleaned = MultiPolygon(raw_polygons[level])

#       if i < n-1:
#            currnt = MultiPolygon(raw_polygons[level])
#            nxt = MultiPolygon(raw_polygons[sorted_levels[i+1]])
#
#            cleaned = subtract_preserving_holes(currnt, nxt)
#
##            cleaned = subtract_as_holes(cleaned, nxt)
#
#        else:
#            currnt = MultiPolygon(raw_polygons[level])
#            cleaned = currnt
#
        clean_polygons[level] = cleaned

    # Step 3: assemble final GeoJSON
    feature_collection = {
        "type": "FeatureCollection",
        "properties": {
            "levels": {i: level for i, level in enumerate(contours)},
            "hex_palette": hex_palette
        },
        "features": []
    }

    for i in sorted_levels:
        geom = clean_polygons[i]
        if geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            geoms = [geom]
        else:
            geoms = list(geom.geoms)

        for g in geoms:
            if g.is_empty or not g.is_valid or g.area <= 0:
                continue
            g = orient(g, sign=1.0)
            feature_collection["features"].append({
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "ObjectType": "data-contour",
                    "level": i,
                    "level_value": contours[i] if i < len(contours) else None,
                    "threshold": thresholds[i] if i < len(thresholds) else None
                }
                })

    return feature_collection


def get_contour_feature_data_old(var, lat, lon, contours, thresholds, metadata, hex_palette, contour_method):
    """
    Generate GeoJSON features from data

    Parameters
    ----------
    var : array
        Array specifying data
    lat : array
        Array of latitude values
    lon : array
        Array of longitude values
    contours : array
        contour names
    thresholds : array
        Minimum values for each contour

    Returns
    -------
    list
        List of GeoJSON features

    """

    lat_flat = lat.flatten()
    lon_flat = lon.flatten()
    data_flat = var.flatten()

    valid_mask = np.isfinite(data_flat)

    lat_flat = lat_flat[valid_mask]
    lon_flat = lon_flat[valid_mask]
    data_flat = data_flat[valid_mask]

    triang = tri.Triangulation(lon_flat, lat_flat)

    fig, ax = plt.subplots()
    cs = ax.tricontourf(triang, data_flat, levels=thresholds)
    fig.colorbar(cs, orientation='vertical')
    plt.show()
    plt.close(fig)

    # Initialize GeoJSON structure
    feature_collection = {
        "type": "FeatureCollection",
        "properties": {
            "levels": {i: level for i, level in enumerate(contours)},
            "hex_palette": hex_palette
        },
        "features": []
    }

    for i, collection in enumerate(cs.collections):
        shells = []
        holes = []
        final_polygons = []

        for path in collection.get_paths():
            for coords in path.to_polygons():
                if len(coords) < 3:
                    continue
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    if is_ccw(poly):
                        shells.append(poly)
                    else:
                        holes.append(poly)

        for shell in shells:
            contained_holes = []
            for hole in holes:
                if shell.contains(hole):
                    contained_holes.append(hole.exterior.coords)
            poly_with_holes = Polygon(shell.exterior.coords, holes=contained_holes)
            final_polygons.append(poly_with_holes)

        for g in final_polygons:
            if g.is_empty or not g.is_valid or g.area <= 0:
                continue
            g = orient(g, sign=1.0)
            feature_collection["features"].append({
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "ObjectType": "data-contour",
                    "level": i,
                    "level_value": contours[i] if i < len(contours) else None,
                    "threshold": thresholds[i] if i < len(thresholds) else None
                }
            })







#
#
#
#
#
#
#
#
#    # Create feature collection
#
#    feature_collection = {
#        "type": "FeatureCollection",
#        "properties": {
#            "levels": {i: level for i, level in enumerate(contours)},
#            "hex_palette":hex_palette
#        },
#        "features": []
#    }
#
#    n = len(cs.collections)
#
#    for i, collection in enumerate(cs.collections):
#        shells = []
#        holes = []
#        for path_idx, path in enumerate(collection.get_paths()):
#            polygons = path.to_polygons()
#            level_polys = []
#            for coords in polygons:
#                if len(coords) < 3:
#                    continue
#                poly = Polygon(coords)
#                if poly.is_valid and poly.area > 0:
#                    level_polys.append(poly)
#                    if polygon_winding(poly) == 'CCW':
#                        shells.append(poly)
#                    else:
#                        holes.append(poly)
#
#        final_polygons = []
#        for shell in shells:
#            contained_holes = []
#            for hole in holes:
#                if shell.contains(hole):
#                    contained_holes.append(hole.shell.coords)
#                    final_polygons.append(Polygon(shell.exterior.coords, holes=[h for h in contained_holes]))
#
#        print(final_polygons)
#                    
##            cov_union[i] = coverage_union_all(level_polys)
##            un_union[i] = unary_union(level_polys)
##
##    for i in range(n):
##       shell = un_union[i]
##       if i < n-1:
##           shell = shell.difference(un_union[i+1])
##
##       holes[i] = cov_union[i].difference(shell)
#
#       
#
##       if i < n-1:
##           geoms[i] = geoms_temp[i].difference(geoms_temp[i+1])
##       else:
##           geoms[i] = geoms_temp[i]
#
#
#
#
##    ascending_geoms = [None] * n
##    descending_geoms = [None] * n
##
##    for i in range(n):
##        current = unary_union(level_polygons[i])
##        if i < n - 1:
##            higher = unary_union(level_polygons[i + 1])
##            current = current.difference(higher)
##        ascending_geoms[i] = current
##
##    for i in reversed(range(n)):
##        current = unary_union(level_polygons[i])
##        if i > 0:
##            lower = unary_union(level_polygons[i - 1])
##            current = current.difference(lower)
##        descending_geoms[i] = current
##
##    combined_geoms = []
##
##    for i in range(n):
##        outer = ascending_geoms[i]
##        holes = descending_geoms[i]
##
##        # Normalize to lists
##        outer_polys = outer.geoms if outer.geom_type == "MultiPolygon" else [outer]
##        hole_polys = holes.geoms if holes.geom_type == "MultiPolygon" else [holes]
##
##        for outer_poly in outer_polys:
##            if not outer_poly.is_valid or outer_poly.is_empty:
##                continue
##
##            ring = outer_poly
##            interior_rings = []
##
##            for hole_candidate in hole_polys:
##                if (
##                    hole_candidate.is_valid
##                    and not hole_candidate.is_empty
##                    and hole_candidate.within(ring)
##                ):
##                    interior_rings.append(hole_candidate.exterior.coords)
##
##            # Create a new polygon with holes (if any)
##            polygon_with_holes = Polygon(ring.exterior.coords, holes=interior_rings)
##
##            if polygon_with_holes.is_valid and polygon_with_holes.area > 0:
##                combined_geoms.append((i, polygon_with_holes))
##
#    for i, geom in enumerate(holes):
#        feature_collection["features"].append({
#            "type": "Feature",
#            "geometry": mapping(geom),
#            "properties": {
#                "ObjectType": "data-contour",
#                "level": i,
#            }
#        })
#




#    for i in range(n):
#        current_ascending = unary_union(level_polygons[i])
#
#        if i < n-1:
#            next_union = unary_union(level_polygons[i+1])
#            current_ascending = current_ascending.difference(next_union)
#
#    for i in reversed(range(n)):
#        current_descending = unary_union(level_polygons[i])
#geoms
#        if i != 0:
#            next_union = unary_union(level_polygons[i-1])
#            current_descending = current_descending.difference(next_union)

#        if i + 1 < n:
#            next_polys = level_polygons[i+1]
#            # build tree for speed if there are lots of them
#            tree = STRtree(next_polys)
#            # find the subset that actually overlap 'current'
#            candidates = tree.query(current)
#            print(candidates)
#            to_subtract = [p for p in candidates if p.intersects(current)]
#            if to_subtract:
#               current = current.difference(unary_union(to_subtract))
#
#        if i - 1 >= 0:
#            prev_polys = level_polygons[i-1]
#            tree = STRtree(prev_polys)
#            candidates = tree.query(current)
#            print(candidates)
#            # keep only those that are real “islands” inside this band
#            holes = [p for p in candidates if p.within(current)]
#            if holes:
#                current = current.difference(unary_union(holes))

#        if i - 1 >= 0:
#            prev_union = unary_union(level_polygons[i-1])
#            current = current.difference(prev_union)
#
#        geoms = current_descending.geoms if current_descending.geom_type=="MultiPolygon" else [current_descending]
#        for g in geoms:
#            if g.is_empty or not g.is_valid or g.area<=0: continue
#            g = orient(g, sign=1.0)
#            feature_collection["features"].append({
#                "type": "Feature",
#                "geometry": mapping(g),
#                "properties": {
#                    "ObjectType": "data-contour",
#                    "level": i,
#                }
#            })

    return feature_collection

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


