def add_streamlines(feature_collection, var1, var2, lat, lon):

    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as crs
    import json

    dx = np.max([np.abs(lat[0,0]-lat[1,0]), np.abs(lat[0,0]-lat[0,1])])*111.0
    max_dim = np.max(np.shape(lat))
    max_domain_dimension = dx*max_dim
    max_density = max_dim/20.0
    min_density = 2.0

    if dx < 1.0:
        max_zoom = 10
    elif dx < 10.0:
        max_zoom = 9
    elif dx < 100.0:
        max_zoom = 8
    elif dx < 500.0:
        max_zoom = 7
    elif dx < 1000.0:
        max_zoom = 6

    if max_domain_dimension > 6000.0:
        min_zoom = 4
    elif max_domain_dimension > 2500.0:
        min_zoom = 5
    elif max_domain_dimension > 1000.0:
        min_zoom = 6

    densities = []
    if min_zoom != max_zoom and min_zoom < max_zoom:
        for i, zoom in enumerate(np.arange(min_zoom, max_zoom+1, 1)):
            density = int(min_density + ((max_density - min_density)/float(max_zoom-min_zoom))*(float(i)))
            densities.append((density, zoom))
    else:
        print("Using standard densities and zoom levels")
        min_zoom = 5
        max_zoom = 8
        
        for i, zoom in enumerate(np.arange(min_zoom, max_zoom+1, 1)):
            density = min_density + ((max_density - min_density)/float(max_zoom-min_zoom))*(float(i))
            densities.append((density, zoom))

    features = []

    for density, minzoom in densities:
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes()#plt.axes(projection=cart_proj)



        features = []
        print("density: ", density)
        streamlines = ax.streamplot(lon, lat, var1, var2, density=density, color='black')
#        plt.show()

        paths = streamlines.lines.get_paths()
        print("number of paths: ", len(paths))

        for path in paths:
            verts = path.vertices
            coords = [(float(x), float(y)) for x, y in verts]
            if len(coords) > 1:
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords
                    },
                    "properties": {
                        "fill_name": int(minzoom)
                    }
                })

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(f"streamlines{minzoom}.geojson", "w") as f:
            json.dump(geojson, f)

    '''
    max_dim = np.mean(np.shape(lat))
    

    density = max_dim/25.0
    windspeed = np.sqrt((var1**2.0)+(var2**2.0))
    streamlines = plt.streamplot(lon, lat, var1, var2, density=density, color='white')
    
    paths = streamlines.lines.get_paths()

    features = []

    for path in paths:
        verts = path.vertices

        coords = [(float(x), float(y)) for x, y in verts]

        if len(coords) > 1:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {}
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open("streamlines.geojson", "w") as f:
        json.dump(geojson, f)
    '''
    return feature_collection

