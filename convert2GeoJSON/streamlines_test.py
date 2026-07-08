def add_streamlines(feature_collection, var1, var2, lat, lon):

    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as crs

    max_dim = np.mean(np.shape(lat))
    density = max_dim/25.0
#    print(density)
#
    windspeed = np.sqrt((var1**2.0)+(var2**2.0))

# Create figure and axes
#    from wrf import (getvar, interplevel, vertcross, CoordPair, ALL_TIMES, to_np, get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim, extract_times, extract_global_attrs, ll_to_xy, get_proj_params, getproj)
#    from netCDF4 import Dataset
#
#    wrf_in = Dataset("/home/force-nwr/nwr/uk/data/2026022600/wrfout_d01_2026-02-26_01:00:00")
#    slp_all = getvar(wrf_in, 'slp', timeidx=0)[:,:]
#    cart_proj = get_cartopy(slp_all)

#    fig = plt.figure(figsize=(10,10))
#    ax = plt.axes()#plt.axes(projection=cart_proj)
#    ax = plt.axes(projection=cart_proj)
#    ax.coastlines(linewidth=1.0)
#    gl = ax.gridlines(linewidth=0.5, draw_labels=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
#    gl.right_labels = False
#    gl.bottom_labels = False
#
#    windspeed_lvl = np.arange(0.0, 41.0, 1.0)
# 
#    plt.contourf(lon, lat, windspeed, levels=windspeed_lvl, zorder=1, cmap="viridis", transform=crs.PlateCarree(), extend="max")
#
#    streamlines = ax.streamplot(lon, lat, var1, var2, density=density, color='white', transform=crs.PlateCarree())
    streamlines = plt.streamplot(lon, lat, var1, var2, density=density, color='white')
    
    paths = streamlines.lines.get_paths()
    print(paths)


#    ax.scatter(
#        [-1.54, -0.11, -3.16],
#        [53.79, 51.50, 55.95],
#        color='yellow',
#        transform=crs.PlateCarree(),
#        zorder=10
#    )
#
#    plt.show()

    return feature_collection

