import utils 
import read_data
import contouring_par
import test_data

def main():
    """
    Main function of the convert2GeoJSON code. This function calls on
    other functions within the repository to manage arguments, read data
    and convert to contours. Once contour polygons ahve been created the
    result is exported as a GeoJSON file ready to be used within mapping
    applications.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Check input arguments
    input_file, output_dir, VAR_NAME, SOURCE, smooth_dict, contour_dict, level_dict, parallel_dict, simplify_dict = utils.input_args()

    # Read data file

    data = read_data.read_datafile(SOURCE, input_file, VAR_NAME, contour_dict, level_dict, parallel_dict["max_workers"])


    # Instead of using real data test data can be generated to make sure that the processing of data after the reading process is all working as expected.
#    data = test_data.generate_test_data()

    # Identify max and min values from read datasets. This is only useful if specific contour thresholds have not been provided
    for entry in data:
        if entry == "entry000":
            max_int_data = np.ceil(np.nanmax(data[entry]["values"]))
            min_int_data = np.floor(np.nanmin(data[entry]["values"]))
        else:
            if np.ceil(np.nanmax(data[entry]["values"])) > max_int_data:
                max_int_data = np.ceil(np.nanmax(data[entry]["values"]))
            if np.floor(np.nanmin(data[entry]["values"])) < min_int_data:
                min_int_data = np.floor(np.nanmin(data[entry]["values"]))

    for entry in data:
        if entry == "entry000":
            max_data = np.nanmax(data[entry]["values"])
            min_data = np.nanmin(data[entry]["values"])
        else:
            if np.nanmax(data[entry]["values"]) > max_data:
                max_data = np.nanmax(data[entry]["values"])
            if np.nanmin(data[entry]["values"]) < min_data:
                min_data = np.nanmin(data[entry]["values"])

    # Calculate data range to identify if valid data is present (not all 1 value) and round min and max values to approriate values
    data_range = max_data - min_data

    if data_range != 0.0:
        if data_range <= 2.0:
            exponent = np.floor(np.log10(np.abs(data_range)))

            min_data = round(min_data, -1*int(exponent))
            max_data = round(max_data, -1*int(exponent))

    # Define LEVELS and THRESHOLDS using the generate_contours function in utils
        if data_range >= 2.0:
            LEVELS, THRESHOLDS = utils.generate_contours(contour_dict, max_int_data, min_int_data)
        else:
            LEVELS, THRESHOLDS = utils.generate_contours(contour_dict, max_data, min_data)

    # Check flags and values provided by arguments
    # Check smoothing status
        if smooth_dict["smooth_flag"]:
            for entry in data:
                data[entry]['metadata']['smooth_flag'] = True
                if smooth_dict["sigma_override"]:
                    data[entry]['metadata']['sigma'] = smooth_dict["sigma"]
        else:
            for entry in data:
                data[entry]['metadata']['smooth_flag'] = False

    # Loop over entries in data and smooth data if required
        for entry in data:
            if data[entry]['metadata']['smooth_flag']:
                data[entry]['values'] = utils.filter_var_data(data[entry]['values'], data[entry]['metadata']['sigma'])

        # create colour palette
            if "colormap" in contour_dict:
                hex_palette = utils.create_color_palette(LEVELS, contour_dict["colormap"])
            else:
                hex_palette = ["#"+hexcode for hexcode in contour_dict["color_pal"] ]
        # Get features using generate_geojson function in contouring_par (parallel contouring)
            feature_collection = contouring_par.generate_geojson(var=data[entry]['values'], lat=data[entry]['lat'], lon=data[entry]['lon'], contours=LEVELS, thresholds=THRESHOLDS, metadata=data[entry]['metadata'], hex_palette=hex_palette, max_workers=parallel_dict["max_workers"], tolerance =simplify_dict["tolerance"])

            if 'site_lat' in data[entry]['metadata'] and 'site_lon' in data[entry]['metadata']:
                feature_collection = utils.add_points(feature_collection, [data[entry]['metadata']['site_lat']], [data[entry]['metadata']['site_lon']], {"name": "site location"})

        # Save output
            contouring_par.write_geojson(feature_collection, output_dir, input_file, entry, VAR_NAME, data[entry]['metadata'])

    else:
        print("There is no data to create a geojson for.")


if __name__ == "__main__":
    main()

