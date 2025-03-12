#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert variables from a variety of sources to GeoJSON

This script requires that the following modules are installed in the
Python environment within which you are running this script:
`netCDF4`, `numpy`, `scipy`, `skimage`, `wrf`, `pyart`, `satpy`,
`argparse`, `shapely`, `matplotlib`, `os`, `datetime`, `yaml`,
`json`.

This script was developed by The Forecasting Operations for Research 
Campaigns and Experiments (FORCE) from a CEMAC script for the FASTA
nowcasting project.

Authors:
  * Alexander Roberts <A.J.Roberts1@leeds.ac.uk>
  * Tamora D. James <T.D.James1@leeds.ac.uk>
  * Richard Rigby <R.Rigby@leeds.ac.uk>

"""

# Copyright (c) 2024 University of Leeds

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

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

def format_float(x):
    """
    Reduce precision of values to be stored as GeoJSON.

    Parameters
    ----------
    x : float
        Numeric value
    precision : integer
        Number of digits after decimal point to retain

    Returns
    -------
    float
        Float value with specified precision.

    """

    fmt_str = '{{:.{}f}}'.format(3)

    return float(fmt_str.format(x))

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
            contour_dict["contour_method"] = args.contour_method
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

    return args.input_file, args.output_dir, args.var_name, args.source, smooth_dict, contour_dict, level_dict

def contour_rect(im):
    '''
    Converts a trinary image (values of 0, 1 and 3) to represent values below min, within range and above max values respectively
    to line segements that describe the outline of the pixels on which data was supplied. 
   
    Parameters
    ----------
    im : array
        2D trinary mask array

    Returns
    -------
    lines : List
        List of line segements ready to be converted to contours
    '''

    lines = []
    pad = np.pad(im, [(1, 1), (1, 1)])  # zero padding

    im0_pos = np.maximum(np.diff(pad, n=1, axis=0)[:, 1:], 0) # positive horizontal difference
    im0_pos[im0_pos==3] = 0 # remove line segments where pixels that are over the max and under the min of range meet.
    im0_neg = np.minimum(np.diff(pad, n=1, axis=0)[:, 1:], 0) # negative horizontal difference
    im0_neg[im0_neg==-3] = 0 # remove line segments where pixels that are over the max and under the min of range meet.

    im1_pos = np.maximum(np.diff(pad, n=1, axis=1)[:, 1:], 0) # positive vertical difference
    im1_pos[im1_pos==3] = 0 # remove line segments where pixels that are over the max and under the min of range meet.
    im1_neg = np.minimum(np.diff(pad, n=1, axis=1)[:, 1:], 0) # negative vertical difference
    im1_neg[im1_neg==-3] = 0 # remove line segments where pixels that are over the max and under the min of range meet.

    # Process positive difference horizontal lines
    im0 = np.diff(im0_pos, n=1, axis=1)
    for i, row in enumerate(im0[:,:-1]):
        starts = []
        ends = []

        # Loop through the row to identify start and end points where differences exist
        count = 0
        for j in range(0, len(row)):
            if row[j] != 0:
                if len(starts) == len(ends):
                    starts.append(j)
                    count = count + row[j]
                else:
                    ends.append(j)
                    count = count + row[j]
                    if count > 0 :
                        starts.append(j)

        # Combine corresponding starts and ends into line segments
        for s, e in zip(starts, ends):
            lines.append(([i - 0.5, i - 0.5], [s + 0.5, e + 0.5]))

    # Process negative difference horizontal lines
    im0 = np.diff(im0_neg, n=1, axis=1)
    for i, row in enumerate(im0[:,:-1]):
        starts = []
        ends = []

        # Loop through the row to identify start and end points where differences exist
        count = 0
        for j in range(0, len(row)):
            if row[j] != 0:
                if len(starts) == len(ends):
                    starts.append(j)
                    count = count + row[j]
                else:
                    ends.append(j)
                    count = count + row[j]
                    if count < 0 :
                        starts.append(j)

        # Combine corresponding starts and ends into line segments
        for s, e in zip(starts, ends):
            lines.append(([i - 0.5, i - 0.5], [s + 0.5, e + 0.5]))

    # Process positive difference vertical lines
    im1 = np.diff(im1_pos, n=1, axis=0).T
    for i, row in enumerate(im1[:,:-1]):
        starts = []
        ends = []

        # Loop through the row to identify start and end points where differences exist
        count = 0
        for j in range(0, len(row)):
            if row[j] != 0:
                if len(starts) == len(ends):
                    starts.append(j)
                    count = row[j]
                else:
                    ends.append(j)
                    count = count + row[j]
                    if count > 0 :
                        starts.append(j)

        # Combine corresponding starts and ends into line segments
        for s, e in zip(starts, ends):
            lines.append(([s - 0.5, e - 0.5], [i + 0.5, i + 0.5]))

    # Process negative difference vertical lines
    im1 = np.diff(im1_neg, n=1, axis=0).T
    for i, row in enumerate(im1[:,:-1]):
        starts = []
        ends = []

        # Loop through the row to identify start and end points where differences exist
        count = 0
        for j in range(0, len(row)):
            if row[j] != 0:
                if len(starts) == len(ends):
                    starts.append(j)
                    count = row[j]
                else:
                    ends.append(j)
                    count = count + row[j]
                    if count < 0 :
                        starts.append(j)

        # Combine corresponding starts and ends into line segments
        for s, e in zip(starts, ends):
            lines.append(([s - 0.5, e - 0.5], [i + 0.5, i + 0.5]))

    return lines

def add_intervening_points(start, end):
    '''
    Takes the start and end point of a line segement generated by contour_rect (only horizontal and vertical)
    and returns both the start and end points and all the intervening points.

    Parameters
    ----------
    start: List
        List contianing the coordinates of the start point of the line segment.
    end: List
        List containing the coordinates of the end point of the line segment.

    Returns
    -------
    points: List
        List of coordinates of start and end points and all points between (step of 1).
    '''

    points = []
    
    if start[0] == end[0]:  # Vertical line
        y_range = np.linspace(start[1], end[1], int(abs(end[1]-start[1]))+1)
        for y in y_range:
            points.append([start[0], y])
    elif start[1] == end[1]:  # Horizontal line
        x_range = np.linspace(start[0], end[0], int(abs(end[0]-start[0]))+1)
        for x in x_range:
            points.append([x, start[1]])
    
    return points

def find_next_lines(current_point, line_list):
    '''
    Find the next line segment produced by contour_rect connected to the current point.

    Parameters
    ----------
    current_point : tuple
        Tuple containing the coordinates of the point being searched for in line_list
    line_list : list
        List containing line segments produced by contour_rect

    Returns
    -------
    len(list_indices) : integer
        length of the list of line indices which match the current point
    list_indices: list
        List of indices relating to line segments that match the criteria
    '''

    list_indices = []

    for i, line in enumerate(line_list):
        # Check if the current_point matches any endpoint of the line
        if (line[1][1], line[0][1]) == current_point or (line[1][0], line[0][0]) == current_point:
            list_indices.append(i)

    return len(list_indices), list_indices

def find_num_connections(test_line, line_list):
    '''
    Find the number of connections a line segment has at either end when compared with a list of line segments
    This is not called in general use but is a useful function for development and bug detection.

    Parameters
    ----------
    test_line: List
        List that contains coordinates that describe a line segment
    line_list: List
        List of line segments like that one being tested generated contour_rect

    Returns
    -------
    len(list_indices1): integer
        Length of the list of indices that match the coordinates of end 1 of the tetsed line segment
    len(list_indices2): integer
        Length of the list of indices that match the coordinates of end 2 of the tetsed line segment
    '''

    list_indices1=[]
    list_indices2=[]

    # end 1
    current_point = (test_line[1][0], test_line[0][0])
    for i, line in enumerate(line_list):
        if (line[1][1], line[0][1]) == current_point or (line[1][0], line[0][0]) == current_point:
            list_indices1.append(i)

    # end 2
    current_point = (test_line[1][1], test_line[0][1])
    for i, line in enumerate(line_list):
        if (line[1][1], line[0][1]) == current_point or (line[1][0], line[0][0]) == current_point:
            list_indices2.append(i)

    return len(list_indices1), len(list_indices2)

def order_lines_into_contours(lines, cont_mask, mask):
    '''
    Takes the line segments generated by contour_rect and organizes the line segments into geojson compliant contour structures

    Parameters
    ----------
    lines: list
        List of line segments generated by contour_rect
    cont_mask: array
        Mask of contoured regions with contiguous regions split into labelled values
    mask: array
        Trinary mask of regions being contoured values of 0 indicate values below the valid contoured range, 1 between min and max values and 3 over the max value.

    Returns
    -------
    contours : list
        list of contours generated from line segments.

    '''

    contours = []

    while lines:    # Keep creating contours while line segments have not yet been used.
        for i in np.arange(0, len(lines)-1, 1):
            connections1, connections2 = find_num_connections(lines[i], lines)    # Make sure the starting points are not adjacent to branch point (this could complicate things later).
            if connections1 == 2 and connections2 == 2:
                starting_index = i
                break

        current_line = lines.pop(starting_index)    # Select the starting line segement and work out its orientation
        if current_line[0][0] == current_line[0][1]:
            orientation_flag = 0
        elif current_line[1][0] == current_line[1][1]:
            orientation_flag = 1

        # Start a new contour
        contour = []

        # Add points from starting line to contour and create start point and the current contour end (this will be updated)
        contour = add_intervening_points([current_line[1][0], current_line[0][0]], [current_line[1][1], current_line[0][1]])
        current_mask_val = check_adjoining_mask((contour[0], contour[1]), cont_mask)
        start_point = (current_line[1][0], current_line[0][0])  # Save the starting point
        contour_end = (current_line[1][1], current_line[0][1])  # Track the current end point

        # Until the contour end matches the contour start keep adding sections to the contour.       
        while True:
            # Find number of lines that share the current contour end point.
            num_lines, list_indices = find_next_lines(contour_end, lines)

            if num_lines == 1: #If only one other line segment shares the same coordinates as the contour_end add that line.
                new_line = lines.pop(list_indices[0])
                # Calculate orientation of new line segment
                if new_line[0][0] == new_line[0][1]:
                    orientation_flag = 0
                elif new_line[1][0] == new_line[1][1]:
                    orientation_flag = 1

                # Add new line to contour and update end point
                if (new_line[1][0], new_line[0][0]) == contour_end:
                    contour += add_intervening_points([new_line[1][0], new_line[0][0]], [new_line[1][1], new_line[0][1]])
                    contour_end = (new_line[1][1], new_line[0][1])
                else:
                    contour += add_intervening_points([new_line[1][1], new_line[0][1]], [new_line[1][0], new_line[0][0]])
                    contour_end = (new_line[1][0], new_line[0][0])

                print(contour_end)
                # Check if the end matches the start point and break while loop if it does
                if contour_end == start_point:
                    break

            else: # If more than one line shares the contour_end coordinate a branch has been found
                contour_loop, loop_lines, orientation_flag, unresolved_loop = identify_loop_path(start_point, contour_end, lines[:], orientation_flag, current_mask_val, cont_mask, False)
                contour.extend(contour_loop)
                contour_end = (contour[-1][0], contour[-1][1])
                # remove all the line segments identified as part of the returned loop from the list of lines.
                for line in loop_lines:
                    if line in lines:
                        lines.remove(line)

                # Check if the end matches the start point and break while loop if it does
                if contour_end == start_point:
                     break

        # When a contour has been identified check whether it is an exterior or interior contour and its winding direction and fix if needed
        winding_dir, ext_int = check_exterior_interior(contour, mask)

        if ext_int == "interior":
            if winding_dir == "cw":
                contour = contour[::-1]

        if ext_int == "exterior":
            if winding_dir == "ccw":
                contour = contour[::-1]

        # add each contour to the contours list
        contour = remove_adjacent_duplicates(contour)
        contours.append(np.array(contour))

    return contours


def order_lines_into_contours2(lines, cont_mask, inv_cont_mask, mask):
    '''
    Takes the line segments generated by contour_rect and organizes the line segments into geojson compliant contour structures

    Parameters
    ----------
    lines: list
        List of line segments generated by contour_rect
    cont_mask: array
        Mask of contoured regions with contiguous regions split into labelled values
    mask: array
        Trinary mask of regions being contoured values of 0 indicate values below the valid contoured range, 1 between min and max values and 3 over the max value.

    Returns
    -------
    contours : list
        list of contours generated from line segments.

    '''

    import matplotlib.pyplot as plt
    import random
    import time

    # Create a dictionary of all the points where line segments end. Line segements are stored in both the start and end points of the dictiionary. This will enable the path of the contour to be mapped without refeering to the full list.    
   
    start = time.time()
    line_dict = {}
    for line in lines:
        end_point1 = (line[0][0],line[1][0])
        end_point2 = (line[0][1],line[1][1])
       
        if end_point1 in line_dict:
            line_dict[end_point1].append(line)
        else:
            line_dict[end_point1] = [line]

        if end_point2 in line_dict:
            line_dict[end_point2].append(line)
        else:
            line_dict[end_point2] = [line]

    end = time.time()
    elapsed = end - start
    print("Time to produce line segment dictionary: ", elapsed) 

    # Create empty contours list
    contours = []

    # While there are still entries in the line_dictionary continue to create new contours.
    while bool(line_dict):

        print(len(line_dict.keys()))

        # Select random starting point
        random_point = random.choice(list(line_dict.items()))

        start_point = random_point[0]
        value = random_point[1]

        current_line = value[0]
        if start_point == (current_line[0][0],current_line[1][0]):
            contour_end = (current_line[0][1],current_line[1][1])
        if start_point == (current_line[0][1],current_line[1][1]):
            contour_end = (current_line[0][0],current_line[1][0])

        line_dict[start_point].remove(current_line)
        line_dict[contour_end].remove(current_line)

        # Calculate the orientation flag for the starting line
        if current_line[0][0] == current_line[0][1]:
            orientation_flag = 0
        elif current_line[1][0] == current_line[1][1]:
            orientation_flag = 1

        contour = []

        # Add the interventing points from start_point to contour_end to the current contour
        contour = add_intervening_points([start_point[0], start_point[1]], [contour_end[0], contour_end[1]])
        # Calculate which region is being enclosed by the current contour.
        current_mask_val = check_adjoining_mask((contour[0], contour[1]), cont_mask.T)
        current_inv_mask_val = check_adjoining_mask((contour[0], contour[1]), inv_cont_mask.T)

        while True:
            if len(line_dict[contour_end]) == 1:
                new_line = line_dict[contour_end][0]
                if new_line[0][0] == new_line[0][1]:
                    orientation_flag = 0
                elif new_line[1][0] == new_line[1][1]:
                    orientation_flag = 1

                if (new_line[0][0], new_line[1][0]) == contour_end:
                    line_start = (new_line[0][0], new_line[1][0])
                    contour_end = (new_line[0][1], new_line[1][1])
                    contour += add_intervening_points([line_start[0],line_start[1]], [contour_end[0], contour_end[1]])
                    line_dict[line_start].remove(new_line)
                    line_dict[contour_end].remove(new_line)

                    if not line_dict[line_start]:
                        del line_dict[line_start]

                else:
                    line_start = (new_line[0][1], new_line[1][1])
                    contour_end = (new_line[0][0], new_line[1][0])
                    contour += add_intervening_points([line_start[0],line_start[1]], [contour_end[0], contour_end[1]])
                    line_dict[line_start].remove(new_line)
                    line_dict[contour_end].remove(new_line)

                    if not line_dict[line_start]:
                        del line_dict[line_start]

                # Check if the end matches the start point and break while loop if it does
                if contour_end == start_point:
                    if not line_dict[contour_end]:
                        del line_dict[contour_end]
                    break
                           
                    
            else: # If more than one line shares the contour_end coordinate a branch has been found

                inv_val_list = [current_inv_mask_val]
                contour_loop, updated_line_dict, orientation_flag, unresolved_loop = identify_loop_path2(start_point, contour_end, line_dict, orientation_flag, current_mask_val, current_inv_mask_val, inv_val_list, cont_mask, inv_cont_mask, False)
                contour.extend(contour_loop)
                contour_end = (contour[-1][0], contour[-1][1])
                line_dict = updated_line_dict

                # Check if the end matches the start point and break while loop if it does
                if contour_end == start_point:
                    if not line_dict[contour_end]:
                        del line_dict[contour_end]
                    break

        # When a contour has been identified check whether it is an exterior or interior contour and its winding direction and fix if needed
        winding_dir, ext_int = check_exterior_interior(contour, mask.T)

        if ext_int == "interior":
            if winding_dir == "cw":
                contour = contour[::-1]

        if ext_int == "exterior":
            if winding_dir == "ccw":
                contour = contour[::-1]

        # add each contour to the contours list
        contour = remove_adjacent_duplicates(contour)
        contours.append(np.array(contour))

    return contours

def identify_loop_path(start_point, branch_point, line_list, orientation_flag, current_mask_val, cont_mask, previously_unresolved_loop):
    '''
    When a branch is identifed in order_lines_into_contours then it must be determined if the branch is valid (both possible routes
    are adjacent to the correct contiguous masked value in cont_mask) or should be treated as a single path. If a banch is valid then a
    loop needs to be identifed and the loop coordinates need to be inserted into the contour before the contour can continue to be traced.
    This function can call itself for numerous iterations due tothe possibility of loops starting within loops.

    Parameters
    ----------
    start_point: tuple
        Tuple of start point coordinates
    branch_point: tuple
        Tuple of branch point coordinates
    line_list: List
        List of line segments (created by contour_rect) that contain all possible contour paths
    orientation_flag : integer
        0 or 1 to indicate the orientation of the last line segment added to the contour, the next line segment must have a different orientation_flag
    current_mask_val : integer
        The label value being enclosed curently (based on cont_mask)
    cont_mask : array
        Mask of contoured regions with contiguous regions split into labelled values.
    previously_unresolved_loop: Boolean
        True or False flag to indicate whether any of the times identify_loop_path was called earlier on (when branches are found within loops) have 
        as yet unsearched paths. If this flag is false and the start of the whole contour is found then the process can stopand the full contour 
        can be returned.

    Returns
    -------
    contour_path: list
        List of contour points to be returned and added to main contour path
    loop_lines: list
        List of loop lines that have been used to create the contour path, passing these is important as they need to be removed from list of lines
        avaialable to continue contours.
    orientation_flag: integer
        0 or 1 to indicate the orientation of the last line segment added to the contour, the next line segment must have a different orientation_flag
    unresolved_loop: Boolean
        Flag to indicate whether a viable path has been returned.
    '''

    # set the found loop flag to false
    found_loop = False

    path_indices = []

    # identify the number of lines and the indices of siad lines in the line_list
    num_lines, list_indices = find_next_lines(branch_point, line_list)

    # for each line returned identify the mask value that it is adjacent to, return the orientation of the line and return the possible path indices.
    for i in np.arange(num_lines):
        current_line = line_list[list_indices[i]]
        test_line = []
        test_line += add_intervening_points([current_line[1][0], current_line[0][0]], [current_line[1][1], current_line[0][1]])
        mask_val = check_adjoining_mask((test_line[0], test_line[1]), cont_mask)
        if current_line[0][0] == current_line[0][1]:
            new_orientation_flag = 0
        elif current_line[1][0] == current_line[1][1]:
            new_orientation_flag = 1

        if new_orientation_flag != orientation_flag:
            if mask_val == current_mask_val:
                path_indices.append(list_indices[i])

    # check to see whether the branch is a real branch or not (lines surrounding different contiguous mask values should not be followed)
    len_path_indices = len(path_indices)

    for index in path_indices:
        # If more than one path has to be checked then the unresolved loop flag should be used (this is set to False if the loop is identified first time).
        if len_path_indices > 1:
            unresolved_loop = True
        else:
            unresolved_loop = False

        line_list_temp = line_list[:]  # make copy of line_list for this iteration of the path indices
        loop_lines = []
        contour_path = []
        contour_end = branch_point

        new_line = line_list_temp.pop(index) # select new line
        loop_lines.append(new_line)

        # Identify orientation of new line
        if new_line[0][0] == new_line[0][1]:
            orientation_flag = 0
        elif new_line[1][0] == new_line[1][1]:
            orientation_flag = 1

        # add points of new line to contour path and update contour_end
        if (new_line[1][0], new_line[0][0]) == contour_end:
            contour_path += add_intervening_points([new_line[1][0], new_line[0][0]], [new_line[1][1], new_line[0][1]])
            contour_end = (new_line[1][1], new_line[0][1])
        else:
            contour_path += add_intervening_points([new_line[1][1], new_line[0][1]], [new_line[1][0], new_line[0][0]])
            contour_end = (new_line[1][0], new_line[0][0])

        # until the loop has been identified continue the while loop
        while True:
            # identify the next lines
            num_lines, list_indices = find_next_lines(contour_end, line_list_temp)
            # If no more matching lines are available it is likely that the new contour end matches the start point of the contour
            # The resolved/unresolved status of the loop should be considered if the start point has been found.
            if num_lines == 0:
                if contour_end == branch_point:
                    found_loop = True
                    unresolved_loop = False
                    break
                if contour_end == start_point:
                    if not unresolved_loop and not previously_unresolved_loop:
                        found_loop=True
                        unresolved_loop = False
                        break
                    else:
                        found_loop = False
                        unresolved_loop = True
                        len_path_indices = len_path_indices - 1
                        break

            # If only one line matches the contour_end point then this line can be added to the contour_path, if the new contour end matches the branch
            # point or the start point then it is possible that this the end of the loop/contour. The resolved/unresolved loop flag informs whether this si the case.
            if num_lines == 1:
                new_line = line_list_temp.pop(list_indices[0])
                loop_lines.append(new_line)

                if new_line[0][0] == new_line[0][1]:
                    orientation_flag = 0
                elif new_line[1][0] == new_line[1][1]:
                    orientation_flag = 1

                if (new_line[1][0], new_line[0][0]) == contour_end:
                    contour_path += add_intervening_points([new_line[1][0], new_line[0][0]], [new_line[1][1], new_line[0][1]])
                    contour_end = (new_line[1][1], new_line[0][1])
                else:
                    contour_path += add_intervening_points([new_line[1][1], new_line[0][1]], [new_line[1][0], new_line[0][0]])
                    contour_end = (new_line[1][0], new_line[0][0])

                if contour_end == branch_point:
                    found_loop = True
                    unresolved_loop = False
                    break
                if contour_end == start_point:
                    if not unresolved_loop and not previously_unresolved_loop:
                        found_loop = True
                        break
                    else:
                        found_loop = False
                        unresolved_loop = True
                        len_path_indices = len_path_indices - 1
                        break

            # If the num_lines values returned is greater than 1 then a secondary branch has been found. 
            else:
                if unresolved_loop == True or previously_unresolved_loop == True:
                   combined_unresolved_loop = True
                else:
                   combined_unresolved_loop = False
                # This function now calls itself, this time an unresolved flag is returned to break the loop and allow for the chacking of the other paths 
                contour_loop, loop_lines2, orientation_flag, unresolved = identify_loop_path(start_point, contour_end, line_list_temp[:], orientation_flag, current_mask_val, cont_mask, combined_unresolved_loop)
                if unresolved:
                    found_loop = False
                    unresolved_loop = True
                    break
                else:
                    # The loop_lines from the identify_loop_path iterations need to be combined so that if a valid path is returned all the line segments used can be removed at the topmost layer. Similarly the contour path is extended by the paths found by identify_loop_path.
                    loop_lines.extend(loop_lines2)
                    contour_path.extend(contour_loop)
                    contour_end = (contour_path[-1][0], contour_path[-1][1])

                    for line in loop_lines2:
                        if line in line_list_temp:
                            line_list_temp.remove(line)
                    # Check if the end point is the start point and whether the unresolved flags are approriate
                    if contour_end == start_point:
                        if not unresolved_loop and not previously_unresolved_loop:
                            found_loop=True
                            unresolved_loop = False
                            break
                        else:
                            found_loop = False
                            len_path_indices = len_path_indices - 1
                            unresolved_loop = True
                        break

        # If a loop (even across multiple iterations of identify_loop_path has been found then break the while loop and return the contours, loop_linesm orientation flag andunresolved loop flag.
        if found_loop == True:
            break

    return contour_path, loop_lines, orientation_flag, unresolved_loop


def identify_loop_path2(start_point, branch_point, line_dict, orientation_flag, current_mask_val, current_inv_mask_val, inv_val_list, cont_mask, inv_cont_mask, previously_unresolved_loop):
    '''
    When a branch is identifed in order_lines_into_contours then it must be determined if the branch is valid (both possible routes
    are adjacent to the correct contiguous masked value in cont_mask) or should be treated as a single path. If a banch is valid then a
    loop needs to be identifed and the loop coordinates need to be inserted into the contour before the contour can continue to be traced.
    This function can call itself for numerous iterations due tothe possibility of loops starting within loops.

    Parameters
    ----------
    start_point: tuple
        Tuple of start point coordinates
    branch_point: tuple
        Tuple of branch point coordinates
    line_list: List
        List of line segments (created by contour_rect) that contain all possible contour paths
    orientation_flag : integer
        0 or 1 to indicate the orientation of the last line segment added to the contour, the next line segment must have a different orientation_flag
    current_mask_val : integer
        The label value being enclosed curently (based on cont_mask)
    cont_mask : array
        Mask of contoured regions with contiguous regions split into labelled values.
    previously_unresolved_loop: Boolean
        True or False flag to indicate whether any of the times identify_loop_path was called earlier on (when branches are found within loops) have 
        as yet unsearched paths. If this flag is false and the start of the whole contour is found then the process can stopand the full contour 
        can be returned.

    Returns
    -------
    contour_path: list
        List of contour points to be returned and added to main contour path
    loop_lines: list
        List of loop lines that have been used to create the contour path, passing these is important as they need to be removed from list of lines
        avaialable to continue contours.
    orientation_flag: integer
        0 or 1 to indicate the orientation of the last line segment added to the contour, the next line segment must have a different orientation_flag
    unresolved_loop: Boolean
        Flag to indicate whether a viable path has been returned.
    '''
    import copy
    import matplotlib.pyplot as plt

    # set the found loop flag to false
    found_loop = False

    possible_routes = []
    possible_routes_inv_mask_vals = []

    for line in line_dict[branch_point]:
        test_start = (line[0][0], line[1][0])
        test_end = (line[0][1], line[1][1])
        test_line = []
        test_line += add_intervening_points([test_start[0],test_start[1]], [test_end[0],test_end[1]])
        mask_val = check_adjoining_mask((test_line[0], test_line[1]), cont_mask.T)
        inv_mask_val = check_adjoining_mask((test_line[0], test_line[1]), inv_cont_mask.T)
        if line[0][0] == line[0][1]:
            new_orientation_flag = 0
        elif line[1][0] == line[1][1]:
            new_orientation_flag = 1

        if new_orientation_flag != orientation_flag:
            if mask_val == current_mask_val:
                possible_routes.append(line)
                possible_routes_inv_mask_vals.append(inv_mask_val)

    if len(possible_routes) > 1:
        for i in range(len(possible_routes)-1):
            if possible_routes_inv_mask_vals[i] in inv_val_list:
                possible_routes.pop(i)
                possible_routes_inv_mask_vals.pop(i)
            else:
                inv_val_list.append(possible_routes_inv_mask_vals[i])

    print("NUMBER OF POSSIBLE ROUTES: ", len(possible_routes))

    if len(possible_routes) > 1:
        unresolved_loop = True
    else:
        unresolved_loop = False

    for line in possible_routes:
        updated_line_dict = copy.deepcopy(line_dict)

        contour_path = []
        contour_end = branch_point

        new_line = line

        if line[0][0] == line[0][1]:
            orientation_flag = 0
        elif line[1][0] == line[1][1]:
            orientation_flag = 1

        if (new_line[0][0], new_line[1][0]) == contour_end:
            line_start = (new_line[0][0], new_line[1][0])
            contour_end = (new_line[0][1], new_line[1][1])
            contour_path += add_intervening_points([line_start[0],line_start[1]], [contour_end[0], contour_end[1]])
            updated_line_dict[line_start].remove(new_line)
            updated_line_dict[contour_end].remove(new_line)

            if new_line[0][0] == new_line[0][1]:
                orientation_flag = 0
            elif new_line[1][0] == new_line[1][1]:
                orientation_flag = 1

            if not updated_line_dict[line_start]:
                del updated_line_dict[line_start]

        else:
            line_start = (new_line[0][1], new_line[1][1])
            contour_end = (new_line[0][0], new_line[1][0])
            contour_path += add_intervening_points([line_start[0],line_start[1]], [contour_end[0], contour_end[1]])
            updated_line_dict[line_start].remove(new_line)
            updated_line_dict[contour_end].remove(new_line)

            if new_line[0][0] == new_line[0][1]:
                orientation_flag = 0
            elif new_line[1][0] == new_line[1][1]:
                orientation_flag = 1

            if not updated_line_dict[line_start]:
                del updated_line_dict[line_start]


        while True:
            if contour_end == branch_point:
                found_loop = True
                unresolved_loop = False
                return contour_path, updated_line_dict, orientation_flag, unresolved_loop

            if contour_end == start_point:
                if not unresolved_loop and not previously_unresolved_loop:
                    found_loop=True
                    unresolved_loop = False
                    return contour_path, updated_line_dict, orientation_flag, unresolved_loop
                else:
                    found_loop = False
                    unresolved_loop = True
                    break

            if len(updated_line_dict[contour_end]) == 1:
                new_line = updated_line_dict[contour_end][0]
                if new_line[0][0] == new_line[0][1]:
                    orientation_flag = 0
                elif new_line[1][0] == new_line[1][1]:
                    orientation_flag = 1

                if (new_line[0][0], new_line[1][0]) == contour_end:
                    line_start = (new_line[0][0], new_line[1][0])
                    contour_end = (new_line[0][1], new_line[1][1])
                    contour_path += add_intervening_points([line_start[0],line_start[1]], [contour_end[0], contour_end[1]])
                    updated_line_dict[line_start].remove(new_line)
                    updated_line_dict[contour_end].remove(new_line)

                    if not updated_line_dict[line_start]:
                        del updated_line_dict[line_start]

                else:
                    line_start = (new_line[0][1], new_line[1][1])
                    contour_end = (new_line[0][0], new_line[1][0])
                    contour_path += add_intervening_points([line_start[0],line_start[1]], [contour_end[0], contour_end[1]])
                    updated_line_dict[line_start].remove(new_line)
                    updated_line_dict[contour_end].remove(new_line)

                    if not updated_line_dict[line_start]:
                        del updated_line_dict[line_start]

            else: # If more than one line shares the contour_end coordinate a branch has been found
                if unresolved_loop == True or previously_unresolved_loop == True:
                   combined_unresolved_loop = True
                else:
                   combined_unresolved_loop = False
                # This function now calls itself, this time an unresolved flag is returned to break the loop and allow for the checking of the other paths 
                contour_loop, updated_line_dict2, orientation_flag, unresolved = identify_loop_path2(start_point, contour_end, updated_line_dict, orientation_flag, current_mask_val, current_inv_mask_val, inv_val_list, cont_mask, inv_cont_mask, combined_unresolved_loop)
                if unresolved:
                    found_loop = False
                    unresolved_loop = True
                    break
                else:
                    contour_path.extend(contour_loop)
                    contour_end = (contour_path[-1][0], contour_path[-1][1])
                    updated_line_dict = updated_line_dict2
                    # Check if the end point is the start point and whether the unresolved flags are approriate
                    if contour_end == branch_point:
                        if not unresolved_loop and not previously_unresolved_loop:
                            found_loop=True
                            unresolved_loop = False
                            return contour_path, updated_line_dict, orientation_flag, unresolved_loop
                        else:
                            found_loop = False
                            unresolved_loop = True
                            continue

    return contour_path, updated_line_dict, orientation_flag, unresolved_loop

def check_adjoining_mask(first_step, cont_mask):
    '''
    Check which contiguous region of a mask is being enclosed by the current contour (using the first step of the contour).

    Parameters
    ----------
    first_step: tuple
        Tuple of coordinate lists for 2 points in a contour.
    cont_mask: array
        Mask of contoured regions with contiguous regions split into labelled values.

    Returns
    -------
    current_mask_val : integer
        Label of current contiguous region being enclosed by a contour.
    '''

    # Check if line is horizontal or vertical
    if first_step[0][1] == first_step[1][1]:
        x = int((first_step[0][0]+first_step[1][0])/2.0) # calculate middle point
        y1 = int(first_step[0][1]-0.5) # offset equal coordinates by + and - 0.5 to give pixel coords.
        y2 = int(first_step[0][1]+0.5)

        if cont_mask[y1, x] != 0:
            current_mask_val = cont_mask[y1, x] # check which side of the line is the masked region and return cont_mask value
        else:
            current_mask_val = cont_mask[y2, x]

    else:
        x1 = int(first_step[0][0]-0.5) # offset equal coordinates by + and - 0.5 to give pixel coords.
        x2 = int(first_step[0][0]+0.5)
        y = int((first_step[0][1] + first_step[1][1])/2.0) # calculate middle point
    
        if cont_mask[y, x1] != 0:
            current_mask_val = cont_mask[y, x1] # check which side of the line is the masked region and return cont_mask value
        else:
            current_mask_val = cont_mask[y, x2]

    return current_mask_val

def remove_adjacent_duplicates(contour):
    '''
    Remove adjacent duplicate points from the contour, except for the first and last points.

    Parameters
    ----------
    contour: list
        Contour created from vertical and horizontal line segments

    Returns
    -------
    cleaned_contour: list
        Contour created from vertical and horizontal line segments but with duplicates removed
    '''

    # Start with the first point
    cleaned_contour = [contour[0]]
    
    # Iterate through the contour, ignoring the first and last points
    for i in range(1, len(contour) - 1):
        if contour[i] != contour[i - 1]:  # Add only if the point is not equal to the previous one
            cleaned_contour.append(contour[i])

    # Add the last point (should match the start for closed contours)
    cleaned_contour.append(contour[-1])

    return cleaned_contour

def check_exterior_interior(contour, mask):
    '''
    Point that is given is a part of the contour iteslf, therefore a neightbourhood approach is taken to first identify
    a point that is inside the contour polygon. Then the exterior/interior nature of the contour is derived based on
    whether the enclosed value (close to the start point) meets the thresholded criteria (equal to 1 on the mask).

    Parameters
    ----------
    contour : list of tuples
        List of (x, y) coordinates representing the contour.
    mask : array
        2d array mask of valid values (equal to 1)

    Returns
    -------
    winding_dir : string
        cw and ccw relating to clockwise and counter-clockwise repsectively
    ext_int : string
        exterior or interior relating to whether a contour is surrounding the masked area or is a hole in it.
    '''

    first_point = contour[0]
    b = {}
    b["ctr"] = contour

    ext_int = ""

    for i in np.arange(int(first_point[0]-0.5), int(first_point[0]+0.5)+1, 1):
        for j in np.arange(int(first_point[1]-0.5), int(first_point[1]+0.5)+1, 1):
            a = {}
            a["ctr"] = np.array([[i, j]])
            if is_enclosed_by(a, b) and mask[j,i] == 1 :
                ext_int = "exterior"
                break
            else:
                ext_int = "interior"
        else:
            continue
        break

    winding_dir = contour_winding_direction(contour)

    return winding_dir, ext_int

def contour_winding_direction(contour):
    '''
    Calculate the signed area of a contour. Positive area means counterclockwise winding,
    negative area means clockwise winding.
    
    Parameters
    ----------
    contour : list of tuples
        List of (x, y) coordinates representing the contour.
        
    Returns
    -------
    direction : string
        'ccw' for counterclockwise, 'cw' for clockwise.
    '''

    x, y = zip(*contour)
    n = len(contour)
    
    # Shoelace formula for polygon area
    area = 0.5 * sum(x[i] * y[(i + 1) % n] - x[(i + 1) % n] * y[i] for i in range(n))
    
    if area > 0:
        return 'ccw'  # Counterclockwise
    else:
        return 'cw'   # Clockwise

def pixel_outline_contour(data, min_val, max_val):
    '''
    Converts a data supplied with max and min values to properly formatted contour paths.

    Parameters
    ----------
    data : array
        2d array of data to be contoured
    min_val: float
        
    max_val: float
        

    Returns
    -------
    contours : List
        List of contours in the same format as other contour finding algorthms.
    mask : array
        Trinary mask of data array with values of 0, 1 and 3
    '''

    import scipy
    import time

    # Create a trinary mask (values of 0, 1 and 3 for values below min, between min and max and above min respectively) needed for contour_rect function.

    mask = np.zeros_like(data.T, dtype=int)
    mask[data.T < min_val] = 0
    mask[(data.T >= min_val) & (data.T < max_val)] = 1
    mask[data.T >= max_val] = 3

    # create binary mask of valid and invalid areas
    bin_mask = np.zeros_like(data.T, dtype=int)
    bin_mask[(data.T >= min_val) & (data.T < max_val)] = 1
    cont_mask, num_features = scipy.ndimage.label(bin_mask)

    # Get the lines that outline the regions
    start = time.time()
    line_segments = contour_rect(mask)
    end = time.time()
    elapsed = end - start

    # retrieve contours based off the line segments and the mask
    contours =  order_lines_into_contours(line_segments[:], cont_mask, mask)

    return contours, mask

def pixel_outline_contour2(data, min_val, max_val):
    '''
    Converts a data supplied with max and min values to properly formatted contour paths.

    Parameters
    ----------
    data : array
        2d array of data to be contoured
    min_val: float
        
    max_val: float
        

    Returns
    -------
    contours : List
        List of contours in the same format as other contour finding algorthms.
    mask : array
        Trinary mask of data array with values of 0, 1 and 3
    '''

    import scipy
    import time

    # Create a trinary mask (values of 0, 1 and 3 for values below min, between min and max and above min respectively) needed for contour_rect function.

    start = time.time()
    mask = np.zeros_like(data, dtype=int)
    mask[data < min_val] = 0
    mask[(data >= min_val) & (data < max_val)] = 1
    mask[data >= max_val] = 3
    end = time.time()
    elapsed = end - start
    print("Time to produce mask: ", elapsed)

    # create binary mask of valid and invalid areas
    start = time.time()
    bin_mask = np.zeros_like(data, dtype=int)
    bin_mask[(data >= min_val) & (data < max_val)] = 1
    cont_mask, num_features = scipy.ndimage.label(bin_mask)
    inv_bin_mask = np.logical_not(bin_mask).astype(int)
    inv_cont_mask, inv_num_features = scipy.ndimage.label(inv_bin_mask)
    end = time.time()
    elapsed = end - start
    print("Time to produce cont and inv cont masks: ", elapsed)

    # Get the lines that outline the regions
    start = time.time()
    line_segments = contour_rect(mask)
    end = time.time()
    elapsed = end - start
    print("Time to produce line segments: ", elapsed)

    # retrieve contours based off the line segments and the mask

    contours =  order_lines_into_contours2(line_segments[:], cont_mask, inv_cont_mask, mask)

    return contours, mask

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
            threshold_value = i 
            thresholds.append(threshold_value)
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
            grid_id = int(extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID'])
            sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')['SIMULATION_START_DATE']
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
        CONTOURS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)

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
            grid_id = int(extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID'])
            sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')['SIMULATION_START_DATE']
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
        CONTOURS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)

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
            grid_id = int(extract_global_attrs(wrf_in, 'GRID_ID')['GRID_ID'])
            sim_start_time = extract_global_attrs(wrf_in, 'SIMULATION_START_DATE')['SIMULATION_START_DATE']
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
        CONTOURS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)

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
    CONTOURS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)

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

    count = 0

    for i in np.arange(0, len(times), 1):
        for j in np.arange(0, len(levels), 1):

            level_type = "H"+levels_str[j]
            valid_time = times_str[i]
            units = "undefined"

            entry_name = f"entry{count:03d}"

            data_temp = data_temp4d[i,j,:,:]

            # Fill variables in empty frame around data with the current lowest value in the dataset
            data_min = np.amin(data_temp)
            edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
            edge_min = np.amin(edge_list)
            data_step = THRESHOLDS[1]-THRESHOLDS[0]
            data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
            data[ data == 0.0 ] =  edge_min - data_step
            data[1:-1,1:-1] = data_temp

            data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'sim_start_time': sim_start_time, 'valid_time': valid_time, 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : rec_sigma, 'origin_lats' : olats, 'origin_lons' : olons, 'origin_levels' : olvls, 'origin_times' : otims}}

            count = count + 1
    
    HYSPLIT_in.close()

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
    lat = np.ma.array(
        np.zeros((np.shape(lat_temp)[0]+2, np.shape(lat_temp)[1]+2)),
        mask=False
    )
    lat[1:-1,1:-1] = lat_temp
    lat[0,:] = lat[1,:]-(lat[2,:]-lat[1,:])
    lat[-1,:] = lat[-2,:] + (lat[-2,:]-lat[-3,:])
    lat[:,0] = lat[:,1]-(lat[:,2]-lat[:,1])
    lat[:,-1] = lat[:,-2] + (lat[:,-2]-lat[:,-3])
    lat.mask[lat < -90] = True

    # update lon values:
    lon = np.ma.array(
        np.zeros((np.shape(lon_temp)[0]+2, np.shape(lon_temp)[1]+2)),
        mask=False
    )
    lon[1:-1,1:-1] = lon_temp
    lon[0,:] = lon[1,:]-(lon[2,:]-lon[1,:])
    lon[-1,:] = lon[-2,:] + (lon[-2,:]-lon[-3,:])
    lon[:,0] = lon[:,1]-(lon[:,2]-lon[:,1])
    lon[:,-1] = lon[:,-2] + (lon[:,-2]-lon[:,-3])
    lon.mask[lon < -180] = True

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
    CONTOURS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)

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

    # Fill variables in empty frame around data with the current lowest value in the dataset
    data_min = np.amin(data_temp)
    edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
    edge_min = np.amin(edge_list)
    data_step = THRESHOLDS[1]-THRESHOLDS[0]
    data = np.zeros((np.shape(data_temp)[0]+2, np.shape(data_temp)[1]+2))
    data[ data == 0.0 ] =  edge_min - data_step
    data[1:-1,1:-1] = data_temp
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
    CONTOURS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)

    # Add additional frame around where data is present and populate with lat and lon values using finite difference approach

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

    for i in np.arange(0, np.shape(var_array)[time_index], 1):
        # Fill variables in empty frame around data with the current lowest value in the dataset
        data_min = np.amin(data_temp)
        edge_list = list(data_temp[0,:])+list(data_temp[-1,:])+list(data_temp[:,0])+list(data_temp[:,-1])
        edge_min = np.amin(edge_list)
        data_step = THRESHOLDS[1]-THRESHOLDS[0]
        data = np.zeros((np.shape(data_temp)[2]+2, np.shape(data_temp)[3]+2))
        data[ data == 0.0 ] =  edge_min - data_step
        data[1:-1,1:-1] = data_temp[i,0,:,:]

        entry_name = f"entry{i:03d}"
        data_dict[entry_name] = {'values': data, 'lat': lat, 'lon': lon, 'metadata':{'varname' : var, 'level_type': level_type, 'grid_id': grid_id, 'valid_time': time_str[i], 'units' : units, 'grid_spacing' : dx, 'grid_units': dx_units, 'sigma' : float(rec_sigma)}}

    return data_dict

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

def get_contour_feature_data(var, lat, lon, contours, thresholds, metadata, hex_palette, contour_method):
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

    # Create feature collection

    feature_collection = {
        "type": "FeatureCollection",
        "levels": {i: level for i, level in enumerate(contours)},
        "hex_palette":hex_palette
    }

    for key, value in metadata.items():
        feature_collection[key] = value

    feature_collection["features"] = []

    # Get contour and cell data using specified contours and thresholds
    level_contours, level_cells = get_level_data(var, contours, thresholds, lat, lon, contour_method)

    for i, lvl in enumerate(contours):
        for j, ctr in enumerate(level_contours[lvl]):
            coords_list = []
            is_out_of_bounds = False

            thiscell = level_cells[lvl][j]
            if not thiscell["hole"]:
                coords, out_of_bounds = get_contour_coords(ctr, lat, lon)
                coords_list.append(coords)
                del coords

                if thiscell["child_level"]:
                    for hole_cell, hole_level in zip(thiscell["child_cell"], thiscell["child_level"]):

                         hole_ctr = level_cells[hole_level][hole_cell]["ctr"]

                         coords, out_of_bounds = get_contour_coords(hole_ctr, lat, lon)
                         coords_list.append(coords)
                         del coords

            if coords_list:
                feature = create_feature(i, coords_list)
                feature_collection["features"].append(feature)

                del feature
                del coords_list

    if "missing_data" in metadata:
        feature_collection["features"].append(metadata["missing_data"])

    return feature_collection

def get_level_data(var, levels, thresholds, lat, lon, contour_method):
    """
    Get contour and cell data using specified levels and thresholds

    Parameters
    ----------
    var : array
        data
    levels : array
        level names
    thresholds : array
        Minimum values for each level

    Returns
   -------
    list
        List comprising level_contours (dict of contours for each
        level) and level_cells (dict of cells for each level)

    """

    from skimage import measure
    import matplotlib.pyplot as plt
    
    level_contours = {}
    level_cells = {}
    interior_cells = {}
    exterior_cells = {}

    # Loop through levels to create all contours
    for i in np.arange(0, len(thresholds)-1, 1):
        lvl = levels[i]
        thresh_low=thresholds[i]
        thresh_high=thresholds[i+1]


        if contour_method == "standard":

#            plt.imshow(var, cmap='viridis', interpolation='none')
#
#            # Adding a colorbar for reference
#            plt.colorbar()
#
#            # Display the plot
#            plt.show()

            mask = np.zeros_like(var)
            mask = np.logical_and(var>float(thresh_low), var<float(thresh_high)).astype(int)

#            plt.imshow(mask, cmap='viridis', interpolation='none')
#
#            # Adding a colorbar for reference
#            plt.colorbar()

            level_contours[lvl] = measure.find_contours(mask, level=0.5)

#            for contour in level_contours[lvl]:
#                plt.plot(contour[:, 1], contour[:, 0], color='red')
#
#            plt.show()


        elif contour_method == "pixel":
            
            print("Level: ", lvl)
            level_contours[lvl], mask = pixel_outline_contour2(var, thresh_low, thresh_high)

        # catagorize cells as interior or exterior (containing higher or lower values)
        interior_cells[lvl] = []
        exterior_cells[lvl] = []

        # Initialise cells with cell id, max/min x and y indices and other
        # (as yet) undefined features.
        level_cells[lvl] = []

        for j, ctr in enumerate(level_contours[lvl]):
            minxindex, minyindex = ctr.min(axis = 0)
            maxxindex, maxyindex = ctr.max(axis = 0)
            thiscell = {
                "cell_id": j,
                "min_xindex": minxindex,
                "max_xindex": maxxindex,
                "min_yindex": minyindex,
                "max_yindex": maxyindex,
                "ctr": ctr,
                "parent_cell": [],
                "parent_level": [],
                "child_cell": [],
                "child_level": [],
                "hole" : False
            }
            # Identify whether cell encloses valid or invalid parts of the mask
            if contour_method == "standard":
                orient_val = get_orientation(ctr)
                is_interior_contour = orient_val is not None and orient_val > 0 

                if is_interior_contour:
                    interior_cells[lvl].append(j)
                    thiscell["hole"] = True
                else:
                    exterior_cells[lvl].append(j)

            elif contour_method == "pixel":
                winding_dir, ext_int = check_exterior_interior(ctr, mask.T)

                if ext_int == "exterior":
                    exterior_cells[lvl].append(j)
                else:
                    interior_cells[lvl].append(j)
                    thiscell["hole"] = True

            # Append "thiscell" to the "level_cells" dictionary
            level_cells[lvl].append(thiscell.copy())

    #Make holes in the exterior contours of a particular level based on interior contours on that same level
    for i in np.arange(0, len(thresholds)-1, 1):
        lvl = levels[i]
        for j in exterior_cells[lvl]:
            for k in interior_cells[lvl]:
                is_enclosed = is_enclosed_by(level_cells[lvl][k], level_cells[lvl][j])
                if is_enclosed:
                    level_cells[lvl][j]["child_cell"].append(level_cells[lvl][k]["cell_id"])
                    level_cells[lvl][j]["child_level"].append(lvl)
                    level_cells[lvl][k]["parent_cell"].append(level_cells[lvl][j]["cell_id"])
                    level_cells[lvl][k]["parent_level"].append(lvl)

    return level_contours, level_cells

def get_contour_coords(ctr, lat, lon):
    """
    Get coordinates for contour specified as a list of x, y indices

    Parameters
    ----------
    ctr : list
        List specifying contour as x, y indices
    lat : array
        Array of latitude values
    lon : array
        Array of longitude values

    Returns
    -------
    list
        Coordinates array and boolean indicating whether any
        coordinates were out of bounds

    """

    coords = []
    out_of_bounds = False

    # Loop through points in contour
    for xy_index in ctr:
        # Convert x, y index to lat, lon
        coord_lat, coord_lon = get_coords(xy_index, lat, lon)

        # Skip points that are out of bounds (corresponding to
        # masked values in the lat/lon lookups)
        if (coord_lat is np.ma.masked or coord_lon is np.ma.masked):
            out_of_bounds = True
            continue

        # Record coordinates
        coords.append([format_float(coord_lon), format_float(coord_lat)])

    if len(coords) > 0 and coords[0] != coords[-1]:
        # Append starting point to avoid a non-closed contour
        coords.append(coords[0])

    return coords, out_of_bounds

def get_orientation(ctr):
    """
    Get orientation of a contour

    Calculates the determinant of a sequence of points at a vertex of
    the convex hull of the polygon defined by the points in the
    contour.  If the determinant is negative, then the polygon is
    oriented clockwise. If the determinant is positive, the polygon is
    oriented counterclockwise.

    https://en.wikipedia.org/wiki/Curve_orientation#Orientation_of_a_simple_polygon

    Parameters
    ----------
    ctr : array
        (n, 2) array of vertices

    Returns
    -------
    scalar
        Value of determinant

    """

#    # 1. Select point on convex hull by finding vertex with smallest Y
#    # index (and maximum X index in case of ties)
#    y = ctr[:, 1]
#    y_min_index = np.argwhere(y == y.min())
#    if len(y_min_index) > 1:
#        x = ctr[y_min_index, 0]
#        print(x)
#        y_min_index = y_min_index[x == x.max()]
#        print(y_min_index)
#    y_min_index = y_min_index.item()

    # 1. Select point on convex hull by finding vertex with smallest Y index
    # (and maximum X index in case of ties)
    y = ctr[:, 1]
    y_min_index = np.argwhere(y == y.min())  # Get all indices where Y is the minimum

    if len(y_min_index) > 1:
        # Multiple points share the same minimum Y value, so resolve using the maximum X
        x = ctr[y_min_index.flatten(), 0]  # Get corresponding X values at the y_min_indices
        y_min_index = y_min_index[x == x.max()]  # Select the index with the maximum X

    # Extract the index as an integer, ensuring it's a single value
    y_min_index = y_min_index.flatten()[0]

    # 2. Label previous and next vertices
    i = y_min_index - 1
    j = y_min_index
    k = y_min_index + 1 if y_min_index < len(ctr) - 1 else 0

    # 3. Calculate determinant of the points
    A = np.hstack((np.ones((3,1)), ctr[(i, j, k), :]))
    det = np.linalg.det(A)

    return det

def get_coords(index, lat, lon):
    """
    Convert indices to latitude/longitude values.

    Parameters
    ----------
    index : array
        Array specifying (possibly non-integer) indices of
        desired coordinates.
    lat : array
        Array of latitude values
    lon : array
        Array of longitude values

    Returns
    -------
    tuple
        Latitude and longitude corresponding to the specified
        indices.

    """

   # Get floor/ceiling values for indices
    lower = tuple(np.floor(index).astype(np.int32))
    upper = tuple(np.ceil(index).astype(np.int32))

    # Get lat and lon values for upper and lower indices

    lat_1 = lat[lower]
    lat_2 = lat[upper]
    lon_1 = lon[lower]
    lon_2 = lon[upper]

    if not (-90 <= lat_1 <= 90):
        if not (-90 <= lat_2 <= 90):
            print("We have a major problem with the latitude value supplied")
        else:
            lat_1 = lat_2

    if not (-90 <= lat_2 <= 90):
        if not (-90 <= lat_1 <= 90):
            print("We have a major problem with the latitude value supplied")
        else:
            lat_2 = lat_1

    if not (-180 <= lon_1 <= 360):
        if not (-180 <= lon_2 <= 360):
            print("We have a major problem with the longitude value supplied")
        else:
            lon_1 = lon_2

    if not (-180 <= lon_2 <= 360):
        if not (-180 <= lon_1 <= 360):
            print("We have a major problem with the longitude value supplied")
        else:
            lon_2 = lon_1

    # Return average of lat / lon values
    return ((lat_1 + lat_2)/2.0, (lon_1 + lon_2)/2.0)

def is_enclosed_by(a, b):

    """
    Check whether a cell `a` is enclosed by a candidate cell `b`

    Arguments are dicts that define the minimum and maximum x and y
    values and an array of xy coordinates.

    Parameters
    ----------
    a : dict
        Dict of values defining the candidate enclosed cell
    b : dict
        Dict of values defining the candidate enclosing cell

    Returns
    -------
    is_enclosed: boolean
        Boolean value indicating whether cell `a` is enclosed
        by cell 'b'.

    """
    from matplotlib.path import Path

    tolerance = 0.01

    enclosing_path = Path(b["ctr"])

    encl_points = enclosing_path.contains_points(a["ctr"], radius=tolerance)
    count = np.count_nonzero(encl_points)
    if count == (len(a["ctr"])):
        is_enclosed = True
    else:
        is_enclosed = False

    return is_enclosed

def create_feature(lvl, coords_list):
    """
    Given a level and a list of coordinates a geojson feature is created

    Parameters
    ----------
    lvl : int or string
        Identifier linking the feature to the correct level in the level feature collection.
    coords_list : list
        list of coordinates

    Returns
    -------
    dict
        Dictionary in form of a geojson feature.
    """

    if lvl == 'missing_data':
        feature_properties = {
            "ObjectType": "missing-data-contour"
        }
    else:
        feature_properties = {
            "ObjectType": "data-contour",
            "level": lvl,
        }

    feature = {
        "type": "Feature",
        "properties": feature_properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": coords_list,
        }
    }
    return feature


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

def main():

    import matplotlib.pyplot as plt
    import numpy as np

    # Check input arguments
    input_file, output_dir, VAR_NAME, SOURCE, smooth_dict, contour_dict, level_dict = input_args()

    # Read data file

    data = read_datafile(SOURCE, input_file, VAR_NAME, contour_dict, level_dict)

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

    data_range = max_data - min_data
   
    if data_range != 0.0:
        if data_range <= 2.0:
            exponent = np.floor(np.log10(np.abs(data_range)))
        
            min_data = round(min_data, -1*int(exponent))
            max_data = round(max_data, -1*int(exponent))

    # Define LEVELS and THRESHOLDS
        if data_range >= 2.0:
            LEVELS, THRESHOLDS = generate_contours(contour_dict, max_int_data, min_int_data)
        else:
            LEVELS, THRESHOLDS = generate_contours(contour_dict, max_data, min_data)

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
                data[entry]['values'] = filter_var_data(data[entry]['values'], data[entry]['metadata']['sigma'])

        # create colour palette
            if "colormap" in contour_dict:
                hex_palette = create_color_palette(LEVELS, contour_dict["colormap"])
            else:
                hex_palette = ["#"+hexcode for hexcode in contour_dict["color_pal"] ]

        # Get features
            feature_collection = get_contour_feature_data(data[entry]['values'], data[entry]['lat'], data[entry]['lon'], LEVELS, THRESHOLDS, data[entry]['metadata'], hex_palette, contour_dict["contour_method"])

            if 'site_lat' in data[entry]['metadata'] and 'site_lon' in data[entry]['metadata']:
                feature_collection = add_points(feature_collection, [data[entry]['metadata']['site_lat']], [data[entry]['metadata']['site_lon']], {"name": "site location"})

        # Save output
            write_geojson(feature_collection, output_dir, input_file, entry, VAR_NAME, data[entry]['metadata'])

    else:
        print("There is no data to create a geojson for.")

if __name__ == "__main__":
    main()
