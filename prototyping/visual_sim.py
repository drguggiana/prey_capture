import numpy as np
import pandas as pd
import functions_bondjango as bd
import paths
import processing_parameters
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import cv2


# define auxiliary src
def deg2width(deg, dimension):
    """Convert the input degrees into pixels based on the given dimension for 360 degrees"""
    return dimension*deg/360


def deg2position(deg, dimension):
    """Convert the input degrees [-180:180] into an angular position in pixels"""
    return deg2width(deg + 180, dimension)


# load the real data to provide to the model

# get the search query
search_string = processing_parameters.search_string

# get the paths from the database
all_path = bd.query_database('analyzed_data', search_string)
input_path = [el['analysis_path'] for el in all_path if '_preproc' in el['slug']]
# get the day, animal and rig
day = '_'.join(all_path[0]['slug'].split('_')[0:3])
rig = all_path[0]['rig']
animal = all_path[0]['slug'].split('_')[3:6]
animal = '_'.join([animal[0].upper()] + animal[1:])

# allocate memory for the data
raw_data = []
# allocate memory for excluded trials
excluded_trials = []
# for all the files
for files in input_path[:10]:
    # load the data
    with pd.HDFStore(files, mode='r') as h:
        if ('/matched_calcium' in h.keys()) & ('/latents' in h.keys()):

            # concatenate the latents
            dataframe = pd.concat([h['matched_calcium'], h['latents']], axis=1)
            # store
            raw_data.append((files, dataframe))
        else:
            excluded_trials.append(files)
print(f'Number of files loaded: {len(raw_data)}')
# get the angular size and position of the cricket
prey_visual_angle = raw_data[0][1]['cricket_0_visual_angle'].to_numpy()
prey_distance = raw_data[0][1]['cricket_0_mouse_distance'].to_numpy()
prey_angle = raw_data[0][1]['cricket_0_delta_heading'].to_numpy()

# create a temporal sequence of the visual scene based on the cricket position
# define the scene parameters (width and height will have to be scaled to -180 to 180 degrees)
width = 1000
height = 1000
number_frames = prey_angle.shape[0]
prey_aspect_ratio = 0.5
ground_level = 500
# will need to render a small version of a 2D scene, of arbitrary resolution
# allocate the sequence
# frames = np.zeros((number_frames, width, height))

# run through the frames
for idx in np.arange(number_frames):
    # get the ellipse dimensions
    ellipse_width = deg2width(prey_visual_angle[idx], width)
    ellipse_height = ellipse_width*prey_aspect_ratio

    # get the ellipse position
    ellipse_x = deg2position(prey_angle[idx], width)
    # draw the ellipse


    print('yay2')

# use geometry to translate the position of the cricket relative to the mouse into a matrix
# render the scene as a 3D matrix, with the third dimension being time

# define the filters to use
# create/load 2D filters based on real RFs
# can start with a very simple, manually programmed grid and go from there
filter_templates = [[[0, 0, 0], [1, 1, 1], [0, 0, 0]]]
filter_positions = [0, 0]

# multiply the filter by the visual scene (not convolution, since the filter only sees a part of the scene)
# do that for every filter
# compare the variety of responses obtained to the real responses

print('yay')
