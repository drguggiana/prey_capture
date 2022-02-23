# imports
import numpy as np
import pandas as pd
import functions_bondjango as bd
import paths
import processing_parameters
import cv2
import h5py
import os
from PIL import Image


def simple_random(columns, number_rows):
    """Generate a random dataframe with the desired columns and rows"""
    # get the number of columns
    number_columns = len(columns)
    # generate the random matrix
    random_matrix = np.random.rand(number_rows, number_columns)
    # create the dataframe
    data_out = pd.DataFrame(random_matrix, columns=columns)
    return data_out


# get the search_string
search_string = processing_parameters.search_string
# define the target model
if 'miniscope' in search_string:
    target_model = 'video_experiment'
# elif 'vtuning' in search_string:
#     target_model = 'vtuning'
else:
    target_model = 'vr_experiment'

# Load the target file's info
files = bd.query_database(target_model, search_string)[0]
# TODO: make the whole script rig-agnostic

# Calcium #
# get the tif file
tif_path = files['tif_path']

# count the calcium frames
with Image.open(tif_path) as img:
    calcium_count = img.n_frames

# get the template path
template_calcium_path = os.path.join(paths.videoexperiment_path,
                                     processing_parameters.template_paths['vtuning']['calcium_path'])
# load the template file
with h5py.File(template_calcium_path, mode='r') as f:
    calcium_data = np.array(f['calcium_data']).T
# the column names are not needed for the calcium, but I'd rather keep the randomizing function structure constant
template_calcium_columns = ['cell_'+str(el) for el in np.arange(calcium_data.shape[1])]
# generate the data according to a given distribution (also consider the pose repair network, might come in handy)
generated_calcium = simple_random(template_calcium_columns, calcium_count).to_numpy()
# assemble the output path
calcium_out_path = os.path.join(paths.vrexperiment_path, files['slug']+'_calcium.hdf5')
# save the data as an h5py
with h5py.File(calcium_out_path, 'w') as file:
    file.create_dataset('calcium_data', data=generated_calcium.T)
# update the bondjango entry (need to sort out some fields)
ori_data = files.copy()
ori_data['fluo_path'] = calcium_out_path
mouse = ori_data['mouse']
ori_data['mouse'] = '/'.join((paths.bondjango_url, 'mouse', mouse, ''))
ori_data['experiment_type'] = '/'.join((paths.bondjango_url, 'experiment_type', 'Free_behavior', ''))

# fix the preprocessing file field too
for idx, el in enumerate(ori_data['preproc_files']):
    ori_data['preproc_files'][idx] = \
        '/'.join((paths.bondjango_url, 'analyzed_data', ori_data['preproc_files'][idx], ''))

update_url = '/'.join((paths.bondjango_url, target_model, ori_data['slug'], ''))
output_entry = bd.update_entry(update_url, ori_data)
print(output_entry.status_code)

# DLC #
# get the avi file
avi_path = files['avi_path']
# count the video frames
cap = cv2.VideoCapture(avi_path)
video_frames = []
video_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
    video_count += 1

# get the template path
template_dlc_path = os.path.join(paths.videoexperiment_path,
                                 processing_parameters.template_paths['vtuning']['dlc_path'])

# load the dlc file
# load the bonsai info
raw_h5 = pd.read_hdf(template_dlc_path)
# get the column names
template_dlc_columns = raw_h5.columns
# generate random data
generated_dlc = simple_random(template_dlc_columns, video_count)
# TODO: make this more obviously tunable, as right now we might easily forget
# increase the likelihood threshold so all points pass
likelihood_columns = [el for el in generated_dlc if 'likelihood' in el]
generated_dlc[likelihood_columns] = 0.91
# assemble the output path
dlc_out_path = os.path.join(paths.vrexperiment_path, files['slug']+'_dlc.h5')
# save the file
generated_dlc.to_hdf(dlc_out_path, 'raw_data')
# update the bondjango entry (need to sort out some fields)
ori_data = files.copy()
ori_data['dlc_path'] = dlc_out_path
mouse = ori_data['mouse']
ori_data['mouse'] = '/'.join((paths.bondjango_url, 'mouse', mouse, ''))
ori_data['experiment_type'] = '/'.join((paths.bondjango_url, 'experiment_type', 'Free_behavior', ''))

# fix the preprocessing file field too
for idx, el in enumerate(ori_data['preproc_files']):
    ori_data['preproc_files'][idx] = \
        '/'.join((paths.bondjango_url, 'analyzed_data', ori_data['preproc_files'][idx], ''))

update_url = '/'.join((paths.bondjango_url, target_model, ori_data['slug'], ''))
output_entry = bd.update_entry(update_url, ori_data)
print(output_entry.status_code)

# TODO:  tag as artificial ? (won't be an issue for the virtual mice, but maybe we'll need simulated data for modeling
# from real mice)
