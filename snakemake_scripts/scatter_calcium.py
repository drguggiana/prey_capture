import sys
import json
import os
import h5py
import functions_bondjango as bd
import paths
import pandas as pd
import numpy as np

try:
    # get the target video path
    video_path = sys.argv[1]
    calcium_path = video_path['analysis_path']
    out_path = sys.argv[2]
    data_all = json.loads(sys.argv[3])

    name_parts = out_path.split('_')
    day = name_parts[0]
    # day = '_'.join((day[:2], day[2:4], day[4:]))
    animal = name_parts[1]
    rig = name_parts[2]
except IndexError:
    # define the target animal and date
    animal = 'DG_200701_a'
    day = '09_08_2020'
    rig = 'miniscope'
    # define the search string
    search_string = 'slug:%s' % '_'.join((day, animal, rig, 'calciumday'))
    # search_string = 'slug:08_06_2020_18_07_32_miniscope_DG_200701_a_succ'
    # query the database for data to plot
    data_all = bd.query_database('analyzed_data', search_string)
    video_data = data_all[0]
    calcium_path = video_data['analysis_path']
    # video_path = video_data['tif_path']
    # video_path = [el['tif_path'] for el in data_all]
    # assemble the output path
    # out_path = video_path[0].replace('.tif', '_calcium.hdf5')
    # out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'calciumday.hdf5')))

# if there are no ROIs detected, skip the file and print the name
try:
    # load the contents of the ca file
    with h5py.File(calcium_path) as f:
        # calcium_data = np.array((f['sigfn'])).T
        calcium_data = np.array(f['estimates/C'])
        frame_list = np.array(f['frame_list'])

    # grab the processed files and split them based on the log file

    # # read the log file
    # files_list = pd.read_csv(out_path_log)

    files_list = [str(el) for el in frame_list[:, 0]]
    # initialize a counter for the frames
    frame_counter = 0

    # for all the rows in the dataframe
    for index, row in files_list.iterrows():
        # get the frames from this file
        current_calcium = calcium_data[:, frame_counter:row['frame_number'] + frame_counter]

        # assemble the save path for the file
        new_calcium_path = os.path.join(out_path[index],
                                        os.path.basename(
                                            row['filename'].replace('.tif', '_calcium_data.h5')))

        # save the data as an h5py
        with h5py.File(new_calcium_path) as file:
            file.create_dataset('calcium_data', data=current_calcium)

        # update the frame counter
        frame_counter += row['frame_number']
except KeyError:
    print('This file did not contain any ROIs: ' + calcium_path)

# update the bondjango entry (need to sort out some fields)
# ori_data = deepcopy(video_data)
ori_data['fluo_path'] = out_path
mouse = ori_data['mouse']
ori_data['mouse'] = '/'.join((paths.bondjango_url, 'mouse', mouse, ''))
ori_data['experiment_type'] = '/'.join((paths.bondjango_url, 'experiment_type', 'Free_behavior', ''))
# fill in the preprocess_file field if present
if len(ori_data['preproc_files']) > 0:
    # for all the elements there
    for idx, el in enumerate(ori_data['preproc_files']):
        ori_data['preproc_files'][idx] = \
            '/'.join((paths.bondjango_url, 'analyzed_data', el, ''))

update_url = '/'.join((paths.bondjango_url, target_model, ori_data['slug'], ''))
output_entry = bd.update_entry(update_url, ori_data)