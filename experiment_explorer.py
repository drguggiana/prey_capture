# imports
import functions_bondjango as bd
import processing_parameters
import paths
import os
import numpy as np
import pandas as pd
import cv2
from skimage import io
import functions_preprocessing as fp
import matplotlib.pyplot as plt


# define the search string
search_string = processing_parameters.search_string

# define the target model
if 'miniscope' in search_string:
    target_model = 'video_experiment'
else:
    target_model = 'vr_experiment'

# get the queryset
files = bd.query_database(target_model, search_string)[0]
raw_path = files['avi_path']
tif_path = files['tif_path']
sync_path = files['sync_path']

if 'track_path' in files.keys():
    motive_path = files['track_path']
else:
    motive_path = []


calcium_path = files['avi_path'][:-4] + '_calcium.hdf5'
match_path = os.path.join(paths.analysis_path, '_'.join((files['mouse'], files['rig'], 'cellMatch.hdf5')))
# assemble the save paths
# save_path = os.path.join(paths.analysis_path,
#                          os.path.basename(files['avi_path'][:-4])) + '_rawcoord.hdf5'
# pic_path = save_path[:-14] + '.png'


# generate demo dlc and calcium data based on the given raw data files

# count the video frames
cap = cv2.VideoCapture(raw_path)
video_frames = []
video_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
    video_count += 1


# count the calcium frames
calcium_stack = io.imread(tif_path)
calcium_count = calcium_stack.shape[0]

print(f'Video frames in the avi file: {video_count}')
print(f'Tif frames in the tif file: {calcium_count}')

# if there is a motive path
if len(motive_path) > 0:
    # count the motive frames
    try:
        motive_data = pd.read_csv(motive_path, header=None)
    except pd.errors.ParserError:
        # This occurs for files that have more complicated headers
        arena_corners, obstacle_positions, df_line = fp.read_motive_header(motive_path)
        motive_data = pd.read_csv(motive_path, header=0, skiprows=df_line)

    # trim the motive data to the first trial
    try:
        motive_data = motive_data.iloc[np.argwhere(motive_data.loc[:, [' trial_num']].to_numpy() == 0)[0][0]:, :]
    except KeyError:
        motive_data = motive_data.iloc[np.argwhere(motive_data.loc[:, ['trial_num']].to_numpy() == 0)[0][0]:, :]
    motive_count = motive_data.shape[0]
    print(f'Motive frames in the motive file: {motive_count}')

# TODO: make this compatible with all versions
if len(motive_path) > 0:
    # load the sync file
    sync_data = pd.read_csv(sync_path, names=['Time', 'projector_frames', 'camera_frames',
                                              'sync_trigger', 'mini_frames', 'wheel_frames'], index_col=False)
    # get the start and end triggers
    sync_start = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 1)[0][0]
    sync_end = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 2)[0][0]

    # trim the sync data to the experiment
    sync_data = sync_data.iloc[sync_start:sync_end, :].reset_index(drop=True)
else:
    sync_data = pd.read_csv(sync_path, names=['Time', 'mini_frames', 'camera_frames'], index_col=False)

# count the respective frames

sync_camera = np.argwhere(np.diff(np.round(sync_data.loc[:, 'camera_frames'])) > 0).flatten() + 1

sync_calcium = np.argwhere(np.diff(np.round(sync_data.loc[:, 'mini_frames'])) > 0).flatten() + 1

# print the sync frames
print(f'Video frames in the sync file: {sync_camera.shape[0]}')
print(f'Tif frames in the sync file: {sync_calcium.shape[0]}')

if len(motive_path) > 0:
    sync_motive = np.argwhere(np.abs(np.diff(np.round(sync_data.loc[:, 'projector_frames']))) > 0).flatten() + 1

    print(f'Motive frames in the sync file: {sync_motive.shape[0]}')

# TODO: expand to preprocessed files
# plot sync frames
fig = plt.figure()

# plot each field
for idx, el in enumerate(sync_data.columns[1:]):
    ax = fig.add_subplot(len(sync_data.columns[1:]), 1, idx+1)
    ax.plot(sync_data['Time'].to_numpy(), sync_data[el].to_numpy())
    plt.ylabel(el)

# show target frame
# define the target frame
target_frame = 700
# define the number of frames to show
number_frames = 12

fig3 = plt.figure()
for frame in np.arange(number_frames):
    ax = fig3.add_subplot(3, 4, frame+1)
    ax.imshow(video_frames[target_frame+frame])

# fig2 = plt.figure()
# ax = fig2.add_subplot(111)
# # ax.plot(np.diff(sync_motive))
# # freq, edges = np.histogram(np.diff(sync_motive))
# ax.hist(np.diff(sync_calcium), 50)
# plt.show()



