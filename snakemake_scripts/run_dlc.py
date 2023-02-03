# imports
import os
import sys
import shutil
import json

os.environ["DLClight"] = "True"
import deeplabcut as dlc

# print(os.getcwd())
# sys.path.insert(0, r'C:\Users\mmccann\repos\bonhoeffer\prey_capture')
import paths
sys.path.insert(0, os.path.abspath(paths.prey_capture_repo_directory))

import functions_bondjango as bd
import functions_io as fi
import functions_misc as fm
import processing_parameters

# # define the config_path
# config_path = paths.config_path

try:
    # get the target video path
    video_path = sys.argv[1]
    out_path = sys.argv[2]
    video_data = json.loads(sys.argv[3])
except IndexError:
    # define the search string
    search_string = processing_parameters.search_string
    if 'miniscope' in search_string:
        target_model = 'video_experiment'
    else:
        target_model = 'vr_experiment'
    # query the database for data to plot
    data_all = bd.query_database(target_model, search_string)
    video_data = data_all[0]
    video_path = video_data['avi_path']
    # assemble the output path
    out_path = video_path.replace('.avi', '_dlc.h5')

# define the new video path
temp_video_path = os.path.join(paths.temp_path, os.path.basename(video_path))
# copy the video to the working folder
shutil.copyfile(video_path, temp_video_path)

# select which network to use
if video_data['rig'] == 'miniscope':
    target_model = 'video_experiment'
    config = paths.config_vame_path
else:
    target_model = 'vr_experiment'
    if video_data['rig'] in ['VWheel', 'VWheelWF']:
        config = paths.config_path_VWheel
    elif video_data['rig'] == 'VTuningWF':
        config = paths.config_path_VTuningWF
    else:
        config = paths.config_path_VTuning

# Load the config and get necessary parameters (namely shuffle)
cfg = dlc.auxiliaryfunctions.read_config(config)
shuffle = cfg.get('shuffle', 1)    # Defaults to 1 if not present in config file

# analyze the video
dlc.analyze_videos(config, [temp_video_path], shuffle=shuffle, destfolder=paths.temp_path)

# filter the data
dlc.filterpredictions(config, [temp_video_path], shuffle=shuffle, destfolder=paths.temp_path, save_as_csv=False)

# move and rename the file

# get a list of the files present in the temp folder
origin_file = [el for el in os.listdir(paths.temp_path)
               if ('.h5' in el) and (video_data['slug'] in fm.slugify(el))]

assert len(origin_file) > 0, 'The target file was not found'

# rename and move the tracking file to the final path
shutil.move(os.path.join(paths.temp_path, origin_file[0]), out_path)
# delete the folder contents
fi.delete_contents(paths.temp_path)

# update the bondjango entry (need to sort out some fields)
ori_data = video_data.copy()
ori_data['dlc_path'] = out_path
mouse = ori_data['mouse']
ori_data['mouse'] = '/'.join((paths.bondjango_url, 'mouse', mouse, ''))
ori_data['experiment_type'] = '/'.join((paths.bondjango_url, 'experiment_type', 'Free_behavior', ''))

update_url = '/'.join((paths.bondjango_url, target_model, ori_data['slug'], ''))
output_entry = bd.update_entry(update_url, ori_data)

print(output_entry.status_code)
