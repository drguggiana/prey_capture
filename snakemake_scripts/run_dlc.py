# imports
import os
import sys
sys.path.insert(0, os.path.abspath(r'D:\Code Repos\prey_capture'))
os.environ["DLClight"] = "True"

import os
import sys
import shutil
import deeplabcut as dlc
import paths
import functions_bondjango as bd
import functions_io as fi
import functions_misc as fm
import json
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

# analyze the video
# dlc.analyze_videos?

# select which network to use
if video_data['rig'] == 'miniscope':
    dlc.analyze_videos(paths.config_vame_path, [temp_video_path], destfolder=paths.temp_path)
    target_model = 'video_experiment'
else:
    # dlc.analyze_videos(paths.config_path, [temp_video_path], destfolder=paths.temp_path)
    # uncomment when the vr network is trained
    # dlc.analyze_videos(paths.config_path_vr, [temp_video_path], destfolder=paths.temp_path)
    # TODO: replace by correct DLC network, will probs have to add a case for vwheel too
    dlc.analyze_videos(paths.config_vame_path, [temp_video_path], destfolder=paths.temp_path)
    target_model = 'vr_experiment'

# filter the data
# dlc.filterpredictions(config_path, [temp_video_path], filtertype='median',
#                       windowlength=11, destfolder=paths.temp_path, save_as_csv=False)
# dlc.filterpredictions?

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
