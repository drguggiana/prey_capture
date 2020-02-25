from tkinter import filedialog
import functions_misc
import functions_matching
import functions_plotting
import functions_io
import paths

import tracking_preprocessBonsai
import matplotlib.pyplot as plt
import kinematic_S1_calculations
import datetime
import os
import functions_bondjango as bd

# get rid of the tk main window
# functions_misc.tk_killwindow()

# define the search string
search_string = 'result=succ, lighting=normal'
target_model = 'video_experiment'
# get the queryset
file_path_bonsai = bd.query_database(target_model)
# file_path_bonsai = bd.query_database(target_model, search_string)

# run through the files and select the appropriate analysis script
for files in file_path_bonsai:
    # get the file date
    # file_date = functions_io.get_file_date(files)
    file_date = datetime.datetime.strptime(files['date'], '%Y-%m-%dT%H:%M:%SZ')

    # check for the nomini flag
    if ('nomini' in files['notes']) or ('nofluo' in files['notes']):
        nomini_flag = True
    else:
        nomini_flag = False

    # assemble the save path
    save_path = os.path.join(paths.analysis_path, os.path.basename(files['bonsai_path']))
    # decide the analysis path based on the file name and date
    # if miniscope with the _nomini flag, run bonsai only
    if target_model == 'video_experiment' and nomini_flag:
        # run the first stage of preprocessing
        out_path, filtered_traces, pic_path = tracking_preprocessBonsai.run_preprocess([files['bonsai_path']],
                                                                                       save_path,
                                                                                       ['cricket_x', 'cricket_y'])
        # TODO: add corner detection to calibrate the coordinate to real size
        # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, filtered_traces)

    # if miniscope regular, run with the matching of miniscope frames
    elif target_model == 'video_experiment' and not nomini_flag:
        # run the first stage of preprocessing
        out_path, filtered_traces, pic_path = tracking_preprocessBonsai.run_preprocess([files['bonsai_path']],
                                                                                       save_path,
                                                                                       ['cricket_x', 'cricket_y'])

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, filtered_traces)

        # get the calcium file path
        calcium_path = files['fluo_path']

        # find the sync file
        sync_path = files['sync_path']
        # get a dataframe with the calcium data matched to the bonsai data
        matched_calcium = functions_matching.match_calcium(calcium_path, sync_path, kinematics_data)

        matched_calcium.to_hdf(out_path[0], key='matched_calcium', mode='a', format='table')

    elif target_model != 'video_experiment' and file_date <= datetime.datetime(year=2019, month=11, day=10):
        # run the first stage of preprocessing
        out_path, filtered_traces, pic_path = tracking_preprocessBonsai.run_preprocess([files['bonsai_path']],
                                                                                       save_path,
                                                                                       ['cricket_x', 'cricket_y'])
        # TODO: add corner detection to calibrate the coordinate to real size
        # in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

        # TODO: add the old motive-bonsai alignment as a function

        # run the preprocessing kinematic calculations
        kinematics_data = kinematic_S1_calculations.kinematic_calculations(out_path, paths.kinematics_path)
    # TODO: if no miniscope and after sync, run the new analysis
    else:
        pic_path = ''
        out_path = []

    # assemble the entry data
    entry_data = {
        'analysis_type': 'preprocessing',
        'analysis_path': out_path[0],
        'pic_path': pic_path,
        'result': files['result'],
        'rig': files['rig'],
        'lighting': files['lighting'],
        'slug': files['slug'] + '_preprocessing',
        'video_analysis': [files['url']]
    }
    # check if the entry already exists, if so, update it, otherwise, create it
    if len(files['preproc_files']) > 0:
        update_url = '/'.join((paths.bondjango_url, 'analyzed_data', files['preproc_files'][0], ''))
        output_entry = bd.update_entry(update_url, entry_data)
    else:
        # build the url for creating an entry
        create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
        output_entry = bd.create_entry(create_url, entry_data)
    # print the result
    print('The output status was %i, reason %s' % (output_entry.status_code, output_entry.reason))

print('yay')
print('<3')
