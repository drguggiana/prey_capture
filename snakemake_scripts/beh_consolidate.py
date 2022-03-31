import processing_parameters
import functions_bondjango as bd
import os
import paths
import pandas as pd
import functions_misc as fm
import yaml
import datetime
import numpy as np
from scipy.ndimage.measurements import label
import scipy.stats as stat


try:
    # get the input
    input_path = snakemake.input
    # get the file info
    file_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.info]

    # read the output path and the input file urls
    out_path = snakemake.output[0]
except NameError:
    # get the search string
    search_string = processing_parameters.search_string

    # get the paths from the database
    file_info = bd.query_database('analyzed_data', search_string)
    file_info = [el for el in file_info if '_preprocessing' in el['slug']]
    input_path = [el['analysis_path'] for el in file_info]

    # assemble the output path
    out_path = os.path.join(paths.analysis_path, 'test_behconsolidate.hdf5')

# get the rig
if 'miniscope' in input_path[0]:
    rig = 'miniscope'
else:
    rig = 'VR'

# allocate the output
output_df = []
# for all the input paths
for idx, file in enumerate(input_path):
    print(file)
    # open the file
    with pd.HDFStore(file) as h:
        # load the dataframe
        if '/matched_calcium' in h.keys():
            data = h['matched_calcium']
        elif '/full_traces' in h.keys():
            data = h['full_traces']
        else:
            continue
        # if it's a bad file, skip
        if data.iloc[0, 0] == 'badFile':
            continue

        # if sync frames is present, remove
        if 'sync_frames' in data.columns:
            data = data.drop(['sync_frames'], axis=1)
        # get the result and translate into a number
        result = processing_parameters.interpret_result[file_info[idx]['result']]

        # get the day
        day = datetime.datetime.strptime(data.loc[0, 'datetime'][:10], '%Y-%m-%d')

        # get the mouse
        mouse = data.loc[0, 'mouse']

        # only if it was a success trial
        if result == 1:
            # get the trial time
            # get the speed
            mouse_speed = data.loc[:, 'mouse_speed'].to_numpy()
            # find the last point where the threshold is passed
            duration_idx = np.argwhere(mouse_speed < processing_parameters.speed_threshold)[-1][0]
            # calculate the time until then
            duration = data.loc[duration_idx, 'time_vector'] - data.loc[0, 'time_vector']

            # get the hunt vector
            hunt_trace = data.loc[:, 'hunt_trace'].to_numpy()
            # get the latency to start the hunt
            try:
                latency_approach_idx = np.argwhere(hunt_trace == 1)[0][0]
            except IndexError:
                # assume the approach happened right at the start, as usually if the mouse is placed
                # too close to the cricket this will happen
                latency_approach_idx = 0
            approach_latency = data.loc[latency_approach_idx, 'time_vector'] - data.loc[0, 'time_vector']
            # get the number of approaches
            [_, approach_number] = label(hunt_trace == 1)

            # get the latency to the first contact
            try:
                latency_contact_idx = np.argwhere(hunt_trace == 2)[0][0]
            except IndexError:
                # assume the contact was not detected before the cricket was occluded
                latency_contact_idx = np.argmin(data.loc[:, 'cricket_0_mouse_distance'].to_numpy())
            contact_latency = data.loc[latency_contact_idx, 'time_vector'] - data.loc[0, 'time_vector']
            # get the number of encounters
            [_, contact_number] = label(hunt_trace == 2)
        else:
            duration = np.nan
            approach_latency = np.nan
            approach_number = np.nan
            contact_latency = np.nan
            contact_number = np.nan

        # define the columns to drop before averaging
        cell_columns = [el for el in data.columns if 'cell' in el]
        drop_columns = ['time_vector', 'mouse', 'datetime', 'cricket_0_quadrant']
        # drop for linear averaging
        angle_columns = ['mouse_heading', 'head_direction', 'cricket_0_heading',
                         'cricket_0_delta_heading', 'cricket_0_delta_head']

        # drop for angular averages
        linear_columns = [el for el in data.columns if (
                el not in angle_columns and el not in drop_columns and el not in cell_columns)]
        # get the averages and sem per column
        averages_linear = np.nanmean(data.loc[:, linear_columns], axis=0)
        sem_linear = stat.sem(data.loc[:, linear_columns], axis=0, nan_policy='omit')

        # if it's not a habi trial, add the cricket columns
        if result < 0:
            angle_columns_sub = ['mouse_heading', 'head_direction']
        else:
            angle_columns_sub = angle_columns
        averages_circular = np.rad2deg(stat.circmean(np.deg2rad(data.loc[:, angle_columns_sub]),
                                                     axis=0, nan_policy='omit'))
        sem_circular = np.rad2deg(stat.circstd(np.deg2rad(data.loc[:, angle_columns_sub]),
                                               axis=0, nan_policy='omit')/np.sqrt(data.shape[0]))
        # if it's a habi trial
        if result < 0:
            # TODO: make this less arbitrary, maybe just leave the columns in from preprocessin
            averages_circular = np.hstack((averages_circular, [np.nan, np.nan, np.nan]))
            sem_circular = np.hstack((sem_circular, [np.nan, np.nan, np.nan]))
            averages_linear = np.hstack((averages_linear, [np.nan]*10))
            sem_linear = np.hstack((sem_linear, [np.nan] * 10))

        # if there are cells
        if len(cell_columns) > 0:
            # average all the calcium activity
            calcium_average = np.nanmean(data.loc[:, cell_columns])
            calcium_sem = stat.sem(data.loc[:, cell_columns], nan_policy='omit', axis=None)
        else:
            calcium_average = np.nan
            calcium_sem = np.nan

        # if a success, save the column names for the final dataframe
        # TODO: this is hella ugly, fix
        if result == 1:
            final_linear = linear_columns.copy()
            final_circular = angle_columns.copy()
        # assemble the output tuple and
        output_tuple = (day, mouse, result, duration, approach_latency, approach_number,
                        contact_latency, contact_number, *averages_linear, *sem_linear, *averages_circular,
                        *sem_circular, calcium_average, calcium_sem)
        output_df.append(output_tuple)

# prep the column names
lin_columns_mean = [el+'_mean' for el in final_linear]
lin_columns_sem = [el+'_sem' for el in final_linear]
ang_columns_mean = [el+'_mean' for el in final_circular]
ang_columns_sem = [el+'_sem' for el in final_circular]

# turn the list into a dataframe
output_df = pd.DataFrame(output_df, columns=['day', 'mouse', 'result', 'duration', 'latency_approach',
                                             'approach_number', 'latency_contact', 'contact_number'] +
                         lin_columns_mean + lin_columns_sem + ang_columns_mean + ang_columns_sem +
                                            ['calcium_average', 'calcium_sem'])
# save the file
output_df.to_hdf(out_path, 'data')

# generate database entry
# assemble the entry data
entry_data = {
    'analysis_type': 'beh_consolidate',
    'analysis_path': out_path,
    'date': '',
    'pic_path': '',
    'result': 'multi',
    'rig': rig,
    'lighting': 'multi',
    'imaging': 'multi',
    'slug': fm.slugify(os.path.basename(out_path)[:-5]),

}

# check if the entry already exists, if so, update it, otherwise, create it
update_url = '/'.join((paths.bondjango_url, 'analyzed_data', entry_data['slug'], ''))
output_entry = bd.update_entry(update_url, entry_data)
if output_entry.status_code == 404:
    # build the url for creating an entry
    create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
    output_entry = bd.create_entry(create_url, entry_data)

print('The output status was %i, reason %s' %
      (output_entry.status_code, output_entry.reason))
if output_entry.status_code in [500, 400]:
    print(entry_data)

