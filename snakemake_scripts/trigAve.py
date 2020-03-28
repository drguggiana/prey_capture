# find the activity triggered average for the neural activity
import yaml
import functions_data_handling as fd
import paths
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import datetime
import functions_plotting as fp

try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # get the input paths
    paths_all = snakemake.input
    # get the parsed path info
    path_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.file_info]
    # get the ATA interval
    interval = snakemake.params.interval
    # get the analysis type
    analysis_type = dict_path['analysis_type']
    # get a list of all the animals and dates involved
    animal_list = [el['mouse'] for el in path_info]
    date_list = [datetime.datetime.strptime(el['date'], '%Y-%m-%dT%H:%M:%SZ').date() for el in path_info]
except NameError:
    # define the analysis type
    analysis_type = 'trigAveCA'
    # define the search query
    search_query = 'result:succ,lighting:normal,rig:miniscope,imaging:doric'
    # define the origin model
    ori_type = 'preprocessing'
    # get a dictionary with the search terms
    dict_path = fd.parse_search_string(search_query)

    # get the info and paths
    path_info, paths_all, parsed_query, date_list, animal_list = \
        fd.fetch_preprocessing(search_query + ', =analysis_type:' + ori_type)
    # get the raw output_path
    dict_path['analysis_type'] = analysis_type
    basic_name = '_'.join(dict_path.values())
    raw_path = os.path.join(paths.analysis_path, '_'.join((ori_type, basic_name))+'.hdf5')
    # define the interval for ATA calculation (in seconds, before and after)
    interval = [2, 3]


# read the data
data = [pd.read_hdf(el, 'matched_calcium') for el in paths_all]
# get the unique animals and dates
unique_mice = np.unique(animal_list)
unique_dates = np.unique(date_list)
# set the flag for first run
first_run = 1
# for all the mice
for mouse in unique_mice:
    # for all the dates
    for date in unique_dates:
        # allocate memory for the peaks
        ata_frame = []
        # get the data for this mouse and day
        sub_data = [el for idx, el in enumerate(data)
                    if (date_list[idx] == date) & (animal_list[idx] == mouse)]

        if len(sub_data) == 0:
            continue
        elif len(sub_data) == 1:
            sub_data = sub_data[0]
        elif len(sub_data) > 1:
            sub_data = pd.concat(sub_data)

        # separate the calcium data from the rest
        labels = sub_data.columns
        cells = [el for el in labels if 'cell' in el]
        others = [el for el in labels if 'cell' not in el]
        calcium_data = sub_data.loc[:, cells]
        variables = sub_data.loc[:, others].reset_index()
        # calculate the framerate (in fps)
        framerate = 1/np.nanmean(np.diff(variables['time_vector'].to_numpy()[:100]))
        # determine how many frames before and after are needed for the peaks
        frames = np.round(np.array(interval)*framerate).astype(np.uint16)
        # for each cell
        for cell in cells:
            # get the cells data
            cell_data = calcium_data.loc[:, cell].reset_index().to_numpy()[:, 1]
            # allocate memory for the ATAs
            ata_list = []
            # calculate the 8th percentile for the deadline
            baseline = np.percentile(cell_data, 8)
            baseline_std = np.nanstd(calcium_data)
            # find the peaks in activity
            peaks = np.expand_dims(find_peaks(cell_data, height=baseline+baseline_std*2)[0], axis=1)
            # for all the peaks
            for idx, peak in enumerate(peaks):
                # if out of bounds, just skip
                try:
                    # get the interval for the peak
                    ata_list.append(variables.iloc[(peak[0]-frames[0]):(peak[0]+frames[1]), :].copy())
                    # combine it with a peak id vector
                    ata_list[-1]['peak_id'] = idx
                except IndexError:
                    continue
            # if there were no peaks, skip the cell
            if len(ata_list) == 0:
                continue
            # concatenate them in a dataframe
            ata_frame.append(pd.concat(ata_list, axis=0))
            ata_frame[-1]['cell'] = cell

        # cell_peaks = cell_data[peaks]
        # # assemble the plotting
        # cell_plot = np.concatenate([peaks, cell_peaks], axis=1)
        # fp.histogram([[cell_data]], dpi=50, bins=100)
        # fp.plot_2d([[cell_data, cell_plot]], linestyle=[['-', 'None']], dpi=50)
        # concatenate the final data
        ata_final = pd.concat(ata_frame, axis=0)
        # assemble the group key for the hdf5 file (making sure the date is natural)
        group_key = '/'.join(('', mouse, 'd' + str(date)[:10].replace('-', '_'), analysis_type))

        # if its the first run, overwrite the file, otherwise, append
        if first_run == 1:
            mode = 'w'
            first_run += 1
        else:
            mode = 'a'
        # save to file
        fd.save_create_snake(ata_final, paths_all, raw_path, group_key, dict_path, action='save', mode=mode)

    # create the entry
    fd.save_create_snake([], paths_all, raw_path, analysis_type, dict_path, action='create')

print('yay')
