import pandas as pd
import numpy as np
import os
import processing_parameters
import functions_bondjango as bd


def pad_latents(input_list, target_size):
    """Get the latents from a preprocessing file and add them to the main dataframe with the correct padding"""

    # allocate an output list
    output_list = []
    # for all the elements in the input list
    for el in input_list:
        # determine the delta size for padding
        delta_frames = target_size - el.shape[0]
        # pad latents due to the VAME calculation window
        latent_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(el.columns))) * np.nan,
                                      columns=el.columns)
        # motif_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(motifs.columns))) * np.nan,
        #                              columns=motifs.columns)
        # pad them with nan at the edges (due to VAME excluding the edges
        latents = pd.concat([latent_padding, el, latent_padding], axis=0).reset_index(drop=True)
        # add to the output list
        output_list.append(latents)
        # motifs = pd.concat([motif_padding, motifs, motif_padding], axis=0).reset_index(drop=True)

    return output_list


def query_search_list():
    """Query the database based on the search_list field from processing_parameters"""
    # get the search list
    search_list = processing_parameters.search_list

    # allocate a list for all paths (need to preload to get the dates)
    path_list = []
    query_list = []
    # for all the search strings
    for search_string in search_list:
        # query the database for data to plot
        data_all = bd.query_database('analyzed_data', search_string)
        data_path = [el['analysis_path'] for el in data_all if '_preproc' in el['slug']]
        path_list.append(data_path)
        query_list.append(data_all)
    return path_list, query_list


def load_preprocessing(input_path, data_all, latents_flag=True, matching_flag=True, behavior_flag=False):
    """Load data from a list of preprocessing paths and their database data"""
    # load the data
    data_list = []
    meta_list = []
    frame_list = []
    for idx, el in enumerate(input_path):
        # get the trial timestamp (for frame calculations)
        time_stamp = int(''.join(os.path.basename(el).split('_')[3:6]))
        try:
            try:
                temp_data = pd.read_hdf(el, 'matched_calcium')
            except KeyError:
                if behavior_flag:
                    temp_data = pd.read_hdf(el, 'full_traces')
                else:
                    continue
            temp_data['id'] = data_all[idx]['id']
            meta_list.append([data_all[idx][el1] for el1 in processing_parameters.meta_fields])
            # if the latents flag is on, load the latents too
            if latents_flag:
                # try to load the motifs and latents
                try:
                    latents = pd.read_hdf(el, 'latents')
                    motifs = pd.read_hdf(el, 'motifs')
                    egocentric_coords = pd.read_hdf(el, 'egocentric_coord')
                    egocentric_coords = egocentric_coords.loc[:, ['cricket_0_x', 'cricket_0_y']]
                    egocentric_coords = egocentric_coords.rename(columns={'cricket_0_x': 'ego_cricket_x',
                                                                          'cricket_0_y': 'ego_cricket_y'})

                    # pad them with nan at the edges (due to VAME excluding the edges
                    [latents, motifs] = pad_latents([latents, motifs], temp_data.shape[0])
                    # concatenate with the main data
                    temp_data = pd.concat([temp_data, egocentric_coords, latents, motifs], axis=1)
                except KeyError:
                    print(f'No latents in file {el}')

            # if the matching flag is on, also load the ROI matching

            data_list.append(temp_data)
            frame_list.append([time_stamp, 0, temp_data.shape[0]])
        except KeyError:
            # data_list.append([])
            frame_list.append([time_stamp, 0, 0])
    return data_list, frame_list, meta_list


def load_regression():
    """Load regression files from a list of paths and the databased info"""
    return


def load_tc():
    """Load the tuning curves from a list of paths and the database info"""
    return


