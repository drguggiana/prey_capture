# imports
import os
import functions_misc as fm
import paths
import functions_bondjango as bd
import pandas as pd
import datetime
# import deepdish as dd


def generate_entry(in_path, out_path, parsed_search, analysis_t, pic_path=''):
    """Given an in path, out path, search string and options, create or update an analysis entry"""

    # assemble the entry data
    entry_data = {
        'analysis_type': analysis_t,
        'analysis_path': out_path,
        'input_path': ','.join(in_path),
        'pic_path': pic_path,
        'result': parsed_search['result'],
        'rig': parsed_search['rig'],
        'lighting': parsed_search['lighting'],
        'slug': fm.slugify(os.path.basename(out_path)[:-5]),
        'notes': parsed_search['notes'],
    }
    # try to update, if not possible, create
    update_url = '/'.join((paths.bondjango_url, 'analyzed_data', entry_data['slug'], ''))
    output_entry = bd.update_entry(update_url, entry_data)
    if output_entry.status_code == 404:
        # build the url for creating an entry
        create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
        output_entry = bd.create_entry(create_url, entry_data)
    # print the result
    print('The output status was %i, reason %s' % (output_entry.status_code, output_entry.reason))
    return entry_data['slug']


def parse_search_string(string_in):
    """Get the queries out of the search string"""

    # split the string into parts
    string_parts = string_in.split(',')
    # allocate a dictionary
    string_dict = {
        'analysis_type': '',
        'result': '',
        'rig': '',
        'lighting': '',
        'slug': '',
        'notes': '',
    }
    # for all the keys, find the matching terms and fill in the required entries
    for key in string_dict.keys():
        # for all the elements in string_parts
        for parts in string_parts:
            if key in parts:
                string_dict[key] = parts.split('=')[1]

    return string_dict


def save_create(data_in, paths_in, hdf5_key, parsed_query, action='both', mode='w'):
    """Save the data frame and create a database entry"""
    # get the actual analysis type from the hdf5 key
    analysis_type = hdf5_key.split('/')[-1]
    # save a file
    file_name = os.path.join(paths.analysis_path,
                             '_'.join([el for el in parsed_query.values() if len(el) > 0] +
                                      [analysis_type])+'.hdf5')

    # check which actions to perform based on the kwarg
    if action in ['save', 'both']:
        # save as dataframe
        data_in.to_hdf(file_name, key=hdf5_key, mode=mode, format='table')

    if action in ['create', 'both']:
        # generate an entry
        generate_entry(paths_in, file_name, parsed_query, analysis_type)
    return None


def fetch_preprocessing(search_query, sub_key=None):
    """Take the search query and load the data and the input paths"""
    # get the queryset
    file_path = bd.query_database('analyzed_data', search_query)

    assert len(file_path) > 0, 'Query gave no results'

    # parse the search query
    parsed_query = parse_search_string(search_query)

    # if coming from vr or video, also get lists for the dates and animals
    if parsed_query['analysis_type'] == 'preprocessing':
        if parsed_query['rig'] == 'miniscope':
            m2m_field = 'video_analysis'
        else:
            m2m_field = 'vr_analysis'

        # filter the path for the days
        date_list = [datetime.datetime.strptime(el[m2m_field][0][:10], '%m_%d_%Y') for el in file_path]

        # filter the list for animals
        animal_list = [el[m2m_field][0][30:41] for el in file_path]
    else:
        date_list = []
        animal_list = []
    # load the data
    data_all = [pd.read_hdf(el['analysis_path'], sub_key) for el in file_path]
    # get the paths
    paths_all = [el['analysis_path'] for el in file_path]

    return data_all, paths_all, parsed_query, date_list, animal_list
