# imports
import os
import functions_misc as fm
import paths
import functions_bondjango as bd
import itertools as it
import pandas as pd
import datetime
import numpy as np
import re


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
        'imaging': parsed_search['imaging'],
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
        'result': '',
        'rig': '',
        'lighting': '',
        'slug': '',
        'notes': '',
        'imaging': '',
        'date': '',
        'gtdate': '',
        'ltdate': '',
        'mouse': '',
        'analysis_type': '',
    }
    # for all the keys, find the matching terms and fill in the required entries
    for key in string_dict.keys():
        # for all the elements in string_parts
        for parts in string_parts:
            # split by the separator
            subparts = parts.split(':')
            # compare removing spaces
            if key == subparts[0].strip():
                # parse the different arguments, also removing external spaces
                string_dict[key] = subparts[1].strip()
        # fill it up with ALL if not present
        if string_dict[key] == '':
            string_dict[key] = 'ALL'
    return string_dict


def parse_experiment_name(experiment_name):
    """Parse an experiment file path into a dictionary"""
    path_basename = os.path.basename(experiment_name)[:-4]
    path_parts = path_basename.split('_')
    # get the analysis type for the url
    url_type = 'video_experiment/' if path_parts[6] == 'miniscope' else 'vr_experiment/'
    # assemble the parsed dictionary
    parsed_name = {
        'date': '_'.join((path_parts[:6])),
        'mouse': '_'.join((path_parts[7:10])) if path_parts[6] == 'miniscope' else '_'.join((path_parts[6:9])),
        'rig': 'miniscope' if path_parts[6] == 'miniscope' else 'vr',
        'result': path_parts[10] if path_parts[6] == 'miniscope' else path_parts[9],
        # TODO: generalize this
        'lighting': 'normal' if 'dark' not in experiment_name else 'dark',
        'imaging': 'doric',
        'notes': ''.join((path_parts[11:])) if path_parts[6] == 'miniscope' else ''.join((path_parts[10:])),
        'slug': fm.slugify(path_basename),
        'url': 'http://192.168.236.135:8080/loggers/' + url_type+fm.slugify(path_basename)+'/',
        'bonsai_path': experiment_name,
        'fluo_path': experiment_name.replace('.csv', '_calcium_data.h5'),
        'tif_path': experiment_name.replace('.csv', '.tif'),
        'sync_path': experiment_name.replace('miniscope', 'syncMini', 1) if path_parts[6] == 'miniscope' else
        '_'.join((path_parts[:6], 'syncVR', path_parts[6:]))
    }

    return parsed_name


def save_create_snake(data_in, paths_in, file_name, hdf5_key, parsed_query, action='both', mode='w'):
    """Save the data frame and create a database entry for snakemake"""
    # get the actual analysis type from the hdf5 key
    analysis_type = hdf5_key.split('/')[-1]

    # check which actions to perform based on the kwarg
    if action in ['save', 'both']:
        # save as dataframe
        data_in.to_hdf(file_name, key=hdf5_key, mode=mode, format='fixed')

    if action in ['create', 'both']:
        # generate an entry
        generate_entry(paths_in, file_name, parsed_query, analysis_type)
    return None


def combinatorial_query(input_dict):
    """Given the input dict, generate all combinations of the elements as search queries"""
    # get the keys of the dict
    fields_list = input_dict.keys()
    # get the cartesian product of the values
    combinations = it.product(*input_dict.values())
    # allocate the output list
    search_queries = []
    # for all the combinations
    for combination in combinations:
        # assemble the actual search query
        search_queries.append(','.join([field + ':' + term for field, term
                                        in zip(fields_list, combination)]))
    return search_queries


def remove_query_field(input_query, target_field):
    """Remove the target query field from the search query"""
    # split the query
    split_query = input_query.split(',')
    # allocate memory for the result
    output_query = []
    # for all the parts
    for parts in split_query:
        # split in field and query
        field = parts.split(':')[0]
        # if it's the target field, skip it
        if field == target_field:
            continue
        # otherwise put it in the output query
        output_query.append(parts)
    # assemble the new query minus the target field
    output_query = ','.join(output_query)
    return output_query


def fetch_preprocessing(search_query):
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
        # date_list = [datetime.datetime.strptime(el[m2m_field][0][:10], '%m_%d_%Y') for el in file_path]
        date_list = [datetime.datetime.strptime(el['date'], '%Y-%m-%dT%H:%M:%SZ').date() for el in file_path]

        # filter the list for animals - the regex searches for animal names in the 
        # form of initials_YYMMDD_letter
        # animal_list = [el[m2m_field][0][30:41] for el in file_path]
        animal_list = [re.search(r"([a-zA-Z]+)\_(\d+)\_([a-zA-Z]+)", el[m2m_field][0])[0] for el in file_path]
    else:
        date_list = []
        animal_list = []
    # # load the data
    # data_all = [pd.read_hdf(el['analysis_path'], sub_key) for el in file_path]
    # get the paths
    paths_all = [el['analysis_path'] for el in file_path]

    return file_path, paths_all, parsed_query, date_list, animal_list


def aggregate_loader(data_path):
    """Load the data from the given hdf5 file"""
    # with pd.HDFStore(data_all[0]['analysis_path']) as h:
    with pd.HDFStore(data_path) as h:
        # get a list of the existing keys
        keys = h.keys()
        # if it's only one key, just load the file
        if len(keys) == 1:
            data = h[keys[0]]
        else:
            # allocate a dictionary for them
            data = {}
            # extract the animals present
            animal_list = [el.split('/')[1] for el in keys]
            # get the unique animals
            unique_animals = np.unique(animal_list)
            # for all the animals
            for animal in unique_animals:
                # allocate a dictionary for the tables
                sub_dict = {}
                # get the unique dates for this animal
                date_list = [el.split('/')[2] for el in keys if animal in el]
                # for all the unique dates
                for date in date_list:
                    # get the corresponding key
                    current_key = [el for el in keys if (animal in el) and (date in el)][0]
                    print(current_key)
                    # load the table into the dictionary (minus the d at the beginning,
                    # added cause natural python naming)
                    sub_dict[date[1:]] = h[current_key]
                # save the dictionary into the corresponding entry of the animal dictionary
                data[animal] = sub_dict
    return data

