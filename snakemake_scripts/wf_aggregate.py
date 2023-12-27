import pandas as pd
import numpy as np
import os

import functions_data_handling as fd
import functions_bondjango as bd
import paths

import processing_parameters

# Main script
try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # get the input paths
    paths_all = snakemake.input
    # get the parsed path info
    path_info = [yaml.load(el, Loader=yaml.FullLoader) for el in snakemake.params.file_info]
    # get the analysis type
    analysis_type = dict_path['analysis_type']
    # get a list of all the animals and dates involved
    animal_list = [el['mouse'] for el in path_info]
    date_list = [datetime.datetime.strptime(el['date'], '%Y-%m-%dT%H:%M:%SZ').date() for el in path_info]

except NameError:
    # define the search query
    search_query = processing_parameters.search_string

    # define the origin model
    ori_type = 'tc_consolidate'
    # get a dictionary with the search terms
    dict_path = fd.parse_search_string(search_query)

    # get the info and paths
    path_info, paths_all, parsed_query, date_list, animal_list = \
        fd.fetch_preprocessing(search_query + ', analysis_type:' + ori_type)
    # get the raw output_path
    basic_name = '_'.join(dict_path.values())


unique_mice = np.unique(animal_list)

for mouse in unique_mice:

    # get the search string
    search_string = f"mouse:{mouse}, result:multi, lighting:normal"

    parsed_search = fd.parse_search_string(search_string)

    # get the paths from the database
    file_infos = bd.query_database("analyzed_data", search_string)
    input_paths = np.array([el['analysis_path'] for el in file_infos if ('_tcconsolidate' in el['slug']) and
                            (parsed_search['mouse'].lower() in el['slug'])])
    input_paths = np.array([in_path for in_path in input_paths if os.path.isfile(in_path)])
    print(np.sort(input_paths))

    if len(input_paths) == 0:
        continue

    else:
        date_list = [os.path.basename(path)[:10] for path in input_paths]
        mouse = parsed_search['mouse']

        # assemble the output path
        output_path = os.path.join(paths.analysis_path, f"AGG_{'_'.join(parsed_search.values())}.hdf5")

        data_list = []
        for file, date in zip(input_paths, date_list):
            data_dict = {}
            with pd.HDFStore(file, 'r') as tc:
                # print(tc.keys())
                if '/no_ROIs' in tc.keys():
                    continue
                else:
                    for key in tc.keys():
                        label = "_".join(key.split('/')[1:])
                        # if parsed_search['result'] == 'repeat':
                        #     exp_id = label.split('_')[0][:-1]
                        #     label_base = label.split('_')[1:]
                        #     label = '_'.join(([exp_id] + label_base)
                        data = tc[key]
                        data['date'] = date
                        data_dict[label] = data

                    data_list.append(data_dict)

        if len(data_list) > 0:
            # Aggregate it all

            for key in data_list[0].keys():
                df = pd.concat([d[key] for d in data_list]).reset_index(names='old_index')
                df.to_hdf(output_path, key)

            # assemble the entry data
            entry_data = {
                'analysis_type': 'agg_tc',
                'analysis_path': output_path,
                'date': '',
                'pic_path': '',
                'result': parsed_search['result'],
                'rig': parsed_search['rig'],
                'lighting': parsed_search['lighting'],
                'imaging': 'wirefree',
                'slug': misc.slugify(os.path.basename(output_path)[:-5]),
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