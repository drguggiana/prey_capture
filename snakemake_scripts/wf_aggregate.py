import os
import pandas as pd
import numpy as np
import itertools

import paths
import processing_parameters
import functions_data_handling as fdh
import functions_bondjango as bd
from functions_misc import slugify

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def concatenate_cell_matches(data_list, exp_type):

    renamed_data_list = []
    for df in data_list:
        old_cols = list(df.columns)
        if exp_type != 'repeat':
            new_cols = [col.split("_")[-2] for col in old_cols[:2]]
            new_cols += old_cols[2:]
        col_map = dict(zip(old_cols, new_cols))
        new_df = df.rename(columns=col_map)
        renamed_data_list.append(new_df)

    # concatenate the cell matches
    cell_matches = pd.concat(renamed_data_list).reset_index(names='old_index')
    return cell_matches



# Main script
mice = processing_parameters.cohort_1[:1]
results = ['multi']    # ['multi', 'fullfield', 'control'], ['repeat']
lightings = ['normal']
rigs = ['ALL']     # ['VWheelWF', 'VTuningWF'], ['ALL']    # 'ALL' used for everything but repeat aggs
analysis_type = 'tc_consolidate'

for mouse, result, light, rig in itertools.product(mice, results, lightings, rigs):

    # get the search string
    search_string = f"mouse:{mouse},result:{result},lighting:{light},rig:{rig},analysis_type:{analysis_type}"
    # get the paths from the database
    file_infos = bd.query_database("analyzed_data", search_string)

    # parse the search string
    search_string = fdh.remove_query_field(search_string, 'analysis_type')
    parsed_search = fdh.parse_search_string(search_string)

    input_paths = np.array([el['analysis_path'] for el in file_infos if (parsed_search['mouse'].lower() in el['slug'])])
    input_paths = np.array([in_path for in_path in input_paths if os.path.isfile(in_path)])

    if len(input_paths) == 0:
        print('No entries: ' + search_string)
        continue

    else:
        print(search_string)
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
                        data = tc[key]
                        if 'day' not in data.columns:
                            data['day'] = date
                            data['animal'] = mouse
                        data_dict[label] = data

                    data_list.append(data_dict)

        if len(data_list) > 0:
            concat_data_dict = {}
            # Aggregate it all
            for key in data_list[0].keys():
                if key == 'cell_matches':
                    df = concatenate_cell_matches([d[key] for d in data_list], parsed_search['result'])
                else:
                    df = pd.concat([d[key] for d in data_list]).reset_index(names='old_index')

                concat_data_dict[key] = df
                # df.to_hdf(output_path, key)

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
                'slug': slugify(os.path.basename(output_path)[:-5]),
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

print('Done!')
