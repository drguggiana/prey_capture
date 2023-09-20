import os
import numpy as np
import pandas as pd
import warnings

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as dh
import functions_misc as fm
from functions_kinematic import wrap_negative

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def kine_fraction_responsive(ds):
    return ds['Resp_test'].sum() / ds['Resp_test'].count()


def vis_frac_responsive(ds):
    is_resp = ds['responsivity'] > 0.25
    return is_resp.sum() / is_resp.count()


def dicts_to_dataframe(dicts, index):
    df_list = []
    for d, idx in zip(dicts, index):
        vals = np.array(list(d.values()))
        cols = np.array(list(d.keys()))
        df = pd.DataFrame(data=np.expand_dims(vals, axis=-1).T, columns=cols, index=[idx])
        df_list.append(df)

    df_concat = pd.concat(df_list)
    return df_concat


if __name__ == '__main__':
    try:
        # get the input
        input_paths = snakemake.input[0]
        rigs = [path.split('_')[6] for path in input_paths]
        day = np.unique([path[:4] for path in input_paths])
        mouse = np.unique([path[7:10] for path in input_paths])
        # read the output path and the input file urls
        out_path = os.path.join(paths.analysis_path, f'{day[0]}_{mouse[0]}_tcconsolidate.hdf5')
        dummy_out = snakemake.output

    except NameError:
        # get the search string
        search_string = processing_parameters.search_string
        parsed_search = dh.parse_search_string(search_string)

        # get the paths from the database
        all_path = bd.query_database('analyzed_data', search_string)
        input_paths = np.array([el['analysis_path'] for el in all_path if ('_tcday' in el['slug']) and
                                (parsed_search['mouse'].lower() in el['slug'])])
        day = np.unique([path[:4] for path in input_paths])
        slug = [os.path.basename(el) for el in input_paths]
        rigs = np.array([el.split('_')[6] for el in slug])

        # assemble the output path
        out_path = os.path.join(paths.analysis_path, 'test_tcconsolidate.hdf5')
        dummy_out = os.path.join(paths.analysis_path, 'test_tcdummy.txt')

    # cycle through the files
    matches = None
    empty_flag = False
    data_list = []

    for file in input_paths:
        # Load the data
        file_dict = {}
        with pd.HDFStore(file, mode='r') as h:

            for key in h.keys():
                if key == '/no_ROIs':
                    empty_flag = True
                    break
                elif key == '/cell_matches':
                    matches = h[key].dropna().reset_index(drop=True)
                else:
                    file_dict[key.split('/')[-1]] = h[key]

            data_list.append(file_dict)

    if empty_flag:
        empty = pd.DataFrame([])
        empty.to_hdf(out_path, 'no_ROIs')

    else:
        # Save cell matches
        if matches is not None:
            matches.to_hdf(out_path, 'cell_matches')

        visual_shifts = {}
        for data, rig in zip(data_list, rigs):
            all_cells_stats = {}
            matched_stats = {}
            unmatched_stats = {}

            # get matches and save
            match_col = np.where([rig in el for el in matches.columns])[0][0]
            match_idxs = matches.iloc[:, match_col].to_numpy(dtype=int)
            num_cells = len(data['norm_spikes_viewed_direction_props'].index)

            all_cells_stats['num_cells'] = num_cells
            all_cells_stats['num_matches'] = len(match_idxs)
            all_cells_stats['match_frac'] = len(match_idxs)/num_cells
            matched_stats['num_cells'] = len(match_idxs)
            unmatched_stats['num_cells'] = num_cells - len(match_idxs)

            kine_features = [el for el in data.keys() if not any([x in el for x in ['props', 'counts', 'edges']])]
            vis_features = [el for el in data.keys() if 'props' in el]

            for feature in kine_features:
                # Save the whole dataset
                data[feature].to_hdf(out_path, f'{rig}/all_cells/{feature}')
                all_cells_stats[f"frac_resp_{feature}"] = kine_fraction_responsive(data[feature])

                # Save matched TCs
                matched_feature = data[feature].iloc[match_idxs, :].reset_index(names=['original_cell_id'])
                matched_feature.to_hdf(out_path, f'{rig}/matched/{feature}')
                matched_stats[f"frac_resp_{feature}"] = kine_fraction_responsive(matched_feature)

                # save unmatched tcs
                unmatched_feature = data[feature].drop(match_idxs, axis=0)
                unmatched_feature.to_hdf(out_path, f'{rig}/unmatched/{feature}')
                unmatched_stats[f"frac_resp_{feature}"] = kine_fraction_responsive(unmatched_feature)

            exp_vis_features = {}
            for feature in vis_features:
                feat = '_'.join(feature.split('_')[3:-1])
                # Save the whole dataset
                data[feature].reset_index(names=['original_cell_id']).to_hdf(out_path, f'{rig}/all_cells/{feature}')
                all_cells_stats[f"frac_resp_{feat}"] = vis_frac_responsive(data[feature])

                # Save matched TCs
                matched_feature = data[feature].iloc[match_idxs, :].reset_index(names=['original_cell_id'])
                matched_feature.to_hdf(out_path, f'{rig}/matched/{feature}')
                matched_stats[f"frac_resp_{feat}"] = vis_frac_responsive(matched_feature)

                # Get the preferred orientation/direction from the matched cells
                exp_vis_features[feat] = matched_feature.loc[:, ['pref', 'responsivity', 'p_responsivity', 'bootstrap_responsivity']]

                # save unmatched tcs
                unmatched_feature = data[feature].reset_index(names=['original_cell_id']).drop(index=match_idxs)
                unmatched_feature.to_hdf(out_path, f'{rig}/unmatched/{feature}')
                unmatched_stats[f"frac_resp_{feat}"] = vis_frac_responsive(unmatched_feature)

            summary_stats = dicts_to_dataframe([all_cells_stats, matched_stats, unmatched_stats],
                                               ['all_cells', 'matched', 'unmatched'])
            summary_stats.to_hdf(out_path, f'{rig}/summary_stats')

            visual_shifts[rig] = exp_vis_features

        # Calculate delta prefs for all matched cells
        # use the fixed rig as the reference
        delta_pref = pd.DataFrame()
        for tuning_type in visual_shifts[rigs[0]].keys():
            fixed = visual_shifts[rigs[0]][tuning_type]
            free = visual_shifts[rigs[1]][tuning_type]
            diff = fixed.loc[:, ['pref']].subtract(free.loc[:, ['pref']]).to_numpy()
            if 'direction' in tuning_type:
                diff = wrap_negative(diff.flatten())
            delta_pref[f"delta_{tuning_type}"] = diff

        delta_pref.to_hdf(out_path, 'delta_vis_tuning')

        print('hi')

    # assemble the entry data
    entry_data = {
        'analysis_type': 'tc_consolidate',
        'analysis_path': out_path,
        'date': '',
        'pic_path': '',
        'result': 'multi',
        'rig': 'multi',
        'lighting': 'multi',
        'imaging': 'wirefree',
        'slug': fm.slugify(os.path.basename(out_path)[:-5]),

    }

    with open(dummy_out, 'w') as f:
        f.writelines(input_paths)

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

