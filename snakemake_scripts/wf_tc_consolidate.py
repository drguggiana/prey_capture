import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as dh
import functions_misc as fm
from functions_kinematic import wrap_negative, wrap


def kine_fraction_responsive(ds):
    return ds['Qual_test'].sum() / ds['Qual_test'].count()


def vis_frac_responsive(ds):
    is_ori_resp = ds['osi'] > 0.5
    is_dir_resp = ds['dsi_abs'] > 0.5
    frac_ori_resp = is_ori_resp.sum() / is_ori_resp.count()
    frac_dir_resp = is_dir_resp.sum() / is_dir_resp.count()

    is_vis_resp = is_ori_resp + is_dir_resp
    is_vis_resp = is_vis_resp > 0
    frac_vis_resp = is_vis_resp.sum() / is_vis_resp.count()

    return is_vis_resp, frac_vis_resp, frac_ori_resp, frac_dir_resp


# def cell_multimodal_reponses(resp_data):


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
        input_paths = snakemake.input
        slugs = [os.path.basename(el).replace('_tcday.hdf5', '') for el in input_paths]
        rigs = np.unique([slug.split('_')[6] for slug in slugs])
        day = np.unique([slug[:10] for slug in slugs])
        mouse = np.unique(["_".join(slug.split('_')[7:10]) for slug in slugs])
        results = [el for slug in slugs for el in slug.split('_') if
                   any(exp in el for exp in processing_parameters.wf_exp_types)]
        # read the output path and the input file urls
        out_path = os.path.join(paths.analysis_path, f'{day[0]}_{mouse[0]}_tcconsolidate.hdf5')
        dummy_out = snakemake.output[0]

    except NameError:
        # get the search string
        search_string = processing_parameters.search_string
        parsed_search = dh.parse_search_string(search_string)

        # get the paths from the database
        all_path = bd.query_database('analyzed_data', search_string)
        input_paths = np.array([el['analysis_path'] for el in all_path if ('_tcday' in el['slug']) and
                                (parsed_search['mouse'].lower() in el['slug'])])
        slugs = [os.path.basename(el) for el in input_paths]
        day = np.unique([el[:10] for el in slugs])
        rigs = np.unique([el.split('_')[6] for el in slugs])
        results = [el for slug in slugs for el in slug.split('_') if
                   any(exp in el for exp in processing_parameters.wf_exp_types)]

        # assemble the output path
        out_path = os.path.join(paths.analysis_path, 'test_tcconsolidate.hdf5')
        dummy_out = os.path.join(paths.analysis_path, 'test_tcdummy.txt')

    if 'control' in input_paths[0]:
        result = 'control'
    elif 'repeat' in input_paths[0]:
        result = 'repeat'
    elif 'fullfield' in input_paths[0]:
        result = 'fullfield'
    else:
        result = 'multi'

    if 'dark' in input_paths[0]:
        lighting = 'dark'
    else:
        lighting = 'normal'

    # If repeat experiments, rigs aren't unique, and so aren't useful for identifying matches.
    # Instead, use the result
    if result == 'repeat':
        id_flags = results
        rig = rigs[0]
    else:
        # If not repeat, use the rig to identify matches. Note that this is sorted alphabetically, so
        # VTuningWF (free session) comes first
        id_flags = rigs
        rig = 'multi'

    # Sort the input paths by id_flag so they are in the same order
    input_paths = [path for id in id_flags for path in input_paths if id in path]

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
        for data, id_flag in zip(data_list, id_flags):
            all_cells_summary_stats = {}
            matched_summary_stats = {}
            unmatched_summary_stats = {}
            multimodal_tuning = {}

            # get matches and save
            match_col = np.where([id_flag in el for el in matches.columns])[0][0]
            match_idxs = matches.iloc[:, match_col].to_numpy(dtype=int)
            num_cells = len(data['norm_spikes_viewed_props'].index)

            all_cells_summary_stats['num_matches'] = len(match_idxs)
            all_cells_summary_stats['match_frac'] = len(match_idxs) / num_cells
            all_cells_summary_stats['num_cells'] = num_cells
            matched_summary_stats['num_cells'] = len(match_idxs)
            unmatched_summary_stats['num_cells'] = num_cells - len(match_idxs)

            kine_features = [el for el in data.keys() if not any([x in el for x in ['props', 'counts', 'edges']])]
            vis_features = [el for el in data.keys() if 'props' in el]

            for feature in kine_features:
                # Save the whole dataset
                data[feature].to_hdf(out_path, f'{id_flag}/all_cells/{feature}')
                all_cells_summary_stats[f"frac_resp_{feature}"] = kine_fraction_responsive(data[feature])
                multimodal_tuning[feature] = data[feature]['Qual_test']

                # Save matched TCs
                matched_feature = data[feature].iloc[match_idxs, :].reset_index(names=['original_cell_id'])
                matched_feature.to_hdf(out_path, f'{id_flag}/matched/{feature}')
                matched_summary_stats[f"frac_resp_{feature}"] = kine_fraction_responsive(matched_feature)

                # save unmatched tcs
                unmatched_feature = data[feature].drop(match_idxs, axis=0)
                unmatched_feature.to_hdf(out_path, f'{id_flag}/unmatched/{feature}')
                unmatched_summary_stats[f"frac_resp_{feature}"] = kine_fraction_responsive(unmatched_feature)

            exp_vis_features = {}
            for feature in vis_features:
                feat = '_'.join(feature.split('_')[3:-1])

                # When looking at all the data, the feature name is empty, so give it a name
                if feat == "":
                    feat = "all"

                # Save the whole dataset
                all_is_resp, all_frac_resp, frac_ori_resp, frac_dir_resp = vis_frac_responsive(data[feature])
                data[feature]['Resp_test'] = all_is_resp
                data[feature].reset_index(names=['original_cell_id']).to_hdf(out_path, f'{id_flag}/all_cells/{feature}')
                all_cells_summary_stats[f"frac_resp_{feat}"] = all_frac_resp
                all_cells_summary_stats[f"frac_ori_resp_{feat}"] = frac_ori_resp
                all_cells_summary_stats[f"frac_dir_resp_{feat}"] = frac_dir_resp

                # Need to drop the index to match the kine features
                multimodal_tuning[feat] = data[feature]['Resp_test'].reset_index(drop=True)

                # Save matched TCs
                matched_feature = data[feature].iloc[match_idxs, :].reset_index(names=['original_cell_id'])
                matched_feature.to_hdf(out_path, f'{id_flag}/matched/{feature}')
                _, matched_frac_resp, matched_frac_ori_resp, matched_frac_dir_resp = vis_frac_responsive(
                    matched_feature)
                matched_summary_stats[f"frac_resp_{feat}"] = matched_frac_resp
                matched_summary_stats[f"frac_ori_resp_{feat}"] = matched_frac_ori_resp
                matched_summary_stats[f"frac_dir_resp_{feat}"] = matched_frac_dir_resp

                # Get the preferred orientation/direction from the matched cells
                exp_vis_features[feat] = matched_feature.loc[:,
                                         ['pref_dir', 'bootstrap_pref_dir', 'real_pref_dir', 'bootstrap_real_pref_dir',
                                         'resultant_dir', 'bootstrap_resultant_dir', 'shuffle_resultant_dir',
                                         'responsivity_dir', 'bootstrap_responsivity_dir', 'bootstrap_p_responsivity_dir',
                                         'shuffle_responsivity_dir', 'shuffle_p_responsivity_dir',
                                         'pref_ori', 'null_ori', 'bootstrap_pref_ori', 'real_pref_ori',
                                         'bootstrap_real_pref_ori',  'resultant_ori', 'bootstrap_resultant_ori',
                                          'shuffle_resultant_ori', 'responsivity_ori', 'bootstrap_responsivity_ori',
                                         'bootstrap_p_responsivity_ori', 'shuffle_responsivity_ori', 'shuffle_p_responsivity_ori',
                                         ]
                                         ]

                # save unmatched tcs
                unmatched_feature = data[feature].reset_index(names=['original_cell_id']).drop(index=match_idxs)
                unmatched_feature.to_hdf(out_path, f'{id_flag}/unmatched/{feature}')
                _, unmatched_frac_resp, unmatched_frac_ori_resp, unmatched_frac_dir_resp = vis_frac_responsive(
                    unmatched_feature)
                unmatched_summary_stats[f"frac_resp_{feat}"] = unmatched_frac_resp
                unmatched_summary_stats[f"frac_ori_resp_{feat}"] = unmatched_frac_ori_resp
                unmatched_summary_stats[f"frac_dir_resp_{feat}"] = unmatched_frac_dir_resp

            summary_stats = dicts_to_dataframe([all_cells_summary_stats, matched_summary_stats, unmatched_summary_stats],
                                               ['all_cells', 'matched', 'unmatched'])
            summary_stats.to_hdf(out_path, f'{id_flag}/summary_stats')

            visual_shifts[id_flag] = exp_vis_features

            # Make multimodal tuning dataframe
            multimodal_tuned = pd.DataFrame(multimodal_tuning, dtype=int)
            multimodal_tuned.to_hdf(out_path, f'{id_flag}/multimodal_tuned')

        # Calculate delta prefs for all matched cells
        # use the fixed rig as the reference
        delta_pref = pd.DataFrame()
        for tuning_type in visual_shifts[id_flags[0]].keys():
            # Recall that id_flags[0] is the free session, id_flags[1] is the fixed session
            free = visual_shifts[id_flags[0]][tuning_type]
            fixed = visual_shifts[id_flags[1]][tuning_type]
            diff_dir = free.loc[:, ['pref_dir']].subtract(fixed.loc[:, ['pref_dir']]).to_numpy().flatten()
            diff_ori = free.loc[:, ['pref_ori']].subtract(fixed.loc[:, ['pref_ori']]).to_numpy().flatten()

            diff_dir = wrap_negative(diff_dir)
            diff_ori = wrap(diff_ori, bound=180)

            delta_pref[f"delta_{tuning_type}_dir"] = diff_dir
            delta_pref[f"delta_{tuning_type}_ori"] = diff_ori

        # Now do the same, but with the still fixed session as the reference
        still_tuning = [el for el in visual_shifts[id_flags[0]].keys() if 'still' in el]
        all_tuning = [el for el in visual_shifts[id_flags[0]].keys() if 'still' not in el]
        for still_tuning, moving_tuning in zip(still_tuning, all_tuning):
            free_still = visual_shifts[id_flags[0]][still_tuning]
            fixed_still = visual_shifts[id_flags[1]][still_tuning]

            free_all = visual_shifts[id_flags[0]][moving_tuning]
            fixed_all = visual_shifts[id_flags[1]][moving_tuning]

            diff_rig_dir = free_still.loc[:, ['pref_dir']].subtract(fixed_still.loc[:,
                                                                    ['pref_dir']]).to_numpy().flatten()
            diff_moving_fixed_dir = fixed_all.loc[:, ['pref_dir']].subtract(fixed_still.loc[:,
                                                                            ['pref_dir']]).to_numpy().flatten()
            diff_moving_free_dir = free_all.loc[:, ['pref_dir']].subtract(fixed_still.loc[:,
                                                                          ['pref_dir']]).to_numpy().flatten()

            diff_rig_ori = free_still.loc[:, ['pref_ori']].subtract(fixed_still.loc[:,
                                                                    ['pref_ori']]).to_numpy().flatten()
            diff_moving_fixed_ori = fixed_all.loc[:, ['pref_ori']].subtract(fixed_still.loc[:,
                                                                            ['pref_ori']]).to_numpy().flatten()
            diff_moving_free_ori = free_all.loc[:, ['pref_ori']].subtract(fixed_still.loc[:,
                                                                          ['pref_ori']]).to_numpy().flatten()

            diff_rig_dir = wrap_negative(diff_rig_dir)
            diff_moving_fixed_dir = wrap_negative(diff_moving_fixed_dir)
            diff_moving_free_dir = wrap_negative(diff_moving_free_dir)

            diff_rig_ori = wrap(diff_rig_ori, bound=180)
            diff_moving_fixed_ori = wrap(diff_moving_fixed_ori, bound=180)
            diff_moving_free_ori = wrap(diff_moving_free_ori, bound=180)

            delta_pref[f"delta_dir_{moving_tuning}_free_still_rel_fixed_still"] = diff_rig_dir
            delta_pref[f"delta_dir_{moving_tuning}_fixed_moving_rel_fixed_still"] = diff_moving_fixed_dir
            delta_pref[f"delta_dir_{moving_tuning}_free_moving_rel_fixed_still"] = diff_moving_free_dir

            delta_pref[f"delta_ori_{moving_tuning}_free_still_rel_fixed_still"] = diff_rig_ori
            delta_pref[f"delta_ori_{moving_tuning}_fixed_moving_rel_fixed_still"] = diff_moving_fixed_ori
            delta_pref[f"delta_ori_{moving_tuning}_free_moving_rel_fixed_still"] = diff_moving_free_ori

        delta_pref.to_hdf(out_path, 'delta_vis_tuning')

    # assemble the entry data
    entry_data = {
        'analysis_type': 'tc_consolidate',
        'analysis_path': out_path,
        'date': '',
        'pic_path': '',
        'result': result,
        'rig': rig,
        'lighting': lighting,
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

