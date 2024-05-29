import os
import yaml
import warnings

import numpy as np
import pandas as pd

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as dh
import functions_misc as fm
from functions_kinematic import wrap_negative, wrap

# Ignore warnings
warnings.simplefilter(action='ignore')


def rename_match_df(matches, exp_type):
    old_cols = list(matches.columns)
    if exp_type != 'repeat':
        new_cols = [col.split("_")[-2] for col in old_cols[:2]]
        new_cols += old_cols[2:]
    else:
        new_cols = [col.split("_")[-1] for col in old_cols[:2]]
    col_map = dict(zip(old_cols, new_cols))
    new_df = matches.rename(columns=col_map)
    return new_df


def kine_fraction_tuned(ds):
    frac_qual = ds['Qual_test'].sum() / ds['Qual_test'].count()
    frac_cons = ds['Cons_test'].sum() / ds['Cons_test'].count()
    frac_resp = ds['Resp_test'].sum() / ds['Resp_test'].count()

    # is tuned if quality and consistency are both true
    is_resp = ds['Resp_test'].astype(bool) & ds['Qual_test'].astype(bool)
    frac_is_resp = is_resp.sum() / is_resp.count()
    return frac_is_resp, is_resp


def vis_frac_responsive(data, sel_tresh=0.5):
    resp_df = data.loc[:, ['is_vis_resp', 'mod_vis_resp', 'not_vis_resp']].copy()

    # Get boolean vector of direction and orientation tuned cells
    resp_df['vis_resp_dir_tuned'] = (resp_df['is_vis_resp'] == 1) & (data['fit_dsi'] >= sel_tresh)
    resp_df['vis_resp_ori_tuned'] = (resp_df['is_vis_resp'] == 1) & (data['fit_osi'] >= sel_tresh)

    # For those that are both direction and orientation tuned, pick the tuning based on the higher value
    resp_df['vis_resp_dir_tuned'] = resp_df['vis_resp_dir_tuned'] & (data['fit_dsi'] > data['fit_osi'])
    resp_df['vis_resp_ori_tuned'] = resp_df['vis_resp_ori_tuned'] & (data['fit_osi'] > data['fit_dsi'])

    # Do the same but for moderate tuning
    resp_df['mod_resp_dir_tuned'] = (resp_df['mod_vis_resp'] == 1) & (data['fit_dsi'] >= sel_tresh)
    resp_df['mod_resp_ori_tuned'] = (resp_df['mod_vis_resp'] == 1) & (data['fit_osi'] >= sel_tresh)
    resp_df['mod_resp_dir_tuned'] = resp_df['mod_resp_dir_tuned'] & (data['fit_dsi'] > data['fit_osi'])
    resp_df['mod_resp_ori_tuned'] = resp_df['mod_resp_ori_tuned'] & (data['fit_osi'] > data['fit_dsi'])

    # And for those not responsive
    resp_df['not_resp_dir_tuned'] = (resp_df['not_vis_resp'] == 1) & (data['fit_dsi'] >= sel_tresh)
    resp_df['not_resp_ori_tuned'] = (resp_df['not_vis_resp'] == 1) & (data['fit_osi'] >= sel_tresh)
    resp_df['not_resp_dir_tuned'] = resp_df['not_resp_dir_tuned'] & (data['fit_dsi'] > data['fit_osi'])
    resp_df['not_resp_ori_tuned'] = resp_df['not_resp_ori_tuned'] & (data['fit_osi'] > data['fit_dsi'])

    # Get the fraction responsive cells out of all cells
    frac_resp_overall = resp_df.sum() / resp_df.count()

    # Now get fraction responsive cells within groups
    frac_resp_within = dict()
    frac_resp_within['vis_resp_dir_tuned_within'] = resp_df['vis_resp_dir_tuned'].sum() / resp_df['is_vis_resp'].sum()
    frac_resp_within['vis_resp_ori_tuned_within'] = resp_df['vis_resp_ori_tuned'].sum() / resp_df['is_vis_resp'].sum()
    frac_resp_within['mod_resp_dir_tuned_within'] = resp_df['mod_resp_dir_tuned'].sum() / resp_df['mod_vis_resp'].sum()
    frac_resp_within['mod_resp_ori_tuned_within'] = resp_df['mod_resp_ori_tuned'].sum() / resp_df['mod_vis_resp'].sum()
    frac_resp_within['not_resp_dir_tuned_within'] = resp_df['not_resp_dir_tuned'].sum() / resp_df['not_vis_resp'].sum()
    frac_resp_within['not_resp_ori_tuned_within'] = resp_df['not_resp_ori_tuned'].sum() / resp_df['not_vis_resp'].sum()
    frac_resp_within = pd.DataFrame.from_dict(frac_resp_within, orient='index')

    frac_resp = pd.concat([frac_resp_overall, frac_resp_within], axis=0).T
    new_cols = [f'frac_{col}' for col in frac_resp.columns]
    frac_resp.columns = new_cols

    return resp_df, frac_resp


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
        mouse = parsed_search['mouse']

        # get the paths from the database
        all_path = bd.query_database('analyzed_data', search_string)
        input_paths = np.array([el['analysis_path'] for el in all_path if ('_tcday' in el['slug']) and
                                (parsed_search['mouse'].lower() in el['slug'])])
        slugs = [os.path.basename(el) for el in input_paths]
        day = np.unique([el[:10] for el in slugs])[0]
        rigs = np.unique([el.split('_')[6] for el in slugs])
        results = [el for slug in slugs for el in slug.split('_') if
                   any(exp in el for exp in processing_parameters.wf_exp_types)]

        # assemble the output path
        out_path = os.path.join(paths.analysis_path, f'{day}_{mouse}_tcconsolidate.hdf5')
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
        id_flags = np.sort(results)
        rig = rigs[0]
    else:
        # If not repeat, use the rig to identify matches.
        # Note that this is sorted alphabetically, so VTuningWF (free session) comes first
        id_flags = rigs
        rig = 'multi'

    # Set the dataset to use
    used_tc_dataset = processing_parameters.activity_datasets[0]

    # Sort the input paths by id_flag so they are in the same order
    input_paths = [path for id in id_flags for path in input_paths if id in path]

    # Load file to exclude from analysis
    with open(paths.file_exclusion_path, 'r') as f:
        # Load the contents of the file into a dictionary
        files_to_exclude = yaml.unsafe_load(f)

    # Exclude the files
    if result in ['control', 'repeat', 'fullfield']:
        exclude_from_this = files_to_exclude.get(result, [])
    else:
        exclude_from_this = files_to_exclude.get('multi', [])

    slugs_wo_filetype = [slug.replace('_tcday.hdf5', '') for slug in slugs]
    if any([s in exclude_from_this for s in slugs_wo_filetype]):
        print('At least one experiment has a file to exclude. Skipping')

        # Write to the dummy output file
        with open(dummy_out, 'w') as f:
            f.writelines(input_paths)

    else:

        # Put this in a try - excepts structure to catch errors and not write the output file or the database entry
        try:
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
                    matches = rename_match_df(matches, result)
                    matches.to_hdf(out_path, 'cell_matches')

                visual_shifts = {}
                for data, id_flag in zip(data_list, id_flags):

                    kine_features = [el for el in data.keys() if not any([x in el for x in ['props', 'counts', 'edges',
                                                                                            'running_modulated']])]
                    vis_features = [el for el in data.keys() if used_tc_dataset in el]

                    # Initialize some DataFrames for saving the summary stats
                    all_cells_summary_stats = pd.DataFrame(columns=['num_cells', 'num_matches', 'match_frac'])
                    matched_summary_stats = pd.DataFrame(columns=['num_cells', 'num_matches', 'match_frac'])
                    unmatched_summary_stats = pd.DataFrame(columns=['num_cells', 'num_matches', 'match_frac'])

                    # Make a DataFrame for saving the binary tuning state as well as the tuning strengths
                    multimodal_tuning = pd.DataFrame()

                    # get matches and save
                    match_idxs = matches.loc[:, id_flag].to_numpy(dtype=int)
                    num_cells = data[f'{used_tc_dataset}_props'].shape[0]

                    all_cells_summary_stats.loc[0, 'num_cells'] = num_cells
                    all_cells_summary_stats.loc[0, 'num_matches'] = len(match_idxs)
                    all_cells_summary_stats.loc[0, 'match_frac'] = len(match_idxs) / num_cells
                    matched_summary_stats.loc[0, 'num_cells'] = len(match_idxs)
                    unmatched_summary_stats.loc[0, 'num_cells'] = num_cells - len(match_idxs)

                    # Save the running modulated cells
                    run_mod = data['running_modulated_cells']
                    run_mod.to_hdf(out_path, f'{id_flag}/all_cells/running_modulated_cells')
                    frac_run_mod = run_mod['sig_run_modulated'].sum() / run_mod['sig_run_modulated'].count()
                    frac_run_vis_mod = run_mod['sig_vis_run_modulated'].sum() / run_mod['sig_vis_run_modulated'].count()
                    all_cells_summary_stats['frac_run_mod'] = frac_run_mod
                    all_cells_summary_stats['frac_run_vis_mod'] = frac_run_vis_mod

                    matched_run_mod = run_mod.iloc[match_idxs, :].reset_index(drop=True)
                    matched_run_mod.to_hdf(out_path, f'{id_flag}/matched/running_modulated_cells')
                    matched_frac_run_mod = matched_run_mod['sig_run_modulated'].sum() / matched_run_mod[
                        'sig_run_modulated'].count()
                    matched_frac_run_vis_mod = matched_run_mod['sig_vis_run_modulated'].sum() / matched_run_mod[
                        'sig_vis_run_modulated'].count()
                    matched_summary_stats['frac_run_mod'] = matched_frac_run_mod
                    matched_summary_stats['frac_run_vis_mod'] = matched_frac_run_vis_mod

                    unmatched_run_mod = run_mod.reset_index(drop=True).drop(index=match_idxs)
                    unmatched_run_mod.to_hdf(out_path, f'{id_flag}/unmatched/running_modulated_cells')
                    unmatched_frac_run_mod = unmatched_run_mod['sig_run_modulated'].sum() / unmatched_run_mod['sig_run_modulated'].count()
                    unmatched_frac_run_vis_mod = unmatched_run_mod['sig_vis_run_modulated'].sum() / unmatched_run_mod[
                        'sig_vis_run_modulated'].count()
                    unmatched_summary_stats['frac_run_mod'] = unmatched_frac_run_mod
                    unmatched_summary_stats['frac_run_vis_mod'] = unmatched_frac_run_vis_mod

                    multimodal_tuning = pd.concat([multimodal_tuning, run_mod], axis=1)

                    # Run the kinematic features
                    for feature in kine_features:
                        # Save the whole dataset
                        data[feature].to_hdf(out_path, f'{id_flag}/all_cells/{feature}')
                        frac_kine_resp, cells_kine_resp = kine_fraction_tuned(data[feature])
                        all_cells_summary_stats[f"frac_resp_{feature}"] = frac_kine_resp
                        multimodal_tuning[f'is_resp_{feature}'] = cells_kine_resp.values
                        multimodal_tuning[f'{feature}_resp_test'] = data[feature]['Resp_test'].values
                        multimodal_tuning[f'{feature}_resp_index'] = data[feature]['Resp_index'].values
                        multimodal_tuning[f'{feature}_qual_test'] = data[feature]['Qual_test'].values
                        multimodal_tuning[f'{feature}_qual_index'] = data[feature]['Qual_index'].values
                        multimodal_tuning[f'{feature}_cons_test'] = data[feature]['Cons_test'].values
                        multimodal_tuning[f'{feature}_cons_index'] = data[feature]['Cons_index'].values

                        # Save matched TCs
                        matched_feature = data[feature].iloc[match_idxs, :].reset_index(names=['original_cell_id'])
                        matched_feature.to_hdf(out_path, f'{id_flag}/matched/{feature}')
                        matched_frac_kine_resp, _ = kine_fraction_tuned(matched_feature)
                        matched_summary_stats[f"frac_resp_{feature}"] = matched_frac_kine_resp

                        # save unmatched tcs
                        unmatched_feature = data[feature].drop(match_idxs, axis=0)
                        unmatched_feature.to_hdf(out_path, f'{id_flag}/unmatched/{feature}')
                        unmatched_frac_kine_resp, _ = kine_fraction_tuned(unmatched_feature)
                        unmatched_summary_stats[f"frac_resp_{feature}"] = unmatched_frac_kine_resp

                    # Run the visual features
                    exp_vis_features = {}
                    for feature in vis_features:
                        feat = '_'.join(feature.split('_')[3:-1])

                        # When looking at all the data, the feature name is empty, so give it a name
                        if feat == "":
                            feat = "all"

                        feat = 'vis_' + feat

                        # Save the whole dataset
                        resp_df, frac_resp = vis_frac_responsive(data[feature])

                        if 'still' in feature:
                            new_cols = [f'{col}_still' for col in frac_resp.columns]
                            frac_resp.columns = new_cols
                        all_cells_summary_stats = pd.concat([all_cells_summary_stats, frac_resp], axis=1)

                        data[feature].reset_index(names=['original_cell_id']).to_hdf(out_path, f'{id_flag}/all_cells/{feature}')

                        # Add to multimodal tuning dataframe
                        temp_df = data[feature][['vis_resp_pval', 'fit_dsi', 'fit_osi',
                                                 'pref_dir', 'pref_ori', 'real_pref_dir', 'real_pref_ori']]
                        if 'still' in feature:
                            new_cols = [f'{col}_still' for col in temp_df.columns]
                            temp_df.columns = new_cols
                            new_cols = [f'{col}_still' for col in resp_df.columns]
                            resp_df.columns = new_cols

                        multimodal_tuning = pd.concat([multimodal_tuning, temp_df], axis=1)
                        multimodal_tuning = pd.concat([multimodal_tuning, resp_df], axis=1)

                        # Save matched TCs
                        matched_feature = data[feature].iloc[match_idxs, :]
                        matched_feature.to_hdf(out_path, f'{id_flag}/matched/{feature}')
                        matched_resp_df, matched_frac_resp = vis_frac_responsive(matched_feature)

                        if 'still' in feature:
                            new_cols = [f'{col}_still' for col in matched_frac_resp.columns]
                            matched_frac_resp.columns = new_cols

                        matched_summary_stats = pd.concat([matched_summary_stats, matched_frac_resp], axis=1)

                        # Get the preferred orientation/direction from the matched cells
                        exp_vis_features[feat] = \
                            matched_feature.loc[:,
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

                        # save unmatched TCs
                        unmatched_feature = data[feature].drop(data[feature].index[match_idxs], axis=0)
                        unmatched_feature.to_hdf(out_path, f'{id_flag}/unmatched/{feature}')
                        unmatched_resp_df, unmatched_frac_resp = vis_frac_responsive(unmatched_feature)

                        if 'still' in feature:
                            new_cols = [f'{col}_still' for col in unmatched_frac_resp.columns]
                            unmatched_frac_resp.columns = new_cols

                        unmatched_summary_stats = pd.concat([unmatched_summary_stats, unmatched_frac_resp], axis=1)

                    summary_stats = pd.concat([all_cells_summary_stats, matched_summary_stats, unmatched_summary_stats],
                                              axis=0).reset_index(drop=True)
                    summary_stats['new_idx'] = ['all_cells', 'matched', 'unmatched']
                    summary_stats.set_index('new_idx', inplace=True)
                    summary_stats.to_hdf(out_path, f'{id_flag}/summary_stats')

                    visual_shifts[id_flag] = exp_vis_features

                    multimodal_tuning.to_hdf(out_path, f'{id_flag}/multimodal_tuned')

                # Calculate delta prefs for all matched cells
                delta_pref = pd.DataFrame()
                if result in ['multi', 'control', 'fullfield']:
                    # use the fixed rig as the reference
                    for tuning_type in visual_shifts[id_flags[0]].keys():
                        # Recall that id_flags[0] is the free session, id_flags[1] is the fixed session
                        free = visual_shifts[id_flags[0]][tuning_type].reset_index()
                        fixed = visual_shifts[id_flags[1]][tuning_type].reset_index()
                        diff_dir = free.loc[:, 'pref_dir'].subtract(fixed.loc[:, 'pref_dir']).to_numpy().flatten()
                        diff_ori = free.loc[:, 'pref_ori'].subtract(fixed.loc[:, 'pref_ori']).to_numpy().flatten()

                        diff_dir = wrap_negative(diff_dir)
                        diff_ori = wrap(diff_ori, bound=180.1)

                        delta_pref[f"delta_{tuning_type}_dir"] = diff_dir
                        delta_pref[f"delta_{tuning_type}_ori"] = diff_ori

                    # Now do the same, but with the still fixed session as the reference
                    still_tunings = [el for el in visual_shifts[id_flags[0]].keys() if 'still' in el]
                    all_tunings = [el for el in visual_shifts[id_flags[0]].keys() if 'still' not in el]
                    for still_tuning, moving_tuning in zip(still_tunings, all_tunings):
                        free_still = visual_shifts[id_flags[0]][still_tuning].reset_index()
                        fixed_still = visual_shifts[id_flags[1]][still_tuning].reset_index()

                        free_all = visual_shifts[id_flags[0]][moving_tuning].reset_index()
                        fixed_all = visual_shifts[id_flags[1]][moving_tuning].reset_index()

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

                        diff_rig_ori = wrap(diff_rig_ori, bound=180.1)
                        diff_moving_fixed_ori = wrap(diff_moving_fixed_ori, bound=180.1)
                        diff_moving_free_ori = wrap(diff_moving_free_ori, bound=180.1)

                        delta_pref[f"delta_dir_{moving_tuning}_free_still_rel_fixed_still"] = diff_rig_dir
                        delta_pref[f"delta_dir_{moving_tuning}_fixed_moving_rel_fixed_still"] = diff_moving_fixed_dir
                        delta_pref[f"delta_dir_{moving_tuning}_free_moving_rel_fixed_still"] = diff_moving_free_dir

                        delta_pref[f"delta_ori_{moving_tuning}_free_still_rel_fixed_still"] = diff_rig_ori
                        delta_pref[f"delta_ori_{moving_tuning}_fixed_moving_rel_fixed_still"] = diff_moving_fixed_ori
                        delta_pref[f"delta_ori_{moving_tuning}_free_moving_rel_fixed_still"] = diff_moving_free_ori

                # If we have repeat sessions, we can't try fixed as a reference, so just look at delta between sessions
                else:
                    for tuning_type in visual_shifts[id_flags[0]].keys():
                        session1 = visual_shifts[id_flags[0]][tuning_type].reset_index()
                        session2 = visual_shifts[id_flags[1]][tuning_type].reset_index()
                        diff_dir = session2.loc[:, ['pref_dir']].subtract(session1.loc[:, ['pref_dir']]).to_numpy().flatten()
                        diff_ori = session2.loc[:, ['pref_ori']].subtract(session1.loc[:, ['pref_ori']]).to_numpy().flatten()

                        diff_dir = wrap_negative(diff_dir)
                        diff_ori = wrap(diff_ori, bound=180.1)

                        delta_pref[f"delta_{tuning_type}_dir"] = diff_dir
                        delta_pref[f"delta_{tuning_type}_ori"] = diff_ori

                delta_pref['match_idxs'] = match_idxs
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

        # If there was a problem, we then delete the output file
        except Exception as e:
            os.remove(out_path)
            print(f'The following exception occurred: {e}. Deleting the output file {out_path}')

        # Write to the dummy output file
        with open(dummy_out, 'w') as f:
            f.writelines(input_paths)
