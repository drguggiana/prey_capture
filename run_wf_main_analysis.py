#!/usr/bin/env python
# coding: utf-8

import itertools
import os
import warnings
from typing import List

import h5py
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import sklearn.preprocessing as preproc
from tqdm import tqdm
from umap.umap_ import UMAP

import functions_bondjango as bd
import functions_data_handling as fdh
import functions_kinematic as fk
import functions_misc as misc
import functions_plotting as fp
import functions_tuning as ftuning
import paths
import processing_parameters

warnings.filterwarnings('ignore')


def make_aggregate_file(search_string):
    search_string = search_string.replace('agg_all', 'agg_tc')
    parsed_search = fdh.parse_search_string(search_string)
    output_path = os.path.join(paths.analysis_path, f"AGG_{'_'.join(parsed_search.values())}.hdf5")

    # get the paths from the database
    file_infos = bd.query_database("analyzed_data", search_string)
    input_paths = np.array([el['analysis_path'] for el in file_infos if ('agg' in el['slug'])])

    data_list = []
    for file in tqdm(input_paths, desc="Loading files"):
        print(file)
        data_dict = {}
        mouse = '_'.join(os.path.basename(file).split('_')[10:13])

        with pd.HDFStore(file, 'r') as tc:

            if 'no_ROIs' in tc.keys():
                continue

            else:

                for key in tc.keys():
                    label = "_".join(key.split('/')[1:])
                    data = tc[key]
                    if 'animal' in data.columns:
                        data = data.drop(columns='animal')

                    if '08_31_2023_VWheelWF_fixed2' in data.columns:
                        data.drop(columns='08_31_2023_VWheelWF_fixed2', inplace=True)

                    data['mouse'] = mouse
                    data_dict[label] = data

                data_list.append(data_dict)

    if len(data_list) == 0:
        print('No data to aggregate')
        return None
    else:
        # Aggregate it all
        agg_dict = {}

        for key in data_list[0].keys():
            df = pd.concat([d[key] for d in data_list]).reset_index(drop=True)
            df.to_hdf(output_path, key)
            agg_dict[key] = df

        # assemble the entry data
        entry_data = {
            'analysis_type': 'agg_all',
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

        print('The output status was %i, reason %s' % (output_entry.status_code, output_entry.reason))
        if output_entry.status_code in [500, 400]:
            print(entry_data)

        return agg_dict


def check_for_fit_si(data_dict, activity_ds):
    update_keys = [key for key in data_dict.keys() if activity_ds in key]

    for key in update_keys:

        if 'fit_dsi' not in data_dict[key].columns:
            print('calculating DSI and OSI from fit...')

            fit_dsi_list = []
            fit_osi_list = []

            for i, row in data_dict[key].iterrows():
                tcs = row['mean_norm_dir']
                angles = tcs[:, 0]
                magnitudes = tcs[:, 1]
                pref = row['real_pref_dir']
                fit_dsi, fit_osi = ftuning.calculate_dsi_osi_fit(angles, magnitudes, pref)
                fit_dsi_list.append(fit_dsi)
                fit_osi_list.append(fit_osi)

            data_dict[key]['fit_dsi'] = fit_dsi_list
            data_dict[key]['fit_osi'] = fit_osi_list

        else:
            pass

    return data_dict


def get_vis_tuned_cells(ds: pd.DataFrame, vis_stim: str = 'dir', sel_thresh: float = 0.5) -> pd.DataFrame:
    """
    Filter the input DataFrame to select cells that are visually tuned based on the specified visual stimulus.

    Parameters:
        ds (pd.DataFrame): The input DataFrame containing the data.
        vis_stim (str): The visual stimulus type. Valid values are 'dir', 'ori', 'vis', or 'untuned'.
        sel_thresh (float): The threshold for selectivity. Cells with absolute selectivity values below this
                            threshold will be excluded.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the visually tuned cells.

    Raises:
        Exception: If an invalid vis_stim value is provided.

    """

    data = ds.copy()

    if vis_stim == 'vis':
        cells = data[data['is_vis_resp']]
        return cells

    elif vis_stim == 'dir':
        cells = data[(data['is_dir_responsive'] == 1)
                     & (data['fit_dsi'] >= sel_thresh)
                     & (data['is_gen_responsive'] == 1)
                     ]
        return cells

    elif vis_stim == 'ori-gen':
        cells = data[(data['is_ori_responsive'] == 1)
                     & (data['fit_osi'] >= sel_thresh)
                     & (data['is_gen_responsive'] == 0)
                     ]
        return cells

    elif vis_stim == 'ori':
        cells = data[(data['is_ori_responsive'] == 1)
                     & (data['fit_osi'] >= sel_thresh)
                     & (data['is_gen_responsive'] == 1)
                     ]
        return cells

    else:
        return Exception('Invalid vis_stim')


def find_overlap_tuning(dir_tuned, ori_tuned):
    # Find cells that are both direction and orientation tuned, and figure out what to do with them.
    _, comm1, comm2 = np.intersect1d(dir_tuned.index, ori_tuned.index, return_indices=True)
    both_tuned = dir_tuned.iloc[comm1].copy()

    # Remove cells tuned to both from each category
    dir_tuned = dir_tuned.drop(dir_tuned.index[comm1])
    ori_tuned = ori_tuned.drop(ori_tuned.index[comm2])

    return dir_tuned, ori_tuned, both_tuned


def update_vis_tuning(old_vis_resp, full_ds, dir_tuned, ori_tuned, both_tuned):
    resp_cells = np.unique(np.concatenate([dir_tuned.index, ori_tuned.index, both_tuned.index]))
    not_in_resp_cells = np.setdiff1d(resp_cells, old_vis_resp.index, assume_unique=False)
    new_vis_resp = pd.concat([old_vis_resp, full_ds.iloc[not_in_resp_cells, :]])
    new_vis_resp = new_vis_resp.reset_index().drop_duplicates(subset=['index'])
    return new_vis_resp


def generate_count_tuned_df(gen_resp, vis_resp, vis_no_gen_resp, ori_tuned, ori_no_gen_tuned, dir_tuned, dir_no_gen_tuned):
    count_gen_resp = gen_resp.groupby(['mouse', 'day']).old_index.size()
    count_gen_resp = count_gen_resp.reset_index().rename(columns={'old_index': 'general'})

    count_vis_resp = vis_resp.groupby(['mouse', 'day']).old_index.size()
    count_vis_resp = count_vis_resp.reset_index().rename(columns={'old_index': 'visual'})

    count_vis_no_gen_resp = vis_no_gen_resp.groupby(['mouse', 'day']).old_index.size()
    count_vis_no_gen_resp = count_vis_no_gen_resp.reset_index().rename(columns={'old_index': 'visual_no_gen'})

    count_ori_tuned = ori_tuned.groupby(['mouse', 'day']).old_index.size()
    count_ori_tuned = count_ori_tuned.reset_index().rename(columns={'old_index': 'orientation'})

    count_ori_no_gen_tuned = ori_no_gen_tuned.groupby(['mouse', 'day']).old_index.size()
    count_ori_no_gen_tuned = count_ori_no_gen_tuned.reset_index().rename(columns={'old_index': 'orientation_no_gen'})

    count_dir_tuned = dir_tuned.groupby(['mouse', 'day']).old_index.size()
    count_dir_tuned = count_dir_tuned.reset_index().rename(columns={'old_index': 'direction'})

    count_dir_no_gen_tuned = dir_no_gen_tuned.groupby(['mouse', 'day']).old_index.size()
    count_dir_no_gen_tuned = count_dir_no_gen_tuned.reset_index().rename(columns={'old_index': 'direction_no_gen'})

    count_resp = pd.concat([count_gen_resp,
                                 count_vis_resp.drop(['mouse', 'day'], axis=1),
                                 count_vis_no_gen_resp.drop(['mouse', 'day'], axis=1),
                                 count_ori_tuned.drop(['mouse', 'day'], axis=1),
                                 count_ori_no_gen_tuned.drop(['mouse', 'day'], axis=1),
                                 count_dir_tuned.drop(['mouse', 'day'], axis=1),
                                 count_dir_no_gen_tuned.drop(['mouse', 'day'], axis=1),
                                ], axis=1)

    return count_resp


def generate_frac_tuned_df(cell_count, gen_resp, vis_resp, vis_no_gen_resp, ori_tuned, ori_no_gen_tuned, dir_tuned,
                           dir_no_gen_tuned):

    frac_gen_resp = gen_resp.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_gen_resp = frac_gen_resp.reset_index().rename(columns={0: 'general'})

    frac_vis_resp = vis_resp.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_vis_resp = frac_vis_resp.reset_index().rename(columns={0: 'visual'})

    frac_vis_no_gen_resp = vis_no_gen_resp.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_vis_no_gen_resp = frac_vis_no_gen_resp.reset_index().rename(columns={0: 'visual_no_gen'})

    frac_ori_tuned = ori_tuned.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_ori_tuned = frac_ori_tuned.reset_index().rename(columns={0: 'orientation'})

    frac_ori_no_gen_tuned = ori_no_gen_tuned.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_ori_no_gen_tuned = frac_ori_no_gen_tuned.reset_index().rename(columns={0: 'orientation_no_gen'})

    frac_dir_tuned = dir_tuned.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_dir_tuned = frac_dir_tuned.reset_index().rename(columns={0: 'direction'})

    frac_dir_no_gen_tuned = dir_no_gen_tuned.groupby(['mouse', 'day']).old_index.size() / cell_count
    frac_dir_no_gen_tuned = frac_dir_no_gen_tuned.reset_index().rename(columns={0: 'direction_no_gen'})

    frac_resp = pd.concat([frac_gen_resp,
                                frac_vis_resp.drop(['mouse', 'day'], axis=1),
                                frac_vis_no_gen_resp.drop(['mouse', 'day'], axis=1),
                                frac_ori_tuned.drop(['mouse', 'day'], axis=1),
                                frac_ori_no_gen_tuned.drop(['mouse', 'day'], axis=1),
                                frac_dir_tuned.drop(['mouse', 'day'], axis=1),
                                frac_dir_no_gen_tuned.drop(['mouse', 'day'], axis=1)
                                ], axis=1)

    frac_resp.fillna(0, inplace=True)

    return frac_resp


def kine_fraction_tuned(ds: pd.DataFrame, use_test: bool = True, include_responsivity: bool = True,
                        include_consistency: bool = False) -> float:
    """
    Calculate the fraction of tuned cells from the given dataset.

    Parameters:
    - ds (pd.DataFrame): The dataset containing the required columns.
    - use_test (bool): Flag indicating whether to use the test columns or index columns. Default is True.
    - include_responsivity (bool): Flag indicating whether to include responsivity in the calculation. Default is True.
    - include_consistency (bool): Flag indicating whether to include consistency in the calculation. Default is False.

    Returns:
    - float: The fraction of tuned values.

    """

    if use_test:
        resp = ds['Resp_test']
        qual = ds['Qual_test']
        cons = ds['Cons_test']

    else:
        resp = ds['Resp_index'] >= processing_parameters.tc_resp_qual_cutoff/100
        qual = ds['Qual_index'] >= processing_parameters.tc_resp_qual_cutoff/100
        cons = ds['Cons_index'] >= processing_parameters.tc_consistency_cutoff/100

    # is tuned if quality is true
    is_tuned = qual

    if include_responsivity:
        is_tuned = is_tuned + resp
        is_tuned = is_tuned > 1

    if include_consistency:
        is_tuned = is_tuned + cons
        is_tuned = is_tuned > 1

    frac_is_tuned = is_tuned.sum() / is_tuned.count()
    return frac_is_tuned


def get_sig_tuned_kinem_cells(agg_dict: dict, exp_kind: str, which_cells: str, vars: list, use_test: bool = True, include_responsivity: bool = True, include_consistency: bool = False) -> pd.DataFrame:
    """
    Get significant tuned kinematic cells based on the provided parameters.

    Args:
        agg_dict (dict): A dictionary containing aggregated data.
        exp_kind (str): The kind of experiment.
        which_cells (str): The type of cells to consider.
        vars (list): A list of variables to index the input DataFrames.
        use_test (bool, optional): Whether to use bootstrapping test results. Defaults to True.
        include_responsivity (bool, optional): Whether to use responsivity. Defaults to True.
        include_consistency (bool, optional): Whether to use consistency. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing the significant tuned kinematic cells.
    """

    keys = ['_'.join([exp_kind, which_cells, var]) for var in vars]
    df = pd.DataFrame(columns=vars)

    for key, var in zip(keys, vars):
        if key in agg_dict.keys():
            df[var] = agg_dict[key].groupby(['mouse', 'day']).apply(kine_fraction_tuned, use_test, include_responsivity, include_consistency)

    return df


def find_tuned_cell_indices(ref_data: pd.DataFrame, comp_data: pd.DataFrame, stim_kind: str = 'orientation',
                            cutoff: float = 0.5, tuning_criteria: str = 'both') -> np.ndarray:
    """
    This function finds the indices of cells that are tuned in either the reference, comparison, or both arrays.

    Parameters:
    ref_data (pd.DataFrame): The reference DataFrame containing the data.
    comp_data (pd.DataFrame): The comparison DataFrame containing the data.
    stim_kind (str, optional): The kind of stimulus. Valid values are 'orientation' and 'direction'. Defaults to 'orientation'.
    cutoff (float, optional): The threshold for tuning [0-1]. Cells with absolute tuning values below this threshold
        will be excluded. Defaults to 0.5.
    tuning_criteria (str, optional): The criteria for tuning. Valid values are 'both', 'ref', and 'comp'.
        'both' returns the intersection of 'ref' and 'comp', 'ref' returns indices from the reference data, and
        'comp' returns indices from the comparison data. Defaults to 'both'.

    Returns:
    numpy.ndarray: The indices of the tuned cells.

    Raises:
    Exception: If an invalid stim_kind or tuning_criteria value is provided.
    """

    # Find the cells in the reference dataset that are tuned
    if stim_kind == 'orientation':
        index_column = 'fit_osi'
    elif stim_kind == 'direction':
        index_column = 'fit_dsi'
    else:
        raise Exception('Invalid stim_kind')

    index_ref = ref_data.loc[ref_data[index_column].abs() >= cutoff].index.to_numpy()
    index_comp = comp_data.loc[comp_data[index_column].abs() >= cutoff].index.to_numpy()

    # Determine the indices to return based on the tuning criteria
    if tuning_criteria == 'both':
        indices = np.intersect1d(index_ref, index_comp)
    elif tuning_criteria == 'ref':
        indices = index_ref
    elif tuning_criteria == 'comp':
        indices = index_comp
    else:
        raise Exception('Invalid tuning_criteria')

    return indices


def pref_angle_shifts(ds1: pd.DataFrame, ds2: pd.DataFrame, ci_width_cutoff: int = 20, stim_kind: str = 'orientation',
                  method: str = 'fit') -> pd.DataFrame:
    """
    This function calculates the shifts in tuning for a given stimulus kind.

    Parameters:
    ds1 (pd.DataFrame): The first DataFrame containing the data.
    ds2 (pd.DataFrame): The second DataFrame containing the data.
    ci_width_cutoff (int, optional): The cutoff for the confidence interval width. Defaults to 20.
    stim_kind (str, optional): The kind of stimulus. Valid values are 'orientation' and 'direction'. Defaults to 'orientation'.
    method (str, optional): The method to use. Valid values are 'fit' and 'resultant'. Defaults to 'fit'.

    Returns:
    pd.DataFrame: A DataFrame containing the shifts in tuning.

    Raises:
    Exception: If an invalid method value is provided.
    """

    stim_kind = stim_kind[:3]
    if stim_kind == 'ori':
        multiplier = 2
    elif stim_kind == 'dir':
        multiplier = 1
    else:
        raise Exception('Invalid stim_kind')

    if method == 'fit':
        dist_key = f'bootstrap_pref_{stim_kind}'
        pref_key = f'pref_{stim_kind}'
    elif method == 'resultant':
        dist_key = f'bootstrap_resultant_{stim_kind}'
        pref_key = f'resultant_{stim_kind}'
    else:
        raise Exception('Invalid method')

    shifts = []

    for (idxRow, cell_1), (_, cell_2) in zip(ds1.iterrows(), ds2.iterrows()):

        # Get preferred angle
        pref_1 = cell_1[pref_key]
        pref_2 = cell_2[pref_key]

        # Get bootstrap distributions
        pref_dist_1 = cell_1[dist_key]
        pref_dist_2 = cell_2[dist_key]

        if method == 'resultant':
            pref_1 = pref_1[-1]
            pref_2 = pref_2[-1]
            pref_dist_1 = pref_dist_1[:, -1]
            pref_dist_2 = pref_dist_2[:, -1]

            if np.isnan(pref_1) or np.isnan(pref_2):
                pass

        pref_dist_1 = pref_dist_1[~np.isnan(pref_dist_1)]
        pref_dist_2 = pref_dist_2[~np.isnan(pref_dist_2)]

        if pref_dist_1.size == 0 or pref_dist_2.size == 0:
            pass

        # Wrap angles
        pref_dist_1 = fk.wrap(pref_dist_1, bound=(360. / multiplier) + 0.01)
        pref_dist_2 = fk.wrap(pref_dist_2, bound=(360. / multiplier) + 0.01)
        delta_pref = np.abs(pref_2 - pref_1)

        # Calculate confidence intervals
        ci_1 = st.norm.interval(confidence=0.95, loc=np.nanmean(pref_dist_1),
                                scale=st.sem(pref_dist_1, nan_policy='omit'))
        ci_2 = st.norm.interval(confidence=0.95, loc=np.nanmean(pref_dist_2),
                                scale=st.sem(pref_dist_2, nan_policy='omit'))

        # Get CI widths
        ci_width_1 = np.abs(ci_1[-1] - ci_1[0])
        ci_width_2 = np.abs(ci_2[-1] - ci_2[0])

        # Check if tuned or not
        if (ci_width_1 <= ci_width_cutoff) and (ci_width_2 <= ci_width_cutoff):

            # determine significance of shift
            if ~(ci_1[0] <= pref_2 <= ci_1[-1]) and ~(ci_2[0] <= pref_1 <= ci_2[-1]):
                # Shift is significant
                sig_shift = 1
            else:
                # Shift is not significant
                sig_shift = 0
        else:
            # The cell is not tuned
            sig_shift = 0

        shifts.append([idxRow, pref_1, ci_width_1, pref_2, ci_width_2, delta_pref, sig_shift, cell_1.mouse, cell_1.day])

    shifts = pd.DataFrame(data=shifts,
                          columns=['', 'pref_1', 'ci_width_1', 'pref_2', 'ci_width_2', 'delta_pref', 'is_sig', 'mouse',
                                   'date'])
    shifts = shifts.set_index(shifts.columns[0])

    return shifts


def calculate_pref_angle_shifts(ref_ds: pd.DataFrame, comp_ds: pd.DataFrame, stim_kind: str = 'orientation',
                                upper_cutoff: float = 0.5, lower_cutoff: float = 0.3) -> tuple:
    """
    This function calculates the shifts in preferred orientation or direction.

    Parameters:
    ref_ds (pd.DataFrame): The reference DataFrame containing the data.
    comp_ds (pd.DataFrame): The comparison DataFrame containing the data.
    stim_kind (str, optional): The kind of stimulus. Valid values are 'orientation' and 'direction'. Defaults to 'orientation'.
    upper_cutoff (float, optional): The upper cutoff for selectivity indices. Defaults to 0.5.
    lower_cutoff (float, optional): The lower cutoff for selectivity indices. Defaults to 0.3.

    Returns:
    tuple: A tuple containing the shifts, residuals and the root mean square error of the residual orientation.
    """

    # Determine the width of the confidence interval based on the stimulus kind
    if stim_kind == 'orientation':
        ci_width = 45
        pref_col = 'pref_ori'
    elif stim_kind == 'direction':
        ci_width = 90
        pref_col = 'pref_dir'
    else:
        raise Exception('Invalid stim_kind')

    # Find the indices of cells that are tuned
    indices = find_tuned_cell_indices(ref_ds.copy(), comp_ds.copy(), cutoff=upper_cutoff,
                                      stim_kind=stim_kind, tuning_criteria='both')

    # Calculate the shifts in tuning
    diff = np.abs(comp_ds.loc[indices, pref_col].values - ref_ds.loc[indices, pref_col].values)
    shifts = pd.DataFrame(data={'pref_1': ref_ds.loc[indices, pref_col].values,
                                'pref_2': comp_ds.loc[indices, pref_col].values,
                                'delta_pref': diff,
                                'mouse': ref_ds.loc[indices, 'mouse'].values,
                                'day': ref_ds.loc[indices, 'day'].values})

    # Calculate the residuals and the root mean square error of the residuals
    residuals = np.linalg.norm(shifts[['pref_1', 'pref_2']].values - shifts[['pref_1', 'pref_1']].values, axis=1)

    # This is a circular measure, so wrap the residuals
    residuals = fk.wrap(residuals, bound=ci_width)
    rmse_residual = np.sqrt(np.mean(residuals ** 2))

    return shifts, residuals, rmse_residual


def calculate_delta_selectivity(ref_ds: pd.DataFrame, comp_ds: pd.DataFrame, cutoff: float = 0.5,
                                stim_kind: str = 'orientation') -> tuple:
    """
        This function calculates the shifts in preferred orientation or direction.

        Parameters:
        ref_ds (pd.DataFrame): The reference DataFrame containing the data.
        comp_ds (pd.DataFrame): The comparison DataFrame containing the data.
        stim_kind (str, optional): The kind of stimulus. Valid values are 'orientation' and 'direction'. Defaults to 'orientation'.

        Returns:
        tuple: A tuple containing the shifts, residuals and the root mean square error of the residuals.
        """

    if stim_kind == 'orientation':
        sel_key = 'fit_osi'
    elif stim_kind == 'direction':
        sel_key = 'fit_dsi'
    else:
        raise Exception('Invalid stim_kind')

    # Find the indices of cells that are tuned
    indices = find_tuned_cell_indices(ref_ds.copy(), comp_ds.copy(), cutoff=cutoff,
                                      stim_kind=stim_kind, tuning_criteria='both')

    # Calculate the shifts in selectivity
    ref_sel = ref_ds.iloc[indices, ref_ds.columns.get_loc(sel_key)].copy().clip(-1, 1)
    comp_sel = comp_ds.iloc[indices, comp_ds.columns.get_loc(sel_key)].copy().clip(-1, 1)
    residuals = np.linalg.norm(np.array([ref_sel, comp_sel]).T - np.array([ref_sel, ref_sel]).T, axis=1)
    shifts = pd.DataFrame(data={'pref_1': ref_sel, 'pref_2': comp_sel, 'delta_sel': residuals})

    # Calculate the residuals and the root mean square error of the residuals
    rmse_residual = np.sqrt(np.mean(residuals ** 2))

    return shifts, residuals, rmse_residual


def plot_delta_pref(shifts: pd.DataFrame, session_types: List[str],
                    type: str = 'orientation', wrap_neg: bool = True) -> hv.Scatter:
    """
    This function plots the shift in preferred orientation or direction.

    Parameters:
    shifts (pd.DataFrame): A DataFrame containing the shifts in preferred angle.
    session_types (list): A list containing session types.
    type (str): The type of stimulus. Valid values are 'orientation' and 'direction'. Defaults to 'orientation'.
    wrap_neg (bool): Whether to wrap the negative domain. Defaults to True.

    Returns:
    hv.Scatter: A Scatter plot showing the shift in preference.
    """

    if type == 'orientation':
        if wrap_neg:
            upper_lim = 90
            lower_lim = -upper_lim
        else:
            upper_lim = 180
            lower_lim = 0
    elif type == 'direction':
        if wrap_neg:
            upper_lim = 180
            lower_lim = -upper_lim
        else:
            upper_lim = 360
            lower_lim = 0
    else:
        raise Exception('Invalid type')

    unity_line = hv.Curve((np.arange(lower_lim, upper_lim, 1), np.arange(lower_lim, upper_lim, 1))).opts(color='black')

    if wrap_neg:
        prefs = fk.wrap_negative(shifts[['pref_1', 'pref_2']].to_numpy(), bound=upper_lim)
    else:
        prefs = shifts[['pref_1', 'pref_2']].to_numpy()

    scatter_delta = hv.Scatter(prefs, kdims=['pref_1'], vdims=['pref_2'], label='sig').opts(
        color='purple', size=8)

    scatter = unity_line * scatter_delta

    scatter.opts(show_legend=False, width=500, height=500)
    
    scatter.opts(
        hv.opts.Scatter(xlim=(lower_lim, upper_lim), ylim=(lower_lim, upper_lim),
                        xticks=np.linspace(lower_lim, upper_lim, 5),
                        yticks=np.linspace(lower_lim, upper_lim, 5)
                        )
    )

    return scatter


def create_umap_plot(plot_data: np.ndarray, predictor_column: str) -> hv.Scatter:
    """
    This function creates a UMAP plot using holoviews.

    Parameters:
    plot_data (np.ndarray): The data to be plotted.
    predictor_columns (List[str]): The list of predictor columns.
    predictor_column (str): The current predictor column.

    Returns:
    hv.Scatter: A Scatter plot showing the UMAP.
    """
    umap_plot = hv.Scatter(plot_data, vdims=['Dim 2', 'Parameter'], kdims=['Dim 1'])
    umap_plot.opts(colorbar=False, color='Parameter', cmap='Spectral_r', alpha=1, xaxis=None,
                   yaxis=None, tools=['hover'])

    if any([predictor_column.startswith('dsi'), predictor_column.startswith('osi')]):
        umap_plot.opts(title=f"{predictor_column[:3].upper()}")
    else:
        umap_plot.opts(title=processing_parameters.wf_label_dictionary_wo_units[predictor_column])

    umap_plot.opts(width=300, height=300, size=2)

    return umap_plot


# set up the figure theme and saving paths
fp.set_theme()
in2cm = 1./2.54

# define the experimental conditions
results = ['multi', 'repeat']    # 'multi', 'control', 'repeat', 'fullfield'
lightings = ['normal']    # 'normal', 'dark'
rigs = ['', 'VWheelWF', 'VTuningWF']    # '', 'VWheelWF', 'VTuningWF'
analysis_type = 'agg_all'

recalculate_vis_tuning = False

save_base = r'D:\thesis\WF_Figures'       # r'H:\thesis\figures\WF_Figures', r"Z:\Prey_capture\WF_Figures"

for result, light, rig in itertools.product(results, lightings, rigs):

    # Filter out searches that don't make sense
    if (result == 'repeat') and (rig == ''):
        continue
    elif (result != 'repeat') and (rig != ''):
        continue
    elif (result == 'fullfield') and (light == 'dark'):
        continue
    else:
        # get the search string
        search_string = f"result:{result}, lighting:{light}, rig:{rig}, analysis_type:{analysis_type}"

    # Load the data and set up figure saving path
    parsed_search = fdh.parse_search_string(search_string)
    file_infos = bd.query_database("analyzed_data", search_string)
    input_paths = [el['analysis_path'] for el in file_infos]

    if len(input_paths) == 0:
        # Need to create the file
        data_dict = make_aggregate_file(search_string)
        if data_dict is None:
            continue
    else:
        input_path = input_paths[0]

        # If the file exists, just load it
        if os.path.isfile(input_path):

            data_dict = {}
            with pd.HDFStore(input_path, 'r') as tc:
                for key in tc.keys():
                    label = "_".join(key.split('/')[1:])
                    data = tc[key]
                    data_dict[label] = data
            del data
        # Otherwise the entry exists but the file doesn't. Create the file
        else:
            data_dict = make_aggregate_file(search_string)
            if data_dict is None:
                continue

    save_suffix = f"{parsed_search['result']}_{parsed_search['lighting']}_{parsed_search['rig']}"

    # Set color themes depending on experiment type
    if parsed_search['result'] == 'repeat':
        if parsed_search['rig'] in ['VWheelWF', 'VWheel']:
            session_types = ['fixed0', 'fixed1']
            scatter_color_theme = 'red'
            fixed_violin_cmap = 'red'
            free_violin_cmap = fp.hv_blue_hex

        elif parsed_search['rig'] in ['VTuningWF', 'VTuning']:
            session_types = ['free0', 'free1']
            scatter_color_theme = fp.hv_blue_hex
            fixed_violin_cmap = 'red'
            free_violin_cmap = fp.hv_blue_hex
        else:
            raise Exception('Invalid rig')

        session_shorthand = session_types
        ref_idx_order = [1, 2]

    elif parsed_search['result'] == 'control':
        session_types = ['VWheelWF', 'VTuningWF']
        session_shorthand = ['fixed', 'free']
        ref_idx_order = [2, 1]

        if parsed_search['lighting'] == 'normal':
            fixed_violin_cmap = fp.hv_yellow_hex
            free_violin_cmap = fp.hv_yellow_hex
            scatter_color_theme = fp.hv_yellow_hex
        elif parsed_search['lighting'] == 'dark':
            fixed_violin_cmap = fp.hv_gray_hex
            free_violin_cmap = fp.hv_gray_hex
            scatter_color_theme = fp.hv_gray_hex
        else:
            raise Exception('Invalid lighting condition')

    else:
        session_types = ['VWheelWF', 'VTuningWF']
        session_shorthand = ['fixed', 'free']
        fixed_violin_cmap = 'red'
        free_violin_cmap = fp.hv_blue_hex
        scatter_color_theme = 'purple'
        ref_idx_order = [1, 2]

    # Specify the path to the curated cell matches file
    curated_cell_matches_path = os.path.join(r"C:\Users\mmccann\Desktop", 
                                f"curated_cell_matches_{parsed_search['result']}_{parsed_search['lighting']}_{parsed_search['rig']}.xlsx")

    try:
        # Read all sheets into a dict of dataframes
        curated_matches_dict = pd.read_excel(curated_cell_matches_path, sheet_name=None)

        # Concatenate the dataframes into a single dataframe
        curated_matches = pd.concat(curated_matches_dict.values(), ignore_index=True)

        # rename columns
        cols = curated_matches.columns.to_list()
        cols[0] = 'old_index'
        curated_matches.columns = cols

    except Exception as e:
        print(f"Could not find the file {curated_cell_matches_path}. Continuing with CaImAn matches...")
        curated_matches = None

    # Load the correct dataset
    cell_kind = 'all_cells'

    for activity_ds in processing_parameters.activity_datasets:
        print(f'Working with {activity_ds} data...')
        activity_dataset = f'{activity_ds}_props'

        if 'still' in activity_dataset:
            session_shorthand = [f + '_still' for f in session_shorthand]
            activity_ds_basename = activity_ds.rsplit('_', 2)[0]
            figure_save_path = os.path.join(save_base, activity_ds_basename, 'still', save_suffix)
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)

        else:
            activity_ds_basename = activity_ds.rsplit('_', 1)[0]
            figure_save_path = os.path.join(save_base, activity_ds_basename, 'full', save_suffix)
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)

        data_save_path = os.path.join(figure_save_path, 'stats.hdf5')

        # Run a check for the fit_dsi and fit_osi columns
        data_dict = check_for_fit_si(data_dict, activity_dataset)

        # Pick out the summary stats and tuning properties
        fixed_summary_stats = (data_dict[f'{session_types[0]}_summary_stats']
                               .sort_values(['mouse', 'day']).copy().fillna(0))
        free_summary_stats = (data_dict[f'{session_types[1]}_summary_stats']
                              .sort_values(['mouse', 'day']).copy().fillna(0))

        fixed_tcs = data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'].sort_values(['mouse', 'day']).copy()
        free_tcs = data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'].sort_values(['mouse', 'day']).copy()

        fixed_tcs_by_cell = data_dict[f'{session_types[0]}_multimodal_tuned'].sort_values(['mouse', 'day']).copy()
        fixed_tcs_by_cell.old_index = fixed_tcs_by_cell.old_index.apply(lambda x: int(x.split('_')[-1]))
        free_tcs_by_cell = data_dict[f'{session_types[1]}_multimodal_tuned'].sort_values(['mouse', 'day']).copy()
        free_tcs_by_cell.old_index = free_tcs_by_cell.old_index.apply(lambda x: int(x.split('_')[-1]))

        # Create dataframes to store binary tuning information
        fixed_cell_tunings = fixed_tcs[['old_index', 'mouse', 'day']].copy()
        free_cell_tunings = free_tcs[['old_index', 'mouse', 'day']].copy()

        # --- Load all cell matches --- #
        all_cell_matches = data_dict['cell_matches']

        # Get the cell matches for the current dataset and plot a scatter plot of the fraction of cells matched
        if 'still' not in activity_dataset:
            match_frac0 = fixed_summary_stats.loc[fixed_summary_stats.old_index == 'all_cells', 'match_frac'].values
            match_frac1 = free_summary_stats.loc[free_summary_stats.old_index == 'all_cells', 'match_frac'].values

            line = hv.Curve((np.linspace(0, 1, 101), np.linspace(0, 1, 101))).opts(color='gray')
            scatter = hv.Scatter((match_frac0, match_frac1))
            scatter.opts(xlim=(0, 1.05), xlabel=f'Frac. Match {session_shorthand[0].title()}',
                         ylim=(0, 1.05), ylabel=f'Frac. Match {session_shorthand[1].title()}',
                         color=scatter_color_theme, width=500, height=500)
            frac_cell_match = hv.Overlay([line, scatter])

            save_path = os.path.join(figure_save_path, "frac_cells_matched.png")
            frac_cell_match = fp.save_figure(frac_cell_match, save_path=save_path, fig_width=8, dpi=800, fontsize='paper',
                                             target='save', display_factor=0.1)

            with pd.HDFStore(data_save_path, 'a') as store:
                if 'frac_cell_match' in store.keys():
                    del store['frac_cell_match']

                store['frac_cell_match'] = pd.DataFrame(data={f'match_frac_{session_shorthand[0]}': match_frac0,
                                                              f'match_frac_{session_shorthand[1]}': match_frac1})

        # --- Get the visual responsivity of the cells --- #
        # We can load directly from the summary stats page here, or we can recalculate
        if recalculate_vis_tuning:
            print('Recalculating visual tuning...')
            # TODO: do stuff here
        else:
            vis_resp_cols = ['is_vis_resp', 'mod_vis_resp', 'not_vis_resp',
                             'vis_resp_dir_tuned', 'vis_resp_ori_tuned',
                             'mod_resp_dir_tuned', 'mod_resp_ori_tuned',
                             'not_resp_dir_tuned', 'not_resp_ori_tuned']

            if 'still' in activity_dataset:
                vis_resp_cols = [f'{col}_still' for col in vis_resp_cols]

            # Get cell counts and rename the columns
            count_vis_resp_fixed = fixed_tcs_by_cell.groupby(['mouse', 'day'])[vis_resp_cols].sum().reset_index()
            count_vis_resp_free = free_tcs_by_cell.groupby(['mouse', 'day'])[vis_resp_cols].sum().reset_index()

            vis_resp_rename_cols = ['mouse', 'day', 'visual', 'mod_visual', 'not_visual',
                                    'direction', 'orientation', 'mod_direction', 'mod_orientation',
                                    'not_direction', 'not_orientation']
            count_vis_resp_rename_dict = dict(zip(count_vis_resp_fixed.columns.to_list(), vis_resp_rename_cols))
            count_vis_resp_fixed = count_vis_resp_fixed.rename(columns=count_vis_resp_rename_dict)
            count_vis_resp_free = count_vis_resp_free.rename(columns=count_vis_resp_rename_dict)
            vis_resp_cols += ['vis_resp_dir_tuned_within', 'vis_resp_ori_tuned_within',
                              'mod_resp_dir_tuned_within', 'mod_resp_ori_tuned_within',
                              'not_resp_dir_tuned_within', 'not_resp_ori_tuned_within']

            # Get fraction cells and rename the columns
            frac_vis_resp_cols = ['mouse', 'day'] + [f'frac_{col}' for col in vis_resp_cols]
            frac_vis_resp_fixed = fixed_summary_stats.loc[fixed_summary_stats.old_index == 'all_cells',
                                                          frac_vis_resp_cols].reset_index(drop=True)
            frac_vis_resp_free = free_summary_stats.loc[free_summary_stats.old_index == 'all_cells',
                                                        frac_vis_resp_cols].reset_index(drop=True)
            frac_vis_resp_fixed.fillna(0, inplace=True)
            frac_vis_resp_free.fillna(0, inplace=True)

            frac_vis_resp_rename_cols = vis_resp_rename_cols + [f'{col}_within' for col in vis_resp_rename_cols[5:]]
            frac_vis_resp_rename_dict = dict(zip(frac_vis_resp_cols, frac_vis_resp_rename_cols))
            frac_vis_resp_fixed = frac_vis_resp_fixed.rename(columns=frac_vis_resp_rename_dict)
            frac_vis_resp_free = frac_vis_resp_free.rename(columns=frac_vis_resp_rename_dict)

        # --- Plot count and fractions visually tuned

        # Save the counts and fractions
        with pd.HDFStore(data_save_path, 'a') as store:
            if f'count_vis_resp_{session_shorthand[0]}' in store.keys():
                del store[f'count_vis_resp_{session_shorthand[0]}']

            if f'count_vis_resp_{session_shorthand[1]}' in store.keys():
                del store[f'count_vis_resp_{session_shorthand[1]}']

            if f'frac_vis_resp_{session_shorthand[0]}' in store.keys():
                del store[f'frac_vis_resp_{session_shorthand[0]}']

            if f'frac_vis_resp_{session_shorthand[1]}' in store.keys():
                del store[f'frac_vis_resp_{session_shorthand[1]}']

            store[f'count_vis_resp_{session_shorthand[0]}'] = count_vis_resp_fixed
            store[f'count_vis_resp_{session_shorthand[1]}'] = count_vis_resp_free
            store[f'frac_vis_resp_{session_shorthand[0]}'] = frac_vis_resp_fixed
            store[f'frac_vis_resp_{session_shorthand[1]}'] = frac_vis_resp_free

        count_vis_resp_fixed = count_vis_resp_fixed.drop(['mouse', 'day'], axis=1)
        count_vis_resp_free = count_vis_resp_free.drop(['mouse', 'day'], axis=1)
        frac_vis_resp_fixed = frac_vis_resp_fixed.drop(['mouse', 'day'], axis=1)
        frac_vis_resp_free = frac_vis_resp_free.drop(['mouse', 'day'], axis=1)

        # Now we loop through the three states of the visual tuning and plot the violin plots
        vis_categories = ['', 'mod', 'not']

        for vis_cat in vis_categories:
            if vis_cat == '':
                count_cols = [col for col in count_vis_resp_fixed.columns if ('not' not in col) and ('mod' not in col)]
                frac_cols = [col for col in frac_vis_resp_fixed.columns if ('not' not in col) and ('mod' not in col)]

                rename_count_cols = count_cols
                rename_frac_cols = frac_cols

            else:
                count_cols = [col for col in count_vis_resp_fixed.columns if vis_cat in col]
                frac_cols = [col for col in frac_vis_resp_fixed.columns if vis_cat in col]

                rename_count_cols = [col.split('_', 1)[-1] for col in count_cols]
                rename_frac_cols = [col.split('_', 1)[-1] for col in frac_cols]

            count_vis_resp_fixed_temp = count_vis_resp_fixed[count_cols].copy()
            count_vis_resp_free_temp = count_vis_resp_free[count_cols].copy()
            count_vis_resp_fixed_temp.columns = rename_count_cols
            count_vis_resp_free_temp.columns = rename_count_cols

            frac_vis_resp_fixed_temp = frac_vis_resp_fixed[frac_cols].copy()
            frac_vis_resp_free_temp = frac_vis_resp_free[frac_cols].copy()
            frac_vis_resp_fixed_temp.columns = rename_frac_cols
            frac_vis_resp_free_temp.columns = rename_frac_cols

            # Plot fixed counts
            save_path = os.path.join(figure_save_path, f"count_{vis_cat}_vis_tuned_{session_shorthand[0]}.png")
            save_path = save_path.replace('__', '_')
            violinplot_fixed_vis_count = fp.violin_swarm(count_vis_resp_fixed_temp.copy(), save_path,
                                                         cmap=fixed_violin_cmap, font_size='paper', backend='seaborn',
                                                         width=4.5, height=5, save=True, ylim=(0, 150))
            plt.close()

            # Now plot them all individually
            for col in count_vis_resp_fixed_temp.columns:
                save_path = os.path.join(figure_save_path, f"count_{vis_cat}_vis_tuned_{col}_{session_shorthand[0]}.png")
                save_path = save_path.replace('__', '_')
                violinplot_fixed_vis_col = fp.violin_swarm(count_vis_resp_fixed.loc[:, [col]].copy(), save_path,
                                                           cmap=fixed_violin_cmap, font_size='paper', backend='seaborn',
                                                           width=1, height=4, save=True, ylim=(0, 150))
                plt.close()

            # Plot free counts
            save_path = os.path.join(figure_save_path, f"count_{vis_cat}_vis_tuned_{session_shorthand[1]}.png")
            save_path = save_path.replace('__', '_')
            violinplot_free_vis_count = fp.violin_swarm(count_vis_resp_free_temp.copy(), save_path,
                                                        cmap=free_violin_cmap, font_size='paper', backend='seaborn',
                                                        width=4.5, height=5, save=True, ylim=(0, 150))
            plt.close()

            # Now plot them all individually
            for col in count_vis_resp_free_temp.columns:
                save_path = os.path.join(figure_save_path, f"count_{vis_cat}_vis_tuned_{col}_{session_shorthand[1]}.png")
                save_path = save_path.replace('__', '_')
                violinplot_free_vis_col = fp.violin_swarm(count_vis_resp_free_temp.loc[:, [col]].copy(), save_path,
                                                          cmap=free_violin_cmap, font_size='paper', backend='seaborn',
                                                          width=1, height=4, save=True, ylim=(0, 150))
                plt.close()

            # Plot fixed fractions
            save_path = os.path.join(figure_save_path, f"frac_{vis_cat}_vis_tuned_{session_shorthand[0]}.png")
            save_path = save_path.replace('__', '_')
            violinplot_fixed_vis = fp.violin_swarm(frac_vis_resp_fixed_temp.copy(), save_path,
                                                   cmap=fixed_violin_cmap, font_size='paper', backend='seaborn',
                                                   width=4.5, height=5, save=True)
            plt.close()

            # Now plot them all individually
            for col in frac_vis_resp_fixed_temp.columns:
                save_path = os.path.join(figure_save_path, f"frac_{vis_cat}_vis_tuned_{col}_{session_shorthand[0]}.png")
                save_path = save_path.replace('__', '_')
                violinplot_fixed_vis_col = fp.violin_swarm(frac_vis_resp_fixed_temp.loc[:, [col]].copy(), save_path,
                                                           cmap=fixed_violin_cmap, font_size='paper', backend='seaborn',
                                                           width=1, height=4, save=True)
                plt.close()

            # Plot free fractions
            save_path = os.path.join(figure_save_path, f"frac_{vis_cat}_vis_tuned_{session_shorthand[1]}.png")
            save_path = save_path.replace('__', '_')
            violinplot_free_vis = fp.violin_swarm(frac_vis_resp_free_temp, save_path,
                                                  cmap=free_violin_cmap, font_size='paper', backend='seaborn',
                                                  width=4.5, height=5, save=True)
            plt.close()

            # Now plot them all individually
            for col in frac_vis_resp_free_temp.columns:
                save_path = os.path.join(figure_save_path, f"frac_{vis_cat}_vis_tuned_{col}_{session_shorthand[1]}.png")
                save_path = save_path.replace('__', '_')
                violinplot_free_vis_col = fp.violin_swarm(frac_vis_resp_free_temp.loc[:, [col]].copy(), save_path,
                                                          cmap=free_violin_cmap, font_size='paper', backend='seaborn',
                                                          width=1, height=4, save=True)
                plt.close()

        # --- Plot swarm plots with fraction of kinematic tuned cells for each variable --- #

        # Within-animal fraction kinematic - fixed or session 0
        save_path = os.path.join(figure_save_path, f"sig_frac_kinem_{session_shorthand[0]}.png")
        frac_kine_resp_fixed = get_sig_tuned_kinem_cells(data_dict, session_types[0], cell_kind,
                                                         processing_parameters.variable_list_fixed,
                                                         use_test=True, include_responsivity=True, include_consistency=False)

        with pd.HDFStore(data_save_path, 'a') as store:
            if f'frac_kinem_resp_{session_shorthand[0]}' in store.keys():
                del store[f'frac_kinem_resp_{session_shorthand[0]}']
            store[f'frac_kinem_resp_{session_shorthand[0]}'] = frac_kine_resp_fixed

        violinplot_fixed_kinem = fp.violin_swarm(frac_kine_resp_fixed, save_path,
                                                 cmap=fixed_violin_cmap, backend='seaborn', font_size='paper',
                                                 width=3, height=5, save=True)
        plt.close()

        # Now plot them all individually
        for col in frac_kine_resp_fixed.columns:
            save_path = os.path.join(figure_save_path, f"sig_frac_{col}_{session_shorthand[0]}.png")
            violinplot_fixed_kinem_col = fp.violin_swarm(frac_kine_resp_fixed.loc[:, [col]].copy(), save_path,
                                                         cmap=fixed_violin_cmap, font_size='paper', backend='seaborn',
                                                         width=1.25, height=5, save=True)
            plt.close()

        # Within-animal fraction kinematic - freely moving or session 1
        save_path = os.path.join(figure_save_path, f"sig_frac_kinem_{session_shorthand[1]}.png")
        frac_kine_resp_free = get_sig_tuned_kinem_cells(data_dict, session_types[1], cell_kind,
                                                        processing_parameters.variable_list_free,
                                                        use_test=True, include_responsivity=True, include_consistency=False)

        with pd.HDFStore(data_save_path, 'a') as store:
            if f'frac_kinem_resp_{session_shorthand[1]}' in store.keys():
                del store[f'frac_kinem_resp_{session_shorthand[1]}']
            store[f'frac_kinem_resp_{session_shorthand[1]}'] = frac_kine_resp_free

        violinplot_free_kinem = fp.violin_swarm(frac_kine_resp_free, save_path,
                                                cmap=free_violin_cmap, backend='seaborn', font_size='paper',
                                                width=15, height=5, save=True)
        plt.close()

        # Now plot them all individually
        for col in frac_kine_resp_free.columns:
            save_path = os.path.join(figure_save_path, f"sig_frac_{col}_{session_shorthand[1]}.png")
            violinplot_free_kinem_col = fp.violin_swarm(frac_kine_resp_free.loc[:, [col]].copy(), save_path,
                                                        cmap=free_violin_cmap, font_size='paper', backend='seaborn',
                                                        width=1.25, height=5, save=True)
            plt.close()

        # --- Plot the fraction of running modulated cells--- #
        fixed_running_mod = data_dict[f'{session_types[0]}_{cell_kind}_running_modulated_cells']
        frac_run_mod_fixed = fixed_running_mod.groupby(['mouse', 'day'])[['sig_run_modulated',
                                                                          'sig_vis_run_modulated']].sum() / \
                             fixed_running_mod.groupby(['mouse', 'day'])[['sig_run_modulated',
                                                                          'sig_vis_run_modulated']].count()
        rename_dict = dict(zip(list(frac_run_mod_fixed.columns), [col[4:] for col in frac_run_mod_fixed.columns]))
        frac_run_mod_fixed = frac_run_mod_fixed.rename(columns=rename_dict)

        for col in frac_run_mod_fixed.columns:
            save_path = os.path.join(figure_save_path, f"sig_frac_{col}_{session_shorthand[0]}.png")
            violinplot_fixed_run_mod_col = fp.violin_swarm(frac_run_mod_fixed.loc[:, [col]].copy(), save_path,
                                                           cmap=fixed_violin_cmap, font_size='paper', backend='seaborn',
                                                           width=1.5, height=5, save=True)
            plt.close()

        free_running_mod = data_dict[f'{session_types[1]}_{cell_kind}_running_modulated_cells']
        frac_run_mod_free = free_running_mod.groupby(['mouse', 'day'])[['sig_run_modulated',
                                                                        'sig_vis_run_modulated']].sum() / \
                            free_running_mod.groupby(['mouse', 'day'])[['sig_run_modulated',
                                                                        'sig_vis_run_modulated']].count()
        rename_dict = dict(zip(list(frac_run_mod_free.columns), [col[4:] for col in frac_run_mod_free.columns]))
        frac_run_mod_free = frac_run_mod_free.rename(columns=rename_dict)

        for col in frac_run_mod_free.columns:
            save_path = os.path.join(figure_save_path, f"sig_frac_{col}_{session_shorthand[1]}.png")
            violinplot_free_run_mod_col = fp.violin_swarm(frac_run_mod_free.loc[:, [col]].copy(), save_path,
                                                          cmap=free_violin_cmap, font_size='paper', backend='seaborn',
                                                          width=1.5, height=5, save=True)
            plt.close()

        # Save the counts and fractions
        with pd.HDFStore(data_save_path, 'a') as store:
            if f'frac_run_mod_{session_shorthand[0]}' in store.keys():
                del store[f'frac_run_mod_{session_shorthand[0]}']

            if f'frac_run_mod_{session_shorthand[1]}' in store.keys():
                del store[f'frac_run_mod_{session_shorthand[1]}']

            if f'frac_run_vis_mod_{session_shorthand[0]}' in store.keys():
                del store[f'frac_run_vis_mod_{session_shorthand[0]}']

            if f'frac_run_vis_mod_{session_shorthand[1]}' in store.keys():
                del store[f'frac_run_vis_mod_{session_shorthand[1]}']

            store[f'frac_run_mod_{session_shorthand[0]}'] = frac_run_mod_fixed
            store[f'frac_run_mod_{session_shorthand[1]}'] = frac_run_mod_free

        # --- Plot the delta preferred orientation/direction, and delta DSI/OSI --- #
        # This can only me done with matched cells, so try with both matched cells and curated cells
        match_kind = ['all_matches', 'curated_matches']

        for m_kind in match_kind:

            # Get the hand-curated matches and filter the data by these
            if m_kind == 'curated_matches':

                if curated_matches is not None:
                    curated_ref_tcs_list = []
                    curated_comp_tcs_list = []

                    for day, mouse in curated_matches[['day', 'mouse']].drop_duplicates().to_numpy():
                        # There's occasionally bullshit where some days that I did matching are excluded from the
                        # dataset for one reason or another. This throws an an error. Skip them.

                        try:
                            curated_day_mouse_matches = curated_matches.loc[(curated_matches.mouse == mouse) &
                                                                             (curated_matches.day == day), :]

                            ref_original_cell_id = curated_day_mouse_matches.iloc[:, ref_idx_order[0]].astype(int)
                            ref_original_cell_id = [f'cell_{cell_id:04d}' for cell_id in ref_original_cell_id]

                            comp_original_cell_id = curated_day_mouse_matches.iloc[:, ref_idx_order[1]].astype(int)
                            comp_original_cell_id = [f'cell_{cell_id:04d}' for cell_id in comp_original_cell_id]

                            ref_df = fixed_tcs_by_cell.loc[(fixed_tcs_by_cell.mouse == mouse) &
                                                           (fixed_tcs_by_cell.day == day)].copy()
                            comp_df = free_tcs_by_cell.loc[(free_tcs_by_cell.mouse == mouse) &
                                                           (free_tcs_by_cell.day == day)].copy()

                            curated_ref_df = ref_df.set_index('cell').loc[ref_original_cell_id].copy().reset_index()
                            curated_comp_df = comp_df.set_index('cell').loc[comp_original_cell_id].copy().reset_index()

                            curated_ref_tcs_list.append(curated_ref_df.copy())
                            curated_comp_tcs_list.append(curated_comp_df.copy())

                        except:
                            print(f'Weird error with curated matches for {mouse} on day {day}. Skipping...')
                            continue

                    curated_ref_df = pd.concat(curated_ref_tcs_list)
                    curated_comp_df = pd.concat(curated_comp_tcs_list)

                    ref_tcs = curated_ref_df.reset_index(drop=True)
                    comp_tcs = curated_comp_df.reset_index(drop=True)

                else:
                    print('No curated matches for these experiments. Using all matches...')
                    continue

            # Use the CaImAn matches
            else:
                match_ref_tcs_list = []
                match_comp_tcs_list = []
                for day, mouse in all_cell_matches[['day', 'mouse']].drop_duplicates().to_numpy():

                    day_mouse_matches = all_cell_matches.loc[(all_cell_matches.mouse == mouse) &
                                                             (all_cell_matches.day == day)]
                    day_mouse_matches.dropna(inplace=True)

                    ref_original_cell_id = day_mouse_matches.iloc[:, ref_idx_order[0]].astype(int).to_list()
                    ref_original_cell_id = [f'cell_{cell_id:04d}' for cell_id in ref_original_cell_id]

                    comp_original_cell_id = day_mouse_matches.iloc[:, ref_idx_order[1]].astype(int)
                    comp_original_cell_id = [f'cell_{cell_id:04d}' for cell_id in comp_original_cell_id]

                    ref_df = fixed_tcs_by_cell.loc[(fixed_tcs_by_cell.mouse == mouse) &
                                                   (fixed_tcs_by_cell.day == day)].copy()
                    comp_df = free_tcs_by_cell.loc[(free_tcs_by_cell.mouse == mouse) &
                                                   (free_tcs_by_cell.day == day)].copy()

                    match_ref_this_df = ref_df.loc[ref_df.cell.isin(ref_original_cell_id)]
                    match_comp_this_df = comp_df.loc[comp_df.cell.isin(comp_original_cell_id)]

                    match_ref_tcs_list.append(match_ref_this_df.copy())
                    match_comp_tcs_list.append(match_comp_this_df.copy())

                match_ref_df = pd.concat(match_ref_tcs_list)
                match_comp_df = pd.concat(match_comp_tcs_list)

                ref_tcs = match_ref_df
                comp_tcs = match_comp_df

            # --- Calculate the OSI/DSI shifts --- #

            ref_tcs = ref_tcs.reset_index(drop=True)
            comp_tcs = comp_tcs.reset_index(drop=True)

            # Set any selectivity index less than 0 to 0
            ref_tcs.loc[ref_tcs.fit_osi < 0, 'fit_osi'] = 0
            ref_tcs.loc[ref_tcs.fit_dsi < 0, 'fit_dsi'] = 0
            comp_tcs.loc[comp_tcs.fit_osi < 0, 'fit_osi'] = 0
            comp_tcs.loc[comp_tcs.fit_dsi < 0, 'fit_dsi'] = 0

            # The following calculations should only include cells that are visually responsive in both experiments
            both_resp = np.intersect1d(ref_tcs.loc[ref_tcs.is_vis_resp == 1].index,
                                       comp_tcs.loc[comp_tcs.is_vis_resp == 1].index)

            if both_resp.size == 0:
                print('No cells are visually responsive in both experiments. Skipping...')
                continue

            ref_tcs_both_resp = ref_tcs.iloc[both_resp, :].copy().reset_index(drop=True)
            comp_tcs_both_resp = comp_tcs.iloc[both_resp, :].copy().reset_index(drop=True)

            # Loop through the two datasets (resp during both exps or without regard to resp changes)
            for vis_resp_kind, ref, comp in zip(['all_vis_resp', 'both_vis_resp'],
                                                [ref_tcs, ref_tcs_both_resp],
                                                [comp_tcs, comp_tcs_both_resp]):

                with pd.HDFStore(data_save_path, 'a') as store:
                    if f'ref_cells_{m_kind}_{vis_resp_kind}' in store.keys():
                        del store[f'ref_cells_{m_kind}_{vis_resp_kind}']

                    if f'comp_cells_{m_kind}_{vis_resp_kind}' in store.keys():
                        del store[f'comp_cells_{m_kind}_{vis_resp_kind}']

                    store[f'ref_cells_{m_kind}_{vis_resp_kind}'] = ref
                    store[f'comp_cells_{m_kind}_{vis_resp_kind}'] = comp

                # --- Get the shifts in OSI/DSI

                # First do it for all the cells
                all_osi_shifts, all_osi_residuals, rmse_residual_osi_all = \
                    calculate_delta_selectivity(ref.copy(), comp.copy(),
                                                cutoff=0, stim_kind='orientation')

                all_dsi_shifts, all_dsi_residuals, rmse_residual_dsi_all = \
                    calculate_delta_selectivity(ref.copy(), comp.copy(),
                                                cutoff=0, stim_kind='direction')

                # Now do it for the cells that are orientation/direction tuned
                tuned_osi_shifts, tuned_osi_residuals, rmse_residual_osi_tuned = \
                    calculate_delta_selectivity(ref.copy(), comp.copy(),
                                                cutoff=processing_parameters.selectivity_idx_cutoff,
                                                stim_kind='orientation')

                tuned_dsi_shifts, tuned_dsi_residuals, rmse_residual_dsi_tuned = \
                    calculate_delta_selectivity(ref.copy(), comp.copy(),
                                                cutoff=processing_parameters.selectivity_idx_cutoff,
                                                stim_kind='direction')

                # Save the shifts
                with pd.HDFStore(data_save_path, 'a') as store:
                    store[f'osi_shifts_vis_resp_{m_kind}_{vis_resp_kind}'] = all_osi_shifts
                    store[f'dsi_shifts_vis_resp_{m_kind}_{vis_resp_kind}'] = all_dsi_shifts
                    store[f'osi_shifts_tuned_{m_kind}_{vis_resp_kind}'] = tuned_osi_shifts
                    store[f'dsi_shifts_tuned_{m_kind}_{vis_resp_kind}'] = tuned_dsi_shifts

                    store[f'osi_residuals_vis_resp_{m_kind}_{vis_resp_kind}'] = pd.DataFrame.from_dict({'osi': all_osi_residuals})
                    store[f'dsi_residuals_vis_resp_{m_kind}_{vis_resp_kind}'] = pd.DataFrame.from_dict({'dsi': all_dsi_residuals})
                    store[f'osi_residuals_tuned_{m_kind}_{vis_resp_kind}'] = pd.DataFrame.from_dict({'osi': tuned_osi_residuals})
                    store[f'dsi_residuals_tuned_{m_kind}_{vis_resp_kind}'] = pd.DataFrame.from_dict({'dsi': tuned_dsi_residuals})

                    store[f'selectivity_residual_rmse_vis_resp_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'osi': [rmse_residual_osi_all], 'dsi': [rmse_residual_dsi_all]})
                    store[f'selectivity_residual_rmse_tuned_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'osi': [rmse_residual_osi_tuned], 'dsi': [rmse_residual_dsi_tuned]})

                # Plot the shift in OSI
                scatter_OSI_all = plot_delta_pref(all_osi_shifts, session_types, type='orientation')
                scatter_OSI_all.opts(hv.opts.Scatter(color='gray', alpha=0.5))

                scatter_OSI_tuned = plot_delta_pref(tuned_osi_shifts, session_types, type='orientation')
                scatter_OSI_tuned.opts(hv.opts.Scatter(color=scatter_color_theme))

                scatter_OSI = hv.Overlay([scatter_OSI_all, scatter_OSI_tuned]).opts(show_legend=False)

                scatter_OSI.opts(xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} OSI',
                                 ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} OSI')
                scatter_OSI.opts(
                    hv.opts.Scatter(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1),
                                    xticks=(0, 0.5, 1), yticks=(0, 0.5, 1),
                                    size=4))

                save_path = os.path.join(figure_save_path, f"delta_OSI_{m_kind}_{vis_resp_kind}.png")
                scatter_OSI = fp.save_figure(scatter_OSI, save_path=save_path, fig_width=6, dpi=800, fontsize='paper',
                                             target='save', display_factor=0.1)

                # Plot the shift in DSI
                scatter_DSI_all = plot_delta_pref(all_dsi_shifts, session_types, type='direction')
                scatter_DSI_all.opts(hv.opts.Scatter(color='gray', alpha=0.5))

                scatter_DSI_tuned = plot_delta_pref(tuned_dsi_shifts, session_types, type='direction')
                scatter_DSI_tuned.opts(hv.opts.Scatter(color=scatter_color_theme))

                scatter_DSI = hv.Overlay([scatter_DSI_all, scatter_DSI_tuned]).opts(show_legend=False)

                scatter_DSI.opts(xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} DSI',
                                 ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} DSI')
                scatter_DSI.opts(hv.opts.Scatter(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1),
                                                 xticks=(0, 0.5, 1), yticks=(0, 0.5, 1),
                                                 size=4))

                save_path = os.path.join(figure_save_path, f"delta_DSI_{m_kind}_{vis_resp_kind}.png")
                scatter_DSI = fp.save_figure(scatter_DSI, save_path=save_path, fig_width=6, dpi=800, fontsize='paper',
                                             target='save', display_factor=0.1)

                # Get the shift in preferred orientation and direction
                all_po_shifts, all_ori_residuals, rmse_residual_ori_all = \
                    calculate_pref_angle_shifts(ref.copy(), comp.copy(),
                                                stim_kind='orientation', upper_cutoff=0)

                all_pd_shifts, all_dir_residuals, rmse_residual_dir_all = \
                    calculate_pref_angle_shifts(ref.copy(), comp.copy(),
                                                stim_kind='direction', upper_cutoff=0)

                tuned_po_shifts, tuned_ori_residuals, rmse_residual_ori_tuned = \
                    calculate_pref_angle_shifts(ref.copy(), comp.copy(),
                                                stim_kind='orientation',
                                                upper_cutoff=processing_parameters.selectivity_idx_cutoff,)

                tuned_pd_shifts, tuned_dir_residuals, rmse_residual_dir_tuned = \
                    calculate_pref_angle_shifts(ref.copy(), comp.copy(),
                                                stim_kind='direction',
                                                upper_cutoff=processing_parameters.selectivity_idx_cutoff,)

                with pd.HDFStore(data_save_path, 'a') as store:
                    store[f'po_shifts_vis_resp_{m_kind}_{vis_resp_kind}'] = all_po_shifts
                    store[f'pd_shifts_vis_resp_{m_kind}_{vis_resp_kind}'] = all_pd_shifts
                    store[f'po_residuals_vis_resp_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'orientation': all_ori_residuals})
                    store[f'pd_residuals_vis_resp_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'direction': all_dir_residuals})
                    store[f'angle_residual_rmse_vis_resp_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'orientation': [rmse_residual_ori_all],
                                                'direction': [rmse_residual_dir_all]})

                    store[f'po_shifts_tuned_{m_kind}_{vis_resp_kind}'] = tuned_po_shifts
                    store[f'pd_shifts_tuned_{m_kind}_{vis_resp_kind}'] = tuned_pd_shifts
                    store[f'po_residuals_tuned_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'orientation': tuned_ori_residuals})
                    store[f'pd_residuals_tuned_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'direction': tuned_dir_residuals})
                    store[f'angle_residual_rmse_tuned_{m_kind}_{vis_resp_kind}'] = \
                        pd.DataFrame.from_dict({'orientation': [rmse_residual_ori_tuned],
                                                'direction': [rmse_residual_dir_tuned]})

                # Plot the shift in preferred orientation
                scatter_PO_all = plot_delta_pref(all_po_shifts, session_types, type='orientation', wrap_neg=False)
                scatter_PO_all.opts(hv.opts.Scatter(color='gray', size=4, alpha=0.5))

                scatter_PO = plot_delta_pref(tuned_po_shifts, session_types, type='orientation', wrap_neg=False)
                scatter_PO.opts(hv.opts.Scatter(color=scatter_color_theme, size=4, show_legend=False))

                scatter_PO = hv.Overlay([scatter_PO_all, scatter_PO]).opts(show_legend=False)
                scatter_PO.opts(
                    xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} Pref. [°]',
                    ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} Pref. [°]')

                save_path = os.path.join(figure_save_path, f"delta_PO_{m_kind}_{vis_resp_kind}.png")
                scatter_PO = fp.save_figure(scatter_PO, save_path=save_path, fig_width=6, dpi=800, fontsize='paper',
                                            target='save', display_factor=0.1)

                # Plot the shift in preferred direction
                scatter_PD_all = plot_delta_pref(all_pd_shifts, session_types, type='direction', wrap_neg=False)
                scatter_PD_all.opts(hv.opts.Scatter(color='gray', size=4, alpha=0.5))

                scatter_PD = plot_delta_pref(tuned_pd_shifts, session_types, type='direction', wrap_neg=False)
                scatter_PD.opts(hv.opts.Scatter(color=scatter_color_theme, size=4, show_legend=False))

                scatter_PD = hv.Overlay([scatter_PD_all, scatter_PD]).opts(show_legend=False)
                scatter_PD.opts(
                    xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} Pref. [°]',
                    ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} Pref. [°]')

                save_path = os.path.join(figure_save_path, f"delta_PD_{m_kind}_{vis_resp_kind}.png")
                scatter_PD = fp.save_figure(scatter_PD, save_path=save_path, fig_width=6, dpi=800, fontsize='paper',
                                            target='save', display_factor=0.1)

        # --- UMAP --- #
        # Run UMAP on all cell permutations
        reducer = UMAP(min_dist=0.1, n_neighbors=50)

        if (parsed_search['result'] == 'repeat') and (parsed_search['rig'] == 'VWheelWF'):
            kinem_label_list = processing_parameters.variable_list_fixed
        elif (parsed_search['result'] == 'repeat') and (parsed_search['rig'] == 'VTuningWF'):
            kinem_label_list = processing_parameters.variable_list_free
        else:
            kinem_label_list = processing_parameters.variable_list_free + processing_parameters.variable_list_fixed

        label_list = kinem_label_list + [activity_dataset]

        cell_kinds = ['all_cells', 'matched', 'unmatched']  # options: 'all_cells', 'matched', 'unmatched'

        for cell_kind in cell_kinds:
            labels = [f"_{cell_kind}_{label}" for label in label_list]

            # If the cells are matched, we always have the same number of cells
            if cell_kind == "matched":
                umap_dict = {}
                for label in labels:
                    data_keys = [key for key in data_dict.keys() if label in key]

                    for key in data_keys:
                        ds = data_dict[key]
                        base_label = '_'.join(label.split('_')[1 + len(cell_kind.split('_')):])

                        if base_label in kinem_label_list:
                            tuning = ds['Qual_index']
                            umap_dict[base_label] = tuning
                        else:
                            this_rig = key.split('_')[0]
                            tuning_dsi = ds['fit_dsi'].abs().to_numpy()
                            tuning_osi = ds['fit_osi'].to_numpy()
                            umap_dict[f'dsi_{this_rig}'] = np.clip(tuning_dsi, 0, 1)
                            umap_dict[f'osi_{this_rig}'] = np.clip(tuning_osi, 0, 1)

                raw_tunings = pd.DataFrame.from_dict(umap_dict)
                raw_tunings = raw_tunings.fillna(0)

                tunings = preproc.StandardScaler().fit_transform(raw_tunings.to_numpy())
                print(f'{cell_kind} - {tunings.shape[0]} cells')

                # perform umap on the fit cell tuning
                embedded_data = reducer.fit_transform(tunings)

                perc = 99
                predictor_columns = umap_dict.keys()
                plot_list = []

                for i, predictor_column in enumerate(predictor_columns):
                    label_idx = [idx for idx, el in enumerate(predictor_columns) if predictor_column == el]
                    raw_labels = tunings[:, label_idx]

                    raw_labels = np.abs(raw_labels)

                    raw_labels[raw_labels > np.percentile(raw_labels, perc)] = np.percentile(raw_labels, perc)
                    raw_labels[raw_labels < np.percentile(raw_labels, 100 - perc)] = np.percentile(raw_labels, 100 - perc)

                    plot_data = np.concatenate([embedded_data, raw_labels.reshape((-1, 1))], axis=1)

                    umap_plot = create_umap_plot(plot_data, predictor_column)

                    save_name = os.path.join(figure_save_path, f"{cell_kind}_UMAP_{predictor_column}.png")
                    umap_plot = fp.save_figure(umap_plot, save_path=save_name, fig_width=3.5, dpi=800, fontsize='paper',
                                               target='save', display_factor=0.1)

            else:
                # Here there may be uneven numbers of cells between sessions
                umap_dict_1 = {}
                umap_dict_2 = {}

                for label in labels:
                    data_keys = [key for key in data_dict.keys() if label in key]

                    for key in data_keys:
                        ds = data_dict[key]
                        this_rig = key.split('_')[0]
                        base_label = '_'.join(label.split('_')[1 + len(cell_kind.split('_')):])

                        if base_label in kinem_label_list:
                            tuning = ds['Qual_index']

                            if this_rig in ['VWheelWF', 'VTuningWF']:
                                if base_label in processing_parameters.variable_list_free:
                                    umap_dict_1[base_label] = tuning
                                else:
                                    umap_dict_2[base_label] = tuning
                            else:
                                if this_rig in ['free0', 'fixed0']:
                                    umap_dict_1[base_label] = tuning
                                else:
                                    umap_dict_2[base_label] = tuning
                        else:
                            tuning_dsi = ds['fit_dsi'].abs().to_numpy()
                            tuning_osi = ds['fit_osi'].to_numpy()

                            if this_rig in ['VTuningWF', 'free0', 'fixed0']:
                                umap_dict_1[f'dsi_{this_rig}'] = np.clip(tuning_dsi, 0, 1)
                                umap_dict_1[f'osi_{this_rig}'] = np.clip(tuning_osi, 0, 1)
                            else:
                                umap_dict_2[f'dsi_{this_rig}'] = np.clip(tuning_dsi, 0, 1)
                                umap_dict_2[f'osi_{this_rig}'] = np.clip(tuning_osi, 0, 1)

                raw_tunings_1 = pd.DataFrame.from_dict(umap_dict_1)
                raw_tunings_1 = raw_tunings_1.fillna(0)
                raw_tunings_2 = pd.DataFrame.from_dict(umap_dict_2)
                raw_tunings_2 = raw_tunings_2.fillna(0)

                tunings_1 = preproc.StandardScaler().fit_transform(raw_tunings_1.to_numpy())
                tunings_2 = preproc.StandardScaler().fit_transform(raw_tunings_2.to_numpy())

                print(f'{cell_kind} - {tunings_1.shape[0]} cells')
                print(f'{cell_kind} - {tunings_2.shape[0]} cells')

                # perform umap on the fit cell tuning
                embedded_data_1 = reducer.fit_transform(tunings_1)
                embedded_data_2 = reducer.fit_transform(tunings_2)

                perc = 99
                plot_list = []

                for umap_dict, embedded_data, tunings in zip([umap_dict_1, umap_dict_2],
                                                  [embedded_data_1, embedded_data_2],
                                                  [tunings_1, tunings_2]):

                    predictor_columns = umap_dict.keys()

                    for i, predictor_column in enumerate(predictor_columns):
                        label_idx = [idx for idx, el in enumerate(predictor_columns) if predictor_column == el]
                        raw_labels = tunings[:, label_idx]

                        raw_labels = np.abs(raw_labels)

                        raw_labels[raw_labels > np.percentile(raw_labels, perc)] = np.percentile(raw_labels, perc)
                        raw_labels[raw_labels < np.percentile(raw_labels, 100 - perc)] = np.percentile(raw_labels, 100 - perc)

                        plot_data = np.concatenate([embedded_data, raw_labels.reshape((-1, 1))], axis=1)

                        umap_plot = create_umap_plot(plot_data, predictor_column)

                        save_name = os.path.join(figure_save_path, f"{cell_kind}_UMAP_{predictor_column}.png")
                        umap_plot = fp.save_figure(umap_plot, save_path=save_name, fig_width=3.5, dpi=800, fontsize='paper',
                                                   target='save', display_factor=0.1)


# def process_predictor_columns(umap_dict: Dict[str, np.ndarray], tunings: np.ndarray, embedded_data: np.ndarray,
#                               perc: int, figure_save_path: str, cell_kind: str) -> Scatter:
#     """
#     This function processes predictor columns, creates UMAP plots and saves them.
#
#     Parameters:
#     umap_dict (dict): A dictionary containing UMAP data.
#     tunings (np.ndarray): An array containing tuning data.
#     embedded_data (np.ndarray): An array containing embedded data.
#     perc (int): The percentile to use for data normalization.
#     figure_save_path (str): The path where the figure will be saved.
#     cell_kind (str): The kind of cell.
#
#     Returns:
#     hv.Scatter: A Scatter plot showing the UMAP.
#     """
#     predictor_columns = umap_dict.keys()
#
#     for i, predictor_column in enumerate(predictor_columns):
#         label_idx = [idx for idx, el in enumerate(predictor_columns) if predictor_column == el]
#         raw_labels = tunings[:, label_idx]
#
#         raw_labels = np.abs(raw_labels)
#
#         raw_labels[raw_labels > np.percentile(raw_labels, perc)] = np.percentile(raw_labels, perc)
#         raw_labels[raw_labels < np.percentile(raw_labels, 100 - perc)] = np.percentile(raw_labels, 100 - perc)
#
#         plot_data = np.concatenate([embedded_data, raw_labels.reshape((-1, 1))], axis=1)
#
#         umap_plot = create_umap_plot(plot_data, predictor_column)
#
#         save_name = os.path.join(figure_save_path, f"{cell_kind}_UMAP_{predictor_column}.png")
#         umap_plot = fp.save_figure(umap_plot, save_path=save_name, fig_width=6, dpi=800, fontsize='paper',
#                                    target='save', display_factor=0.1)
#
#     return umap_plot

print("Done!")
