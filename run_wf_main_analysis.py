#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import itertools
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.stats as st
import holoviews as hv
import matplotlib.pyplot as plt
import sklearn.preprocessing as preproc
from umap.umap_ import UMAP

import paths
import processing_parameters
import functions_bondjango as bd
import functions_data_handling as fdh
import functions_kinematic as fk
import functions_misc as misc
import functions_plotting as fp

warnings.filterwarnings('ignore')


def get_vis_tuned_cells(ds: pd.DataFrame, vis_stim: str = 'dir', resp_thresh: float = 0.3, sel_tresh: float = 0.5,
                        drop_na: bool = True) -> pd.DataFrame:
    """
    Filter the input DataFrame to select cells that are visually tuned based on the specified visual stimulus.

    Parameters:
        ds (pd.DataFrame): The input DataFrame containing the data.
        vis_stim (str): The visual stimulus type. Valid values are 'dir', 'ori', 'vis', or 'untuned'.
        resp_thresh (float): The threshold for responsivity. Cells with absolute responsivity values below this threshold will be excluded.
        sel_tresh (float): The threshold for selectivity. Cells with absolute selectivity values below this threshold will be excluded.
        drop_na (bool): Whether to drop rows with missing values. If True, rows with missing values will be dropped. If False, rows with missing values will be included.

    Returns:
        pd.DataFrame: The filtered DataFrame containing the visually tuned cells.

    Raises:
        Exception: If an invalid vis_stim value is provided.

    """

    data = ds.copy()

    if vis_stim == 'dir':
        resp_type = f'responsivity_{vis_stim}'
        sel_type = 'dsi_abs'
    elif vis_stim == 'ori':
        resp_type = f'responsivity_{vis_stim}'
        sel_type = 'osi'
    elif (vis_stim == 'vis') or (vis_stim == 'untuned') :
        resp_type = ['responsivity_dir', 'responsivity_ori']
    else:
        raise Exception('Invalid vis_stim')

    if vis_stim == 'vis':
        cells = data[(data[resp_type[0]].abs() >= resp_thresh) & (data[resp_type[1]].abs() >= resp_thresh)]
    elif vis_stim == 'untuned':
        cells = data[(data[resp_type[0]].abs() < resp_thresh) & (data[resp_type[1]].abs() < resp_thresh)]
    else:
        cells = data[(data[resp_type].abs() >= resp_thresh) & (data[sel_type].abs() >= sel_tresh)]
    
    return cells


def kine_fraction_tuned(ds: pd.DataFrame, use_test: bool = True, include_responsivity: bool = True, include_consistency: bool = False) -> float:
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
        index_ref = ref_data.loc[ref_data.osi.abs() >= cutoff].index.to_numpy()
        index_comp = comp_data.loc[comp_data.osi.abs() >= cutoff].index.to_numpy()
    elif stim_kind == 'direction':
        index_ref = ref_data.loc[ref_data.dsi_abs.abs() >= cutoff].index.to_numpy()
        index_comp = comp_data.loc[comp_data.dsi_abs.abs() >= cutoff].index.to_numpy()
    else:
        raise Exception('Invalid stim_kind')

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

    shifts = []

    stim_kind = stim_kind[:3]
    if stim_kind == 'ori':
        multiplier = 2
    else:
        multiplier = 1

    if method == 'fit':
        dist_key = f'bootstrap_pref_{stim_kind}'
        pref_key = f'pref_{stim_kind}'
    elif method == 'resultant':
        dist_key = f'bootstrap_resultant_{stim_kind}'
        pref_key = f'resultant_{stim_kind}'
    else:
        raise Exception('Invalid method')

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
        pref_dist_1 = fk.wrap(pref_dist_1, 360 / multiplier + 0.1)
        pref_dist_2 = fk.wrap(pref_dist_2, 360 / multiplier + 0.1)
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

        # wrap to negative domain for plotting
        pref_1 = fk.wrap_negative(pref_1, bound=360 / (2 * multiplier) + 0.1)
        pref_2 = fk.wrap_negative(pref_2, bound=360 / (2 * multiplier) + 0.1)

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
    else:
        ci_width = 90

    # Find the indices of cells that are tuned
    indices = find_tuned_cell_indices(ref_ds.copy(), comp_ds.copy(), cutoff=upper_cutoff,
                                      stim_kind=stim_kind, tuning_criteria='both')

    # Calculate the shifts in tuning
    shifts = pref_angle_shifts(ref_ds.iloc[indices, :].copy(), comp_ds.iloc[indices, :].copy(),
                           ci_width_cutoff=ci_width, stim_kind=stim_kind, method='resultant')

    # Calculate the residuals and the root mean square error of the residuals
    residuals = shifts['pref_2'].to_numpy() - shifts['pref_1'].to_numpy()
    rmse_residual = np.mean(np.sqrt(residuals ** 2))

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
        sel_key = 'osi'
    else:
        sel_key = 'dsi_abs'

    # Find the indices of cells that are tuned
    indices = find_tuned_cell_indices(ref_ds.copy(), comp_ds.copy(), cutoff=cutoff,
                                      stim_kind=stim_kind, tuning_criteria='both')

    # Calculate the shifts in tuning
    ref_sel = ref_ds.iloc[indices, :][sel_key].copy()
    comp_sel = comp_ds.iloc[indices, :][sel_key].copy(),
    delta_sel = comp_sel - ref_sel
    shifts = pd.DataFrame(data={'pref_1': ref_sel, 'pref_2': comp_sel, 'delta_sel': delta_sel})

    # Calculate the residuals and the root mean square error of the residuals
    residuals = shifts['pref_2'].to_numpy() - shifts['pref_1'].to_numpy()
    rmse_residual = np.mean(np.sqrt(residuals ** 2))

    return shifts, residuals, rmse_residual


def plot_delta_pref(shifts: pd.DataFrame, rmse_residual: float, session_types: List[str],
                    type: str = 'orientation') -> hv.Scatter:
    """
    This function plots the shift in preferred orientation or direction.

    Parameters:
    shifts (pd.DataFrame): A DataFrame containing the shifts in preferred angle.
    rmse_residual (float): The root mean square error of the residuals.
    session_types (list): A list containing session types.
    type (str): The type of stimulus. Valid values are 'orientation' and 'direction'. Defaults to 'orientation'.

    Returns:
    hv.Scatter: A Scatter plot showing the shift in preferred orientation.
    """

    if type == 'orientation':
        upper_lim = 90
        lower_lim = -90
    elif type == 'direction':
        upper_lim = 180
        lower_lim = -180
    else:
        raise Exception('Invalid type')

    unity_line = hv.Curve((np.arange(lower_lim, upper_lim, 1), np.arange(lower_lim, upper_lim, 1))).opts(color='black')

    scatter_delta = hv.Scatter(shifts[['pref_1', 'pref_2']], kdims=['pref_1'], vdims=['pref_2'], label='sig').opts(
        color='purple', size=8)

    scatter = unity_line * scatter_delta * hv.Text(lower_lim+30, upper_lim-10,
                                                   f'RMSE: {rmse_residual:.1f}°').opts(color='black', fontsize=15)

    scatter.opts(show_legend=False, width=500, height=500,
                 xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} Pref. '
                        f'{type[:3].title()}. [°]',
                 ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} Pref. '
                        f'{type[:3].title()}. [°]')
    scatter.opts(
        hv.opts.Scatter(xlim=(lower_lim, upper_lim), ylim=(lower_lim, upper_lim),
                        xticks=np.linspace(lower_lim//2, upper_lim, 4),
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
        umap_plot.opts(title=f"{predictor_column[:3].upper()} "
                             f"{processing_parameters.wf_label_dictionary[predictor_column.split('_')[-1]]}")
    else:
        umap_plot.opts(title=processing_parameters.wf_label_dictionary[predictor_column])

    umap_plot.opts(width=300, height=300, size=2)

    return umap_plot


# set up the figure theme and saving paths
fp.set_theme()
in2cm = 1./2.54

# define the experimental conditions
results = ['multi'] # ['multi', 'fullfield', 'control', 'repeat']
lightings = ['normal'] # ['normal', 'dark']
rigs = [''] # ['', 'VWheelWF', 'VTuningWF']
analysis_type = 'agg_all'

for result, light, rig in itertools.product(results, lightings, rigs):

    if (result == 'repeat') and (rig == ''):
        continue
    elif (result != 'repeat') and (rig != ''):
        continue
    else:
        # get the search string
        search_string = f"result:{result}, lighting:{light}, rig:{rig}, analysis_type:{analysis_type}"

    # Load the data and set up figure saving path
    parsed_search = fdh.parse_search_string(search_string)
    file_infos = bd.query_database("analyzed_data", search_string)
    input_paths = np.array([el['analysis_path'] for el in file_infos])

    if input_paths.shape == 0:
        continue
    else:
        input_path = input_paths[0]

    data_dict = {}
    with pd.HDFStore(input_path, 'r') as tc:
        for key in tc.keys():
            label = "_".join(key.split('/')[1:])
            data = tc[key]
            data_dict[label] = data

    save_suffix = f"{parsed_search['result']}_{parsed_search['lighting']}_{parsed_search['rig']}"
    figure_save_path = os.path.join(paths.wf_figures_path, save_suffix)
    if not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)

    if parsed_search['result'] == 'repeat':
        if parsed_search['rig'] == 'VWheelWF':
            session_types = ['fixed0', 'fixed1']
        else:
            session_types = ['free0', 'free1']
        session_shorthand = session_types
    else:
        session_types = ['VWheelWF', 'VTuningWF']
        session_shorthand = ['fixed', 'free']

    # Specify the path to the curated cell matches file
    curated_cell_matches_path = os.path.join(r"C:\Users\mmccann\Desktop", 
                                f"curated_cell_matches_{parsed_search['result']}_{parsed_search['lighting']}_{parsed_search['rig']}.xlsx")

    try:
        # Read all sheets into a list of dataframes
        curated_matches_dict = pd.read_excel(curated_cell_matches_path, sheet_name=None)

        # Concatenate the dataframes into a single dataframe
        curated_matches = pd.concat(curated_matches_dict.values(), ignore_index=True)

    except Exception as e:
        print(f"Could not find the file {curated_cell_matches_path}. Continuing with CaImAn matches...")
        curated_matches = None

    # Get the cell matches for the current dataset and plot a scatter plot of the fraction of cells matched
    cell_kind = 'all_cells'
    activity_dataset = 'norm_spikes_viewed_props'

    match_nums = data_dict['cell_matches'].groupby(['mouse', 'day'])[session_types[0]].count().values
    num0 = data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'].groupby(['mouse', 'day']).size().values
    num1 = data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'].groupby(['mouse', 'day']).size().values

    match_frac0 = match_nums/num0
    match_frac1 = match_nums/num1

    line = hv.Curve((np.linspace(0, 1, 101), np.linspace(0, 1, 101))).opts(color='gray')
    scatter = hv.Points((match_frac0, match_frac1))
    scatter.opts(xlim=(0, 1), xlabel=f'Frac. Match {session_shorthand[0].title()}',
                 ylim=(0, 1), ylabel=f'Frac. Match {session_shorthand[1].title()}',
                 color='blue', width=500, height=500)
    frac_cell_match = hv.Overlay([line, scatter])

    save_path = os.path.join(figure_save_path, "frac_cells_matched.png")
    frac_cell_match = fp.save_figure(frac_cell_match, save_path=save_path, fig_width=8, dpi=800, fontsize='paper',
                                     target='save', display_factor=0.1)

    # --- Get the visual responsivity of the cells --- #
    # Create dataframes to store binary tuning information
    fixed_cell_tunings = data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'][['old_index', 'mouse', 'day']].copy()
    free_cell_tunings = data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'][['old_index', 'mouse', 'day']].copy()

    # Cells that meet direction selectivity criteria
    free_dir_tuned = get_vis_tuned_cells(data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'], vis_stim='dir', resp_thresh=0.5)
    free_ori_tuned = get_vis_tuned_cells(data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'], vis_stim='ori', resp_thresh=0.5)

    # Cells that meet orientation selectivity criteria
    fixed_dir_tuned = get_vis_tuned_cells(data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'], vis_stim='dir', resp_thresh=0.5)
    fixed_ori_tuned = get_vis_tuned_cells(data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'], vis_stim='ori', resp_thresh=0.5)

    # Cells that meet visual responsivity criteria
    free_vis_resp = get_vis_tuned_cells(data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'], vis_stim='vis', resp_thresh=0.5)
    fixed_vis_resp = get_vis_tuned_cells(data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'], vis_stim='vis', resp_thresh=0.5)

    # Find cells that are both direction and orientation tuned, and figure out what to do with them.
    _, comm1, comm2 = np.intersect1d(free_dir_tuned.index, free_ori_tuned.index, return_indices=True)
    free_both_tuned = free_dir_tuned.iloc[comm1].copy()

    # Remove cells tuned to both from each category
    free_dir_tuned = free_dir_tuned.drop(free_dir_tuned.index[comm1])
    free_ori_tuned = free_ori_tuned.drop(free_ori_tuned.index[comm2])

    _, comm1, comm2 = np.intersect1d(fixed_dir_tuned.index, fixed_ori_tuned.index, return_indices=True)
    fixed_both_tuned = fixed_dir_tuned.iloc[comm1].copy()
    fixed_dir_tuned = fixed_dir_tuned.drop(fixed_dir_tuned.index[comm1])
    fixed_ori_tuned = fixed_ori_tuned.drop(fixed_ori_tuned.index[comm2])

    # Double check cells that are visually responsive, make sure that all are contained in the vis_resp
    free_resp_cells = np.unique(np.concatenate([free_dir_tuned.index, free_ori_tuned.index, free_both_tuned.index]))
    not_in_free_resp_cells = np.setdiff1d(free_vis_resp.index, free_resp_cells, assume_unique=True)
    free_vis_resp = pd.concat([free_vis_resp,
                               data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}'].iloc[not_in_free_resp_cells, :]])
    free_vis_resp = free_vis_resp.reset_index().drop_duplicates(subset=['index'])

    fixed_resp_cells = np.unique(np.concatenate([fixed_dir_tuned.index, fixed_ori_tuned.index, fixed_both_tuned.index]))
    not_in_fixed_resp_cells = np.setdiff1d(fixed_vis_resp.index, fixed_resp_cells, assume_unique=True)
    fixed_vis_resp = pd.concat([fixed_vis_resp,
                                data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}'].iloc[not_in_fixed_resp_cells, :]])
    fixed_vis_resp = fixed_vis_resp.reset_index().drop_duplicates(subset=['index'])

    # Assign the tunings to the binary tuning dataframes
    free_cell_tunings['is_vis_resp'] = free_cell_tunings.index.isin(free_vis_resp.index)
    free_cell_tunings['is_dir_tuned'] = free_cell_tunings.index.isin(free_dir_tuned.index)
    free_cell_tunings['is_ori_tuned'] = free_cell_tunings.index.isin(free_ori_tuned.index)

    fixed_cell_tunings['is_vis_resp'] = fixed_cell_tunings.index.isin(fixed_vis_resp.index)
    fixed_cell_tunings['is_dir_tuned'] = fixed_cell_tunings.index.isin(fixed_dir_tuned.index)
    fixed_cell_tunings['is_ori_tuned'] = fixed_cell_tunings.index.isin(fixed_ori_tuned.index)

    # --- Get fraction of cells that are orientation. direction, and visually tuned --- #
    # For freely moving session
    free_cell_per_day = free_cell_tunings.groupby(['mouse', 'day']).size()
    frac_free_dir_tuned = free_dir_tuned.groupby(['mouse', 'day']).size() / free_cell_per_day
    frac_free_dir_tuned = frac_free_dir_tuned.reset_index().rename(columns={0: 'direction'})

    frac_free_ori_tuned = free_ori_tuned.groupby(['mouse', 'day']).size() / free_cell_per_day
    frac_free_ori_tuned = frac_free_ori_tuned.reset_index().rename(columns={0: 'orientation'})

    frac_free_vis_resp = free_vis_resp.groupby(['mouse', 'day']).size() / free_cell_per_day
    frac_free_vis_resp = frac_free_vis_resp.reset_index().rename(columns={0: 'visual'})
    frac_vis_resp_free = pd.concat([frac_free_dir_tuned, frac_free_ori_tuned, frac_free_vis_resp], axis=1).drop(['mouse', 'day'], axis=1)

    save_path = os.path.join(figure_save_path, "frac_vis_tuned_free.png")
    violinplot_free_vis = fp.violin_swarm(frac_vis_resp_free, save_path,
                                          cmap=fp.hv_blue_hex, font_size='paper', backend='seaborn',
                                          width=4.5, height=5, save=True)
    plt.close()

    # For fixed session
    fixed_cell_per_day = fixed_cell_tunings.groupby(['mouse', 'day']).size()
    
    frac_fixed_dir_tuned = fixed_dir_tuned.groupby(['mouse', 'day']).size() / fixed_cell_per_day
    frac_fixed_dir_tuned = frac_fixed_dir_tuned.reset_index().rename(columns={0: 'direction'})
    
    frac_fixed_ori_tuned = fixed_ori_tuned.groupby(['mouse', 'day']).size() / fixed_cell_per_day
    frac_fixed_ori_tuned = frac_fixed_ori_tuned.reset_index().rename(columns={0: 'orientation'})
    
    frac_fixed_vis_resp = fixed_vis_resp.groupby(['mouse', 'day']).size() / fixed_cell_per_day
    frac_fixed_vis_resp = frac_fixed_vis_resp.reset_index().rename(columns={0: 'visual'})
    frac_vis_resp_fixed = pd.concat([frac_fixed_dir_tuned, frac_fixed_ori_tuned, frac_fixed_vis_resp], axis=1).drop(['mouse', 'day'], axis=1)

    save_path = os.path.join(figure_save_path, "frac_vis_tuned_fixed.png")
    violinplot_fixed_vis = fp.violin_swarm(frac_vis_resp_fixed, save_path,
                                           cmap='red', font_size='paper', backend='seaborn',
                                           width=4.5, height=5, save=True)
    plt.close()

    # --- Plot swarm plots with fraction of kinematic tuned cells for each variable --- #

    # Within-animal fraction kinematic - freely moving
    save_path = os.path.join(figure_save_path, "sig_frac_kinem_free.png")
    frac_kine_resp_free = get_sig_tuned_kinem_cells(data_dict, 'VTuningWF', cell_kind, processing_parameters.variable_list_free,
                                                    use_test=True, include_responsivity=True, include_consistency=False)
    violinplot_free_kinem = fp.violin_swarm(frac_kine_resp_free, save_path, 
                                            cmap=fp.hv_blue_hex, backend='seaborn', font_size='paper', 
                                            width=15, height=5, save=True)
    plt.close()

    # Within-animal fraction kinematic - fixed
    save_path = os.path.join(figure_save_path, "sig_frac_kinem_fixed.png")
    frac_kine_resp_fixed = get_sig_tuned_kinem_cells(data_dict, 'VWheelWF', cell_kind, processing_parameters.variable_list_fixed, 
                                                     use_test=True, include_responsivity=True, include_consistency=False)
    violinplot_fixed_kinem = fp.violin_swarm(frac_kine_resp_fixed, save_path,
                                            cmap='red', backend='seaborn', font_size='paper',
                                            width=3, height=5, save=True)
    plt.close()
    
    # --- Plot the fraction of multimodal tuned cells--- #
    # TODO

    # --- Plot the delta preferred orientation/direction, and delta DSI/OSI --- #

    # Get the hand-curated matches and filter the data by these
    if curated_matches is not None:
        curated_matches[['day', 'mouse']].drop_duplicates().to_numpy()
        fixed_matches = data_dict[f'{session_types[0]}_{cell_kind}_{activity_dataset}']
        free_matches = data_dict[f'{session_types[1]}_{cell_kind}_{activity_dataset}']

        curated_ref_df_list = []
        curated_comp_df_list = []
        for day, mouse in curated_matches[['day', 'mouse']].drop_duplicates().to_numpy():
            curated_idxs = curated_matches.loc[(curated_matches.mouse == mouse) & (curated_matches.day == day)][
                'index'].to_numpy()

            ref_df = fixed_matches.loc[(fixed_matches.mouse == mouse) & (fixed_matches.day == day)].copy()
            comp_df = free_matches.loc[(free_matches.mouse == mouse) & (free_matches.day == day)].copy()

            curated_ref_df = ref_df.loc[ref_df.old_index.isin(curated_idxs)]
            curated_comp_df = comp_df.loc[comp_df.old_index.isin(curated_idxs)]

            curated_ref_df_list.append(curated_ref_df.copy())
            curated_comp_df_list.append(curated_comp_df.copy())

        curated_ref_df = pd.concat(curated_ref_df_list).reset_index(drop=True)
        curated_comp_df = pd.concat(curated_comp_df_list).reset_index(drop=True)

        ref_ds = curated_ref_df
        comp_ds = curated_comp_df

    else:
        ref_ds = data_dict[f'{session_types[0]}_matched_{activity_dataset}']
        comp_ds = data_dict[f'{session_types[1]}_matched_{activity_dataset}']

    # Drop NaN values
    ref_ds_ori = ref_ds.drop(ref_ds[ref_ds.osi.isna()].index).copy()
    ref_ds_dir = ref_ds.drop(ref_ds[ref_ds.dsi_abs.isna()].index).copy()

    comp_ds_ori = comp_ds.drop(comp_ds[comp_ds.osi.isna()].index).copy()
    comp_ds_dir = comp_ds.drop(comp_ds[comp_ds.dsi_abs.isna()].index).copy()

    # Get the shift in preferred orientation and direction
    ori_cutoff_high = 0.5
    dir_cutoff_high = 0.5
    lower_cutoff = 0.3

    po_shifts, ori_residuals, rmse_residual_ori = calculate_pref_angle_shifts(ref_ds.copy(), comp_ds.copy(),
                                                                              stim_kind='orientation',
                                                                              upper_cutoff=ori_cutoff_high)

    pd_shifts, dir_residuals, rmse_residual_dir = calculate_pref_angle_shifts(ref_ds.copy(), comp_ds.copy(),
                                                                              stim_kind='direction',
                                                                              upper_cutoff=dir_cutoff_high)

    # Plot the shift in preferred orientation
    scatter_PO = plot_delta_pref(po_shifts, rmse_residual_ori, session_types, type='orientation')

    save_path = os.path.join(figure_save_path, "delta_PO.png")
    scatter_PO = fp.save_figure(scatter_PO, save_path=save_path, fig_width=8, dpi=800, fontsize='paper',
                                target='save', display_factor=0.1)

    # Plot the shift in preferred direction
    scatter_PD = plot_delta_pref(pd_shifts, rmse_residual_dir, session_types, type='direction')

    save_path = os.path.join(figure_save_path, "delta_PD.png")
    scatter_PD = fp.save_figure(scatter_PD, save_path=save_path, fig_width=8, dpi=800, fontsize='paper',
                                target='save', display_factor=0.1)

    # Get the shifts in OSI/DSI
    osi_shifts, osi_residuals, rmse_residual_osi = calculate_delta_selectivity(ref_ds_ori.copy(), comp_ds_ori.copy(),
                                                                               cutoff=ori_cutoff_high,
                                                                               stim_kind='orientation')

    dsi_shifts, dsi_residuals, rmse_residual_dsi = calculate_delta_selectivity(ref_ds_dir.copy(), comp_ds_dir.copy(),
                                                                               cutoff=dir_cutoff_high,
                                                                               stim_kind='direction')

    # Plot the shift in preferred orientation
    scatter_OSI = plot_delta_pref(osi_shifts, rmse_residual_osi, session_types, type='orientation')
    scatter_OSI.opts(hv.opts.Scatter(
                        xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} OSI',
                        ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} OSI')
                    )
    save_path = os.path.join(figure_save_path, "delta_OSI.png")
    scatter_OSI = fp.save_figure(scatter_OSI, save_path=save_path, fig_width=8, dpi=800, fontsize='paper',
                                target='save', display_factor=0.1)

    # Plot the shift in preferred direction
    scatter_DSI = plot_delta_pref(dsi_shifts, rmse_residual_dsi, session_types, type='direction')
    scatter_DSI.opts(hv.opts.Scatter(
                        xlabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[0]]} DSI',
                        ylabel=f'{processing_parameters.wf_label_dictionary_wo_units[session_types[1]]} DSI')
                    )
    save_path = os.path.join(figure_save_path, "delta_DSI.png")
    scatter_DSI = fp.save_figure(scatter_DSI, save_path=save_path, fig_width=8, dpi=800, fontsize='paper',
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
                        tuning_dsi = ds['dsi_abs'].abs().to_numpy()
                        tuning_osi = ds['osi'].to_numpy()
                        umap_dict[f'dsi_{this_rig}'] = np.clip(tuning_dsi, 0, 1)
                        umap_dict[f'osi_{this_rig}'] = np.clip(tuning_osi, 0, 1)

            raw_tunings = pd.DataFrame.from_dict(umap_dict)
            raw_tunings = raw_tunings.fillna(0)

            tunings = preproc.StandardScaler().fit_transform(raw_tunings.to_numpy())

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
                umap_plot = fp.save_figure(umap_plot, save_path=save_name, fig_width=6, dpi=800, fontsize='paper',
                                           target='save', display_factor=0.1)

        else:
            # Here there may be uneven numbers of cells between sessions
            umap_dict_1 = {}
            umap_dict_2 = {}

            for label in labels:
                data_keys = [key for key in data_dict.keys() if label in key]

                for key in data_keys:
                    ds = data_dict[key]
                    base_label = '_'.join(label.split('_')[1 + len(cell_kind.split('_')):])

                    if base_label in kinem_label_list:
                        tuning = ds['Qual_index']

                        if base_label in processing_parameters.variable_list_free:
                            umap_dict_1[base_label] = tuning
                        else:
                            umap_dict_2[base_label] = tuning
                    else:
                        this_rig = key.split('_')[0]
                        tuning_dsi = ds['dsi_abs'].abs().to_numpy()
                        tuning_osi = ds['osi'].to_numpy()

                        if this_rig == 'VTuningWF':
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
                    umap_plot = fp.save_figure(umap_plot, save_path=save_name, fig_width=6, dpi=800, fontsize='paper',
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