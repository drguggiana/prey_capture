import numpy as np
import pandas as pd
import pycircstat as circ
from scipy.special import i0
from scipy.optimize import least_squares, curve_fit
from sklearn.metrics import r2_score, mean_squared_error

import processing_parameters
from functions_kinematic import wrap, wrap_negative
from functions_misc import find_nearest


# --- Gaussian fitting --- #
def gaussian(x, c, mu, sigma):
    #     (c, mu, sigma) = params
    return c * np.exp(- (x - mu) ** 2.0 / (2.0 * sigma ** 2.0))


def fit_gaussian(params, x, y_data):
    fit = gaussian(x, *params)
    return fit - y_data


def double_gaussian(x, c1, mu1, sigma1, c2, mu2, sigma2):
    res = c1 * np.exp(-(x - mu1) ** 2.0 / (2.0 * sigma1 ** 2.0)) \
          + c2 * np.exp(-(x - mu2) ** 2.0 / (2.0 * sigma2 ** 2.0))
    return res


def fit_double_gaussian(params, x, y_data):
    fit = double_gaussian(x, *params)
    return fit - y_data


def get_FWHM(sigma):
    return 2 * np.sqrt(2 * np.log(2)) * np.abs(sigma)


def get_HWHM(sigma):
    return np.sqrt(2 * np.log(2)) * np.abs(sigma)


# --- Von Mises fitting --- #
def von_mises(theta, a, mu, kappa):
    """
        pdf_von_Mises(theta,mu,kappa)
        =============================

        Calculates the von Mises probability density distribution at the angle theta with mean
        direction mu and concentration kappa.

        INPUT:

            * theta - angle at which to evaluate the von Mises distribution (float or numpy array)
            * mu - mean direction (float)
            * kappa - concentration (float)

        OUTPUT:

             * pdf - the probability density function is an Nx1 (same size as theta) array of values of a von Mises
             distribution with mean direction mu and concentration kappa.

        References:
        ===========

        See the following textbook/monograph

        [1] N. I. Fisher, Statistical analysis of circular data, Cambridge University Press, (1993).

    """

    pdf = a * np.exp(kappa * np.cos(theta - mu)) / (2.0 * np.pi * i0(kappa))

    return pdf


def fit_von_mises_pdf(params, x, y_data):
    fit = von_mises(x, *params)
    return fit - y_data


def double_von_mises(theta, a1, mu1, kappa1, a2, mu2, kappa2):
    pdf = a1 * np.exp(kappa1 * np.cos(theta - mu1)) / (2.0 * np.pi * i0(kappa1)) \
        + a2 * np.exp(kappa2 * np.cos(theta - mu2)) / (2.0 * np.pi * i0(kappa2))
    
    return pdf


def fit_double_von_mises_pdf(params, x, y_data):
    fit = double_von_mises(x, *params)
    return fit - y_data


# --- Tuning curve calculations --- #
def calculate_pref_gaussian(angles, tuning_curve, fit_kind, **kwargs):
    # --- Fit double gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    if fit_kind == 'orientation':
        # Duplicate the tuning curve to properly fit the double von mises
        curve_to_fit = np.tile(curve_to_fit, 2)
        angles = np.concatenate((angles, angles + 180))

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    width = kwargs.get('width', 25)
    mean = kwargs.get('mean', angles[np.argmax(curve_to_fit)])

    # Following Carandini and Ferster, 2000, (https://doi.org/10.1523%2FJNEUROSCI.20-01-00470.2000)
    # initialize the double gaussian with the same width and amplitude, but shift the mean of the second gaussian
    # if wrapping on [-180, 180] domain, use wrap(mean + mean_shift, bound=180) - mean_shift
    mean2 = wrap(mean + 180)

    init_params = [amp, mean, width, amp, mean2, width]
    lower_bound = [0, 0, 10, 0, 0, 10]
    upper_bound = [1, 360, 40, 1, 360, 40]

    # Run regression
    fit = least_squares(fit_double_gaussian, init_params,
                        args=(angles, curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(0, 360, 500, endpoint=True)
    fit_curve = double_gaussian(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    if fit_kind == 'orientation':
        # Cut the fit curve in half
        x = x[:len(x) // 2]
        fit_curve = fit_curve[:len(fit_curve) // 2]

    # Find the preferred orientation/direction from the fit
    pref = x[np.argmax(fit_curve)]

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, np.vstack((x, fit_curve)).T, pref, real_pref


def calculate_pref_direction(angles, tuning_curve, **kwargs):
    # --- Fit double gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    width = kwargs.get('width', 25)
    mean = kwargs.get('mean', angles[np.argmax(curve_to_fit)])


    # Following Carandini and Ferster, 2000, (https://doi.org/10.1523%2FJNEUROSCI.20-01-00470.2000)
    # initialize the double gaussian with the same width and amplitude, but shift the mean of the second gaussian
    # if wrapping on [-180, 180] domain, use wrap(mean + mean_shift, bound=180) - mean_shift
    mean2 = wrap(mean + 180)

    # init_params = [amp, mean, width, amp, mean2, width]
    # lower_bound = [0, 0, 10, 0, 0, 10]
    # upper_bound = [1, 360, 40, 1, 360, 40]

    init_params = [amp, mean, amp, mean2]
    lower_bound = [0, 0, 0, 0]
    upper_bound = [1, 360, 1, 360]

    # Run regression
    fit = least_squares(fit_double_gaussian, init_params,
                        args=(angles, curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(0, 360, 500, endpoint=True)
    fit_curve = double_gaussian(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    # Find the preferred orientation/direction from the fit
    pref = x[np.argmax(fit_curve)]

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, np.vstack((x, fit_curve)).T, pref, real_pref


def calculate_pref_orientation(angles, tuning_curve, **kwargs):
    # --- Fit gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    width = kwargs.get('width', 16)
    mean = kwargs.get('mean', angles[np.argmax(curve_to_fit)])

    # init_params = [amp, mean, width]
    # lower_bound = [0, 0, 10]
    # upper_bound = [1, 180, 30]

    init_params = [amp, mean]
    lower_bound = [0, 0]
    upper_bound = [1, 180]

    # Run regression
    fit = least_squares(fit_gaussian, init_params,
                        args=(angles, curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(0, 180, 500, endpoint=True)
    fit_curve = gaussian(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    # Find the preferred orientation/direction from the fit
    pref = x[np.argmax(fit_curve)]

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, np.vstack((x, fit_curve)).T, pref, real_pref


def calculate_pref_von_mises(angles, tuning_curve, fit_kind, **kwargs):
    # --- Fit double gaussian ---#

    # Get angles in radians
    rads = np.deg2rad(angles)

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    if fit_kind == 'orientation':
        # Duplicate the tuning curve to properly fit the double von mises
        curve_to_fit = np.tile(curve_to_fit, 2)
        rads = np.concatenate((rads, rads + np.pi))

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    kappa = kwargs.get('kappa', 5)
    mean = kwargs.get('mean', rads[np.argmax(curve_to_fit, axis=0)])
    mean2 = wrap(mean + np.pi, bound=2*np.pi)

    init_params = [amp, mean, kappa, amp, mean2, kappa]
    lower_bound = [0, 0, 1, 0, 0, 1]
    upper_bound = [2, 2*np.pi, 12, 2, 2*np.pi, 12]

    # Run regression
    fit = least_squares(fit_double_von_mises_pdf, init_params,
                        args=(rads, curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(0, 2 * np.pi, 500, endpoint=True)
    fit_curve = double_von_mises(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    if fit_kind == 'orientation':
        # Cut the fit curve in half
        x = x[:len(x) // 2]
        fit_curve = fit_curve[:len(fit_curve) // 2]

    # Find the preferred orientation/direction from the fit
    pref = np.rad2deg(x[np.argmax(fit_curve)])

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, np.vstack((np.rad2deg(x), fit_curve)).T, pref, real_pref


def generate_response_vector(responses, function, **kwargs):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell_ids = columns[2:]

    # Get response across trials
    resp = responses.groupby([tuning_kind])[cell_ids].agg(function, **kwargs).fillna(0).reset_index()
    angles = resp[tuning_kind].to_numpy()
    resp = resp[cell_ids]
    resp.insert(0, tuning_kind, angles)

    return resp, angles


def resultant_vector(angles, magnitudes, axial_correction):
    theta_sep = np.mean(np.diff(np.unique(angles)))
    mag = circ.resultant_vector_length(angles, w=magnitudes, d=theta_sep, axial_correction=axial_correction)
    angle = circ.mean(angles, w=magnitudes, d=theta_sep, axial_correction=axial_correction)

    return mag, angle


def subsample_responses(trial_nums_by_angle, min_trials=4, replace=True):
    # select subset of trials, guaranteeing that each angle is represented the same number of times
    trial_subset = trial_nums_by_angle.apply(np.random.choice, size=min_trials, replace=replace).explode().to_numpy(
        dtype=int)
    return trial_subset


def bootstrap_resultant_orientation(responses, multiplier, sampling_method, min_trials=3, num_shuffles=1000):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell = columns[-1]

    shuffled_resultant = np.zeros((num_shuffles, 2))

    # Get the counts per angle
    trial_nums_by_angle = responses.groupby(tuning_kind).trial_num.agg(list)
    angle_counts = responses[tuning_kind].value_counts()
    min_presentations = angle_counts.min()
    if min_presentations < min_trials:
        if sampling_method == 'equal_trial_nums':
            shuffled_resultant.fill(np.nan)
            print('Not enough presentations per angle to calculate resultant')

    else:
        for i in np.arange(num_shuffles):
            if sampling_method == 'shuffle_trials':
                theta_subset = responses[tuning_kind].apply(np.deg2rad).to_numpy()
                np.random.shuffle(theta_subset)
                magnitude_subset = responses[cell].to_numpy()

            else:
                # select subset of trials, guaranteeing that each angle is represented the same number of times
                trial_subset = subsample_responses(trial_nums_by_angle, min_trials=min_presentations)
                theta_subset = responses[responses.trial_num.isin(trial_subset)][tuning_kind].apply(np.deg2rad).to_numpy()
                magnitude_subset = responses[responses.trial_num.isin(trial_subset)][cell].to_numpy()

            resultant_length, resultant_angle = resultant_vector(theta_subset, magnitude_subset, multiplier)
            # Need this correction because pycircstat does mod2pi by default
            resultant_angle = wrap(resultant_angle, bound=np.pi)
            resultant_angle = wrap(np.rad2deg(resultant_angle), bound=180.)

            shuffled_resultant[i, 0] = resultant_length
            shuffled_resultant[i, 1] = resultant_angle

    return shuffled_resultant


def bootstrap_tuning_curve(responses, fit_function, num_shuffles=100, gof_type='rmse', **kwargs):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell = columns[2]

    gof_list = []
    pref_angle_list = []
    real_pref_angle_list = []

    for i in np.arange(num_shuffles):
        # TODO check this
        test_trials = responses.groupby([tuning_kind])['trial_num'].apply(
            lambda x: np.random.choice(x, 2, replace=True)).iloc[1:].to_list()

        test_trials = np.concatenate(test_trials)
        train_trials = np.setdiff1d(responses.trial_num.unique(), test_trials)

        test_set = responses[responses.trial_num.isin(test_trials)]
        train_set = responses[responses.trial_num.isin(train_trials)]

        test_mean, unique_angles = generate_response_vector(test_set, np.nanmean)
        train_mean, _ = generate_response_vector(train_set, np.nanmean)

        # Sometimes the data cannot be fit - catch that here
        try:
            fit, fit_curve, pref_angle, real_pref_angle = \
                fit_function(unique_angles, train_mean[cell].to_numpy(), tuning_kind,  **kwargs)

            gof = goodness_of_fit(test_set[tuning_kind].to_numpy(), test_set[cell].to_numpy(),
                                  fit_curve[:, 0], fit_curve[:, 1], type=gof_type)
        except ValueError:
            gof = np.nan
            pref_angle = np.nan
            real_pref_angle = np.nan

        gof_list.append(gof)
        pref_angle_list.append(pref_angle)
        real_pref_angle_list.append(real_pref_angle)

    return np.array(gof_list), np.array(pref_angle_list), np.array(real_pref_angle_list)


# --- DSI and OSI --- #
def calculate_dsi_osi_resultant(angles, magnitudes, bootstrap=False):
    # Note all angles must be in radians
    # This is only used on direction data, because it assumes that the angles are on the domain [0, 2pi]
    # DSI calculation from https://doi.org/10.1038/nature19818
    # OSI calculation from https://doi.org/10.1523/JNEUROSCI.0095-13.2013

    # First check if the magnitudes are all > 0
    if np.all(magnitudes == 0):
        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN

    # IF we're bootstrapping, we need to get the unique angles and their mean magnitudes
    if bootstrap:
        unique_angles = np.unique(angles)
        tc = [np.nanmean(magnitudes[angles == angle]) for angle in unique_angles]
        angles = unique_angles
        magnitudes = np.array(tc)

    # Get resultant on first half of the data
    half_period_angles_1 = angles[angles <= np.pi]
    half_period_mags_1 = magnitudes[angles <= np.pi]

    res_mag_1, res_angle_1 = resultant_vector(half_period_angles_1, half_period_mags_1, 1)
    res_angle_1 = wrap(res_angle_1, bound=np.pi)     # Need this correction because pycircstat does mod2pi by default
    closest_idx1 = np.argmin(np.abs(angles - res_angle_1))
    resp1 = magnitudes[closest_idx1]
    angle1 = angles[closest_idx1]

    # Get resultant on second half of the data
    half_period_angles_2 = wrap(angles[angles >= np.pi], bound=np.deg2rad(180.1))
    half_period_angles_2[0] = 0.0
    half_period_mags_2 = magnitudes[angles >= np.pi]

    res_mag_2, res_angle_2 = resultant_vector(half_period_angles_2, half_period_mags_2, 1)
    res_angle_2 = wrap(res_angle_2, bound=np.pi)
    res_angle_2 += np.pi
    closest_idx2 = np.argmin(np.abs(angles - res_angle_2))
    resp2 = magnitudes[closest_idx2]
    angle2 = angles[closest_idx2]

    if resp1 > resp2:
        pref = res_angle_1
        resp_pref = resp1
        null = res_angle_2
        resultant_length = res_mag_1
    else:
        pref = res_angle_2
        resp_pref = resp2
        null = res_angle_1
        resultant_length = res_mag_2

    # for dsi
    closest_idx_to_null = np.argmin(np.abs(angles - wrap(pref + np.pi, bound=2*np.pi)))
    resp_null = magnitudes[closest_idx_to_null-1:closest_idx_to_null+2].mean()

    if resp_pref + resp_null == 0:
        dsi_nasal_temporal = 0
    else:
        dsi_nasal_temporal = (resp_pref - resp_null) / (resp_pref + resp_null)

    if resp_pref == 0:
        dsi_abs = 0
    else:
        dsi_abs = 1 - (resp_null/resp_pref)

    # for osi
    resp_pref_osi = np.nanmean([resp_pref, resp_null])
    closest_idx_to_null_1 = np.argmin(np.abs(angles - wrap(pref + np.pi/2, bound=2*np.pi)))
    closest_idx_to_null_2 = np.argmin(np.abs(angles - wrap(pref - np.pi/2, bound=2*np.pi)))
    resp_null_1 = magnitudes[closest_idx_to_null_1]
    resp_null_2 = magnitudes[closest_idx_to_null_2]
    resp_null_osi = np.nanmean([resp_null_1, resp_null_2])

    if resp_pref_osi + resp_null_osi == 0:
        osi = 0
    else:
        osi = (resp_pref_osi - resp_null_osi) / (resp_pref_osi + resp_null_osi)
    
    return dsi_nasal_temporal, dsi_abs, osi, resultant_length, pref, null


def boostrap_dsi_osi_resultant(responses, sampling_method, min_trials=3, num_shuffles=1000):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell = columns[-1]

    shuffled_null_angle = np.zeros(num_shuffles)
    shuffled_resultant = np.zeros((num_shuffles, 2))
    shuffled_dsi_nasal_temporal = np.zeros(num_shuffles)
    shuffled_dsi_abs = np.zeros(num_shuffles)
    shuffled_osi = np.zeros(num_shuffles)

    # Get the counts per angle
    trial_nums_by_angle = responses.groupby(tuning_kind).trial_num.agg(list)
    angle_counts = responses[tuning_kind].value_counts()
    min_presentations = int(angle_counts.min())
    if (min_presentations < min_trials) and (sampling_method == 'equal_trial_nums'):
        shuffled_dsi_nasal_temporal.fill(np.nan)
        shuffled_dsi_abs.fill(np.nan)
        shuffled_osi.fill(np.nan)
        shuffled_resultant.fill(np.nan)
        shuffled_null_angle.fill(np.nan)
        print('Not enough presentations per angle to calculate DSI/OSI')

    else:
        for i in np.arange(num_shuffles):

            if sampling_method == 'shuffle_trials':
                theta_subset = responses[tuning_kind].apply(np.deg2rad).to_numpy()
                np.random.shuffle(theta_subset)
                magnitude_subset = responses[cell].to_numpy()

            else:
                # select subset of trials, guaranteeing that each angle is represented the same number of times
                trial_subset = subsample_responses(trial_nums_by_angle, min_trials=min_presentations)
                theta_subset = responses[responses.trial_num.isin(trial_subset)][tuning_kind].apply(np.deg2rad).to_numpy()
                magnitude_subset = responses[responses.trial_num.isin(trial_subset)][cell].to_numpy()

            dsi_nasal_temporal, dsi_abs, osi, resultant_length, resultant_angle, null_angle = \
                calculate_dsi_osi_resultant(theta_subset, magnitude_subset, bootstrap=True)

            # check if any of the outputs is None
            if any([el is None for el in [dsi_nasal_temporal, dsi_abs, osi, resultant_length, resultant_angle, null_angle]]):
                print('hi')
                # shuffled_dsi_nasal_temporal[i] = np.nan
                # shuffled_dsi_abs[i] = np.nan
                # shuffled_osi[i] = np.nan
                # shuffled_resultant[i, :] = np.nan
                # shuffled_null_angle[i] = np.nan

            shuffled_dsi_nasal_temporal[i] = dsi_nasal_temporal
            shuffled_dsi_abs[i] = dsi_abs
            shuffled_osi[i] = osi
            shuffled_resultant[i, 0] = resultant_length
            shuffled_resultant[i, 1] = np.rad2deg(resultant_angle)
            shuffled_null_angle[i] = np.rad2deg(null_angle)

    return shuffled_dsi_nasal_temporal, shuffled_dsi_abs, shuffled_osi, shuffled_resultant, shuffled_null_angle


# --- Misc. --- #
def goodness_of_fit(angles, responses, fit_angles, fit_values, type='rmse'):
    pred_resp = []
    for angle in angles:
        i, _ = find_nearest(fit_angles, angle)
        pred_resp.append(fit_values[i])
    pred_resp = np.array(pred_resp)

    if type == 'rmse':
        gof = mean_squared_error(responses, pred_resp, squared=False)
    elif type == 'mse':
        gof = mean_squared_error(responses, pred_resp)
    elif type == 'r2':
        gof = r2_score(responses, pred_resp)
    else:
        raise NameError('Invalid goodness of fit type')

    return gof


def wrap_sort_negative(angles, values, bound=180):
    wrapped_angles = wrap_negative(angles, bound=bound)
    sorted_index = np.argsort(wrapped_angles)
    sorted_values = values[sorted_index]
    sorted_wrapped_angles = wrapped_angles[sorted_index]
    return sorted_wrapped_angles, sorted_values


def wrap_sort(angles, bound=360):
    wrapped_angles = wrap(angles, bound=bound)
    sort_idx = np.argsort(wrapped_angles)
    angles_sorted = wrapped_angles[sort_idx]
    return angles_sorted, sort_idx


def append_angle(sorted_angles, data, angle=360.):
    """Duplicates the 0 degree angle and appends it to the end of the sorted
    angles and data arrays to have a continuous tuning curve
    """
    
    idx_0 = np.argwhere(sorted_angles == 0)[0][0]
    trace_append = data[idx_0]
    sorted_angles = np.append(sorted_angles, angle)
    data_360 = np.append(data, [trace_append], axis=0)
    return data_360, sorted_angles


def moving_average(data, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')


def normalize_rows(data_in):
    for idx, el in enumerate(data_in):
        data_in[idx, :] = (el-np.nanmin(el))/(np.nanmax(el)-np.nanmin(el))
    return data_in


def normalize(data_in):
    if np.nanmax(data_in) == 0:
        return data_in
    else:
        return (data_in - np.nanmin(data_in)) / (np.nanmax(data_in) - np.nanmin(data_in))


def normalize_responses(ds, remove_baseline=False, quantile=0.07, columnwise=True):

    ds_norm = ds.copy().fillna(0)
    
    # get the number of cells
    cells = [el for el in ds.columns if "cell" in el]

    # Normalize cell responses across all sessions
    if columnwise:
        ds_norm[cells] = ds_norm[cells].apply(normalize, raw=True)
    else:
        ds_norm[cells] = normalize(ds_norm[cells].to_numpy())

    if remove_baseline:
        # Get the 7th percentile of activity per cell for each stimulus
        # Try 7th/8th percentile
        percentiles = ds_norm.groupby(['direction'])[cells].quantile(quantile)

        # get the baselines - The first row is the inter-trial interval
        baselines = percentiles.iloc[0, :]

        # Subtract baseline from everything
        ds_norm[cells] = ds_norm.loc[:, cells].subtract(baselines, axis=1)
    
    return ds_norm


def calculate_dff(ds, baseline_type='iti_mean', zero_baseline=False, **kwargs):
    
    cells = [el for el in ds.columns if "cell" in el]
    ds[cells] = ds[cells].apply(np.nan_to_num, axis=0, raw=True, nan=0, posinf=0, neginf=0)
    dff = ds.copy()

    if baseline_type == 'iti_mean':
        baselines = ds[ds.trial_num == 0][cells].mean(axis=0).copy()

    elif baseline_type == 'quantile':
        qaunt = kwargs.get('quantile', 0.20)
        baselines = ds[cells].quantile(qaunt, numeric_only=True, axis=0).copy()

    else:
        raise NameError('Invalid baseline type')
    
    # Where the quantile is zero, replace with the mean baseline of all nonzero cells
    baselines[baselines == 0] = baselines[baselines > 0].mean()
    
    dff[cells] -= baselines
    dff[cells] /= baselines

    # Shift so that there are no non-zero values
    if zero_baseline:
        dff[cells] -= dff[cells].min(axis=0)

    return dff


def parse_trial_frames(df, pre_trial=0, post_trial=0):

    trial_idx_frames = df[df.trial_num >= 1.0].groupby(['trial_num']).apply(
        lambda x: [x.index[0] - int(pre_trial * processing_parameters.wf_frame_rate),
                   x.index[0], x.index[-1],
                   x.index[-1] + int(post_trial * processing_parameters.wf_frame_rate)]
                ).to_numpy()

    trial_idx_frames = np.vstack(trial_idx_frames)

    if trial_idx_frames[0, 0] < df.index[0]:
        trial_idx_frames[0, 0] = df.index[0]

    if trial_idx_frames[-1, -1] > df.index[-1]:
        trial_idx_frames[-1, -1] = df.index[-1]

    if trial_idx_frames[-1, -2] == trial_idx_frames[-1, -1]:
        trial_idx_frames[-1, -2:] = int(df.index[-1]) - 1

    traces = []
    for i, frame in enumerate(trial_idx_frames):
        df_slice = df.iloc[frame[0]:frame[-1], :].copy()
        df_slice['frame_num'] = df_slice.trial_num.max()
        traces.append(df_slice)

    traces = pd.concat(traces, axis=0).reset_index(drop=True)
    return traces, trial_idx_frames


def calculate_dsi_osi_fit(angles, magnitudes, pref):
    # DSI with directions
    dir_pref_idx = np.argwhere(angles == pref).squeeze()
    dir_null_idx = np.argmin(np.abs(angles - wrap(pref + 180, bound=360.))).squeeze()
    mag_dir_pref = magnitudes[dir_pref_idx]
    mag_dir_null = magnitudes[dir_null_idx]
    dsi = 1 - (mag_dir_null / mag_dir_pref)

    # OSI from direction data
    mean_mag_ori_pref = (mag_dir_pref + mag_dir_null) / 2
    null_idxs_ori_1 = np.argmin(np.abs(angles - wrap(pref + 90, bound=360.))).squeeze()
    null_idxs_ori_2 = np.argmin(np.abs(angles - wrap(pref - 90, bound=360.))).squeeze()
    mag_ori_1 = magnitudes[null_idxs_ori_1]
    mag_ori_2 = magnitudes[null_idxs_ori_2]
    mean_mag_ori_null = (mag_ori_1 + mag_ori_2) / 2
    osi = 1 - (mean_mag_ori_null / mean_mag_ori_pref)

    return dsi, osi
