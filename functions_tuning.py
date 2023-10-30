import numpy as np
import pandas as pd
import pycircstat as circ
from scipy.special import i0
from scipy.optimize import least_squares, curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import functions_kinematic as fk
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
    # if wrapping on [-180, 180] domain, use fk.wrap(mean + mean_shift, bound=180) - mean_shift
    mean2 = fk.wrap(mean + 180)

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
    # if wrapping on [-180, 180] domain, use fk.wrap(mean + mean_shift, bound=180) - mean_shift
    mean2 = fk.wrap(mean + 180)

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

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    if fit_kind == 'orientation':
        # Duplicate the tuning curve to properly fit the double von mises
        curve_to_fit = np.tile(curve_to_fit, 2)
        angles = np.concatenate((angles, angles + 180))

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    kappa = kwargs.get('kappa', 5)
    mean = kwargs.get('mean', angles[np.argmax(curve_to_fit, axis=0)])

    # Following Carandini and Ferster, 2000, (https://doi.org/10.1523%2FJNEUROSCI.20-01-00470.2000)
    # initialize the double gaussian with the same width and amplitude, but shift the center of the second gaussian
    # if wrapping on [-180, 180] domain, use fk.wrap(center + center_shift, bound=180) - center_shift
    mean2 = fk.wrap(mean + 180)

    init_params = [amp, np.deg2rad(mean), kappa, amp, np.deg2rad(mean2), kappa]
    lower_bound = [0, 0, 1, 0, 0, 1]
    upper_bound = [2, 2*np.pi, 12, 2, 2*np.pi, 12]

    # Run regression
    fit = least_squares(fit_double_von_mises_pdf, init_params,
                        args=(np.deg2rad(angles), curve_to_fit),
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


def calculate_pref_orientation_vm(angles, tuning_curve, **kwargs):
    # --- Fit double gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Duplicate the tuning curve to properly fit the double von mises
    curve_to_fit = np.tile(curve_to_fit, 2)
    angles = angles.append(angles + 180)

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    kappa = kwargs.get('kappa', 5)
    mean = kwargs.get('mean', angles[np.argmax(curve_to_fit)])
    mean2 = fk.wrap(mean + 180)

    # init_params = [amp, np.deg2rad(mean), kappa]
    # lower_bound = [0, 0, 0]
    # upper_bound = [1, np.pi, 10]

    init_params = [amp, np.deg2rad(mean), amp, np.deg2rad(mean2)]
    lower_bound = [0, 0, 0, 0]
    upper_bound = [1, 2 * np.pi, 1, 2 * np.pi]

    # Run regression
    fit = least_squares(fit_double_von_mises_pdf, init_params,
                        args=(np.deg2rad(angles), curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(0, np.pi, 500, endpoint=True)
    fit_curve = von_mises(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    # Find the preferred orientation/direction from the fit
    pref = np.rad2deg(x[np.argmax(fit_curve)])

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, np.vstack((np.rad2deg(x), fit_curve)).T, pref, real_pref


def calculate_pref_direction_vm(angles, tuning_curve, **kwargs):
    # --- Fit double gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', min(1, np.max(curve_to_fit, axis=0)))
    kappa = kwargs.get('kappa', 5)
    mean = kwargs.get('mean', angles[np.argmax(curve_to_fit, axis=0)])

    # Following Carandini and Ferster, 2000, (https://doi.org/10.1523%2FJNEUROSCI.20-01-00470.2000)
    # initialize the double gaussian with the same width and amplitude, but shift the center of the second gaussian
    # if wrapping on [-180, 180] domain, use fk.wrap(center + center_shift, bound=180) - center_shift
    mean2 = fk.wrap(mean + 180)

    # init_params = [amp, np.deg2rad(mean), kappa, amp, np.deg2rad(mean2), kappa]
    # lower_bound = [0, 0, 0, 0, 0, 0]
    # upper_bound = [1, 2*np.pi, 10, 1, 2*np.pi, 10]

    init_params = [amp, np.deg2rad(mean), amp, np.deg2rad(mean2)]
    lower_bound = [0, 0, 0, 0]
    upper_bound = [1, 2 * np.pi, 1, 2 * np.pi]

    # Run regression
    fit = least_squares(fit_double_von_mises_pdf, init_params,
                        args=(np.deg2rad(angles), curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(0, 2 * np.pi, 500, endpoint=True)
    fit_curve = double_von_mises(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

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

    return resp, angles


def bootstrap_responsivity(angles, magnitudes, multiplier, num_shuffles=1000):
    
    shuffled_responsivity = []

    angle_sep = np.mean(np.diff(angles))
    
    for i in range(0, num_shuffles):
        # Shuffle stimulus labels and reassign
        np.random.shuffle(angles)
        
        # -- Get response circular variance -- #
        responsivity = circ.resultant_vector_length(angles, w=magnitudes, d=angle_sep, axial_correction=multiplier)
        shuffled_responsivity.append(responsivity)
        
    shuffled_responsivity = np.array(shuffled_responsivity)
    
    return shuffled_responsivity


def bootstrap_resultant(responses, multiplier, num_shuffles=1000):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell = columns[-1]

    # Get the counts per angle
    angle_counts = responses[tuning_kind].value_counts()
    unique_angles = angle_counts.index.to_numpy()
    min_presentations = np.min(angle_counts.to_numpy())
    trial_nums_by_angle = responses.groupby(tuning_kind).trial_num.agg(list)

    thetas = np.deg2rad(unique_angles)
    theta_sep = np.mean(np.diff(thetas))

    shuffled_resultant_angle = []
    shuffled_resultant_length = []
    for i in np.arange(num_shuffles):
        # select subset of trials, guaranteeing that each angle is represented the same number of times
        trial_subset = trial_nums_by_angle.apply(np.random.choice, size=min_presentations, replace=False).explode().to_numpy(dtype=int)
        theta_subset = responses[responses.trial_num.isin(trial_subset)][tuning_kind].apply(np.deg2rad).to_numpy()
        magnitude_subset = responses[responses.trial_num.isin(trial_subset)][cell].to_numpy()

        resultant_length = circ.resultant_vector_length(theta_subset, w=magnitude_subset, d=theta_sep, axial_correction=multiplier)
        resultant_angle = circ.mean(theta_subset, w=magnitude_subset, d=theta_sep, axial_correction=multiplier)
        resultant_angle = fk.wrap(np.rad2deg(resultant_angle), bound=360/multiplier)

        shuffled_resultant_length.append(resultant_length)
        shuffled_resultant_angle.append(resultant_angle)

    shuffled_resultant_length = np.array(shuffled_resultant_length)
    shuffled_resultant_angle = np.array(shuffled_resultant_angle)
    shuffled_resultant = np.vstack((shuffled_resultant_length, shuffled_resultant_angle)).T
    return shuffled_resultant


def bootstrap_tuning_curve(responses, fit_function, num_shuffles=100, gof_type='rmse', **kwargs):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell = columns[2]

    gof_list = []
    pref_angle_list = []
    real_pref_angle_list = []

    for i in np.arange(num_shuffles):
        test_trials = responses.groupby([tuning_kind])['trial_num'].apply(
            lambda x: np.random.choice(x, 2)).iloc[1:].to_list()

        test_trials = np.concatenate(test_trials)
        train_trials = np.setdiff1d(responses.trial_num.unique(), test_trials)

        test_set = responses[responses.trial_num.isin(test_trials)]
        train_set = responses[responses.trial_num.isin(train_trials)]

        test_mean, unique_angles = generate_response_vector(test_set, np.nanmean)
        train_mean, _ = generate_response_vector(train_set, np.nanmean)

        # Sometimes the data cannot be fit - catch that here
        try:
            fit, fit_curve, pref_angle, real_pref_angle = fit_function(unique_angles,
                                                                       train_mean[cell].to_numpy(),
                                                                       tuning_kind,
                                                                       **kwargs)

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


def polar_vector_sum(magnitudes, thetas):
    thetas = np.deg2rad(thetas)

    r = magnitudes * np.exp(1j * thetas)
    t = np.sum(r)

    mag = np.abs(t)
    angle = np.angle(t)
    
    return mag, angle


# --- DSI and OSI --- #
def calculate_dsi_osi_resultant(angles, magnitudes, bootstrap=False):
    # Note all angles must be in radians
    # This is only used on direction data, because it assumes that the angles are on the domain [0, 2pi]
    # DSI calculation from https://doi.org/10.1038/nature19818
    # OSI calculation from https://doi.org/10.1523/JNEUROSCI.0095-13.2013

    if bootstrap:
        unique_angles = np.unique(angles)
        tc = [np.nanmean(magnitudes[angles == angle]) for angle in unique_angles]
        angles = unique_angles
        magnitudes = np.array(tc)

    theta_sep = np.mean(np.diff(angles))

    # Get half periods
    half_period_angles_1 = angles[angles <= np.pi]
    half_period_mags_1 = magnitudes[angles <= np.pi]

    half_period_angles_2 = angles[angles >= np.pi] - np.pi
    half_period_mags_2 = magnitudes[angles >= np.pi]

    res_mag_1 = circ.resultant_vector_length(half_period_angles_1, w=half_period_mags_1, d=theta_sep,
                                             axial_correction=2)
    res_angle_1 = circ.mean(half_period_angles_1, w=half_period_mags_1, d=theta_sep, axial_correction=2)

    res_mag_2 = circ.resultant_vector_length(half_period_angles_2, w=half_period_mags_2, d=theta_sep,
                                             axial_correction=2)
    res_angle_2 = circ.mean(half_period_angles_2, w=half_period_mags_2, d=theta_sep, axial_correction=2)
    res_angle_2 = fk.wrap(res_angle_2, bound=np.pi) + np.pi

    if res_mag_1 > res_mag_2:
        pref = res_angle_1
        null = res_angle_2
        resultant_length = res_mag_1
    else:
        pref = res_angle_2
        null = res_angle_1
        resultant_length = res_mag_2

    closest_idx_to_pref = np.argmin(np.abs(angles - pref))
    resp_pref = magnitudes[closest_idx_to_pref]

    # for dsi
    closest_idx_to_null = np.argmin(np.abs(angles - fk.wrap(pref + np.pi, bound=2*np.pi)))
    resp_null = magnitudes[closest_idx_to_null]
    dsi_nasal_temporal = (resp_pref - resp_null) / (resp_pref + resp_null)
    dsi_abs = 1 - (resp_null/resp_pref)

    # for osi
    closest_idx_to_null_1 = np.argmin(np.abs(angles - fk.wrap(pref + np.pi/2, bound=2*np.pi)))
    closest_idx_to_null_2 = np.argmin(np.abs(angles - fk.wrap(pref - np.pi/2, bound=2*np.pi)))
    resp_null_1 = magnitudes[closest_idx_to_null_1]
    resp_null_2 = magnitudes[closest_idx_to_null_2]
    resp_null_osi = np.nanmean([resp_null_1, resp_null_2])
    osi = (resp_pref - resp_null_osi) / (resp_pref + resp_null_osi)
    
    return dsi_nasal_temporal, dsi_abs, osi, resultant_length, pref, null


def boostrap_dsi_osi_resultant(responses, sampling_method, min_trials=2, num_shuffles=1000):
    columns = list(responses.columns)
    tuning_kind = columns[0]
    cell = columns[-1]

    # Get the counts per angle
    trial_nums_by_angle = responses.groupby(tuning_kind).trial_num.agg(list)
    angle_counts = responses[tuning_kind].value_counts()
    min_presentations = angle_counts.min()
    if min_presentations < min_trials:
        if sampling_method == 'equal_trial_nums':
            print('Not enough presentations per angle to calculate DSI/OSI')
            return np.nan, np.nan, np.nan, np.nan, np.nan

    shuffled_null_angle = np.zeros(num_shuffles)
    shuffled_resultant = np.zeros((num_shuffles, 2))
    shuffled_dsi_nasal_temporal = np.zeros(num_shuffles)
    shuffled_dsi_abs = np.zeros(num_shuffles)
    shuffled_osi = np.zeros(num_shuffles)

    for i in np.arange(num_shuffles):

        if sampling_method == 'shuffle_trials':
            theta_subset = responses[tuning_kind].apply(np.deg2rad).to_numpy()
            np.random.shuffle(theta_subset)
            magnitude_subset = responses[cell].to_numpy()

        else:
            # select subset of trials, guaranteeing that each angle is represented the same number of times
            trial_subset = trial_nums_by_angle.apply(np.random.choice, size=min_presentations, replace=False).explode().to_numpy(dtype=int)
            theta_subset = responses[responses.trial_num.isin(trial_subset)][tuning_kind].apply(np.deg2rad).to_numpy()
            magnitude_subset = responses[responses.trial_num.isin(trial_subset)][cell].to_numpy()

        dsi_nasal_temporal, dsi_abs, osi, resultant_length, resultant_angle, null_angle = \
            calculate_dsi_osi_resultant(theta_subset, magnitude_subset, bootstrap=True)

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
        gof = r2_score(responses, np.array(pred_resp))
    else:
        raise NameError('Invalid goodness of fit type')

    return gof


def wrap_sort_negative(angles, values, bound=180):
    wrapped_angles = fk.wrap_negative(angles, bound=bound)
    sorted_index = np.argsort(wrapped_angles)
    sorted_values = values[sorted_index]
    sorted_wrapped_angles = wrapped_angles[sorted_index]
    return sorted_wrapped_angles, sorted_values


def wrap_sort(angles, bound=360):
    wrapped_angles = fk.wrap(angles, bound=bound)
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


def normalize_responses(ds, quantile=0.07):
    ds_norm = ds.copy().fillna(0)
    
    # get the number of cells
    # if type(ds) is pd.DataFrame:
    cells = [el for el in ds.columns if "cell" in el]
    # Normalize cell responses across all sessions
    ds_norm[cells] = ds_norm[cells].apply(normalize)

    # Get the 7th percentile of activity per cell for each stimulus
    # Try 7th/8th percentile
    percentiles = ds_norm.groupby(['direction'])[cells].quantile(quantile)

    # get the baselines - The first row is the inter-trial interval
    baselines = percentiles.iloc[0, :]

    # Subtract baseline from everything
    ds_norm[cells].subtract(baselines, axis=1)

    #     # TODO
    # elif type(ds) == xr.Dataset:
        
    #     cells = [el for el in ds.data_vars if "cell" in el]
    
    return ds_norm


def calculate_dff(ds, baseline_type='iti', inplace=True, **kwargs):
    
    cells = [el for el in ds.columns if "cell" in el]
    ds_copy = ds.copy()

    if baseline_type == 'iti':
        baselines = ds.loc[ds.trial_num == 0][cells].copy().mean()
    elif baseline_type == 'percentile':
        baselines = ds[cells].copy().apply(np.percentile, axis=1, args=(kwargs.get('percentile', 8)))
    
    if inplace:
        ds[cells] = ds[cells].subtract(baselines, axis=1).div(baselines, axis=1)
        return ds
    else:
        ds_copy[cells] = ds_copy[cells].subtract(baselines, axis=1).div(baselines, axis=1)
        return ds_copy