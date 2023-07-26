import numpy as np
import pycircstat as circ
from scipy.stats import sem, norm, binned_statistic, percentileofscore, ttest_1samp, ttest_ind
from scipy.optimize import least_squares
import functions_kinematic as fk


### --- Gaussian fitting --- ###
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


### --- Tuning curve calculations --- ###
def calculate_pref_direction(angles, tuning_curve, **kwargs):
    # --- Fit double gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', np.max(curve_to_fit))
    width = kwargs.get('width', 25)
    center = angles[np.argmax(curve_to_fit)]


    # Following Carandini and Ferster, 2000, (https://doi.org/10.1523%2FJNEUROSCI.20-01-00470.2000)
    # initialize the double gaussian with the same width and amplitude, but shift the center of the second gaussian
    # if wrapping on [-180, 180] domain, use fk.wrap(center + center_shift, bound=180) - center_shift
    center2 = fk.wrap(center + 180)

    init_params = [amp, center, width, amp, center2, width]
    lower_bound = [0, 0, 10, 0, 0, 10]
    upper_bound = [1, 360, 40, 1, 360, 40]

    # Run regression
    fit = least_squares(fit_double_gaussian, init_params,
                        args=(angles, curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(angles.min(), angles.max(), 500, endpoint=True)
    fit_curve = double_gaussian(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    # Find the preferred orientation/direction from the fit
    pref = x[np.argmax(fit_curve)]

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, (x, fit_curve), pref, real_pref


def calculate_pref_orientation(angles, tuning_curve, **kwargs):
    # --- Fit gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', np.max(curve_to_fit))
    width = kwargs.get('width', 20)
    center = angles[np.argmax(curve_to_fit)]

    init_params = [amp, center, width]
    lower_bound = [0, 0, 10]
    upper_bound = [1, 180, 30]

    # Run regression
    fit = least_squares(fit_gaussian, init_params,
                        args=(angles, curve_to_fit),
                        bounds=(lower_bound, upper_bound),
                        loss='linear')

    # Generate a fit curve
    x = np.linspace(angles.min(), angles.max(), 500, endpoint=True)
    fit_curve = gaussian(x, *fit.x)
    fit_curve += tuning_curve.min()  # Shift baseline back to match real data

    # Find the preferred orientation/direction from the fit
    pref = x[np.argmax(fit_curve)]

    # Find the direction/orientation of the grating that was shown
    real_pref = angles[np.argmin(np.abs(angles - pref))]

    return fit, (x, fit_curve), pref, real_pref


def generate_response_vector(cell_responses, function, tuning_kind, **kwargs):
    # Get response across trials
    resp = cell_responses.groupby([tuning_kind]).apply(function, **kwargs)

    # Fill any NaNs with zeros
    resp = resp.fillna(0)
    
    # Get a list of directions/orientations
    angles = resp.index.to_numpy()

    # Wrap angles from [-180, 180] to [0, 360] for direction tuning, and sort angles and cell responses
    # Needed for pycircstat toolbox
    sorted_angles, sort_idx = wrap_sort(angles)
    sorted_resp = resp.values[sort_idx]

    if 'direction' in tuning_kind:
        # Make sure to explicitly represent 360 degrees in the data if looking at direction tuning. 
        # This is done by duplicating the 0 degrees value
        sorted_resp, sorted_angles = append_360(sorted_angles, sorted_resp)
        
    
    return sorted_resp, sorted_angles

def bootstrap_responsivity(trial_responses, tuning_kind, num_shuffles=1000):
    
    shuffled_responsivity = []
    
    # Get a list of directions/orientations
    angles = trial_responses.index.unique().to_numpy()
    
    trial_idxs = trial_responses.index.to_numpy()

    for i in range(0, num_shuffles):
        # Shuffle stimulus labels and reassign
        np.random.shuffle(trial_idxs)
        shuffled_responses = trial_responses.copy().reindex(trial_idxs)

        # Generate mean response vector from shuffled data
        shuffled_response_vector, angles = generate_response_vector(shuffled_responses, np.nanmean, tuning_kind)
        
         #-- Get response circular variance --#
        circ_var = get_circ_var(np.deg2rad(angles), shuffled_response_vector, tuning_kind)
        responsivity = 1 - circ_var
        shuffled_responsivity.append(responsivity)

    shuffled_responsivity = np.array(shuffled_responsivity)
    
    return shuffled_responsivity


def polar_vector_sum(magnitudes, directions):
    directions = np.deg2rad(directions)
    x = np.sum(magnitudes * np.cos(directions))
    y = np.sum(magnitudes * np.sin(directions))
    
    r = np.sqrt(x**2 + y**2)
    theta = np.rad2deg(np.arctan2(y, x))
    
    return r, theta

### --- For circular statistics --- ###

def get_circ_var(angles, tuning_weights, tuning_kind):
    
    # When calculating mean resultant, need to make orientation (bound from [0, pi]) bound from [0, 2pi]
    if tuning_kind == 'orientation':
        multiplier = 2
    else:
        multiplier = 1
        
    return circ.var(multiplier*angles, w=tuning_weights, d=multiplier*np.mean(np.diff(angles))) / multiplier


def get_resultant_vector(angles, tuning_weights, tuning_kind):
    
    # When calculating mean resultant, need to make orientation (bound from [0, pi]) bound from [0, 2pi]
    # See comment here: https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics
    if tuning_kind == 'orientation':
        multiplier = 2
    else:
        multiplier = 1
        
    mean_resultant = circ.moment(multiplier*angles,  w=tuning_weights, d=multiplier*np.mean(np.diff(angles)))
    angle = np.angle(mean_resultant) / multiplier
    length = np.absolute(mean_resultant)
    
    return length, angle

# def fit_von_mises(angles, magnitudes):
#     resultant_length, mean_angle = polar_vector_sum(magnitudes, angles)
#     kappa = circ.kappa(resultant_length)
#
#     x = np.deg2rad(np.linspace(angles.min(), angles.max(), 1000, endpoint=True))
#     mean, var = circ.distributions.vonmises.stats(kappa, moments='mv')
#     fit_curve = circ.distributions.vonmises.pdf(x, kappa, loc=np.deg2rad(mean_angle))
#     return fit_curve, mean, var


# def fit_von_mises2(angles, magnitudes):
#     if max(angles) > 2 * np.pi:
#         angles = np.deg2rad(angles)
#
#     kappa = circ.kappa(angles)
#
#     x = np.linspace(angles.min(), angles.max(), 1000, endpoint=True)
#
#     # Take a guess at the location of the peak using max value
#     loc_max = angles[np.argmax(magnitudes)]
#
#     r = circ.distributions.vonmises.rvs(kappa, loc=loc_max)
#     vonmises = circ.distributions.vonmises.fit()
#
#     return fit_curve, mean, var

### --- Misc. --- ###
def wrap_sort_negative(angles, values, bound=180):
    wrapped_angles = fk.wrap_negative(angles)
    sorted_index = np.argsort(wrapped_angles)
    sorted_values = values[sorted_index]
    sorted_wrapped_angles = wrapped_angles[sorted_index]
    return sorted_wrapped_angles, sorted_values

def wrap_sort(angles, bound=360):
    wrapped_angles = fk.wrap(angles)
    sort_idx = np.argsort(wrapped_angles)
    angles_sorted = wrapped_angles[sort_idx]
    return angles_sorted, sort_idx

def append_360(sorted_angles, data):
    idx_0 = np.argwhere(sorted_angles == 0)[0][0]
    trace_360 = data[idx_0]
    sorted_angles = np.append(sorted_angles, 360.)
    data_360 = np.append(data, [trace_360], axis=0)
    return data_360, sorted_angles