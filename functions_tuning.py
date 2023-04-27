import numpy as np
import pycircstat as circ
from scipy.stats import sem, norm, binned_statistic, percentileofscore, ttest_1samp, ttest_ind
from scipy.optimize import least_squares

import functions_kinematic as fk



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


def calculate_pref_direction(angles, tuning_curve, **kwargs):
    # --- Fit double gaussian ---#

    # Shift baseline to zero for fitting the data
    curve_to_fit = tuning_curve - tuning_curve.min()

    # Approximate parameters for the fit
    amp = kwargs.get('amplitude', np.max(curve_to_fit))
    center = angles[np.argmax(curve_to_fit)]
    width = kwargs.get('width', 45)

    # Following Carandini and Ferster, 2000, (https://doi.org/10.1523%2FJNEUROSCI.20-01-00470.2000)
    # initialize the double gaussian with the same with and amplitude, but shift the center of the second gaussian
    # if wrapping on [-180, 180] domain, use fk.wrap(center + center_shift, bound=180) - center_shift
    center2 = fk.wrap(center + 180)

    init_params = [amp, center, width, amp, center2, width]
    lower_bound = [0, 0, -np.inf, 0, 0, -np.inf]
    upper_bound = [1, 360, np.inf, 1, 360, np.inf]

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
    center = angles[np.argmax(curve_to_fit)]
    width = kwargs.get('width', 45)

    init_params = [amp, center, width]
    lower_bound = [0, 0, -np.inf]
    upper_bound = [1, 180, np.inf]

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