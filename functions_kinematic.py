import numpy as np
import scipy.stats as st
from functions_matching import interp_motive


def wrap(angles, bound=360.):
    """wrap angles to the range 0 to 360 deg"""
    # modified from https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    out_angles = angles % bound
    return out_angles


def wrap_negative(angles, bound=180.):
    """Wrap angles to the range [-180, 180]"""

    if isinstance(angles, (int, float)):
        if angles > bound:
            return (angles % bound) - bound
        else:
            return angles
    else:
        bound_excess_idx = np.argwhere(angles > bound)
        out_angles = angles.copy()
        out_angles[bound_excess_idx] = (angles[bound_excess_idx] % bound) - bound
        return out_angles

def unwrap(angles, discont=3.141592653589793, axis=0):
    """unwrap angles in degrees"""
    return np.rad2deg(np.unwrap(np.deg2rad(angles), discont=discont, axis=axis))


def heading_calculation(data_target, data_origin):
    """Calculate the heading angle of vectors, taking in the vector origin and the end"""
    heading_vector = np.rad2deg(np.array([np.arccos(point[0] / np.linalg.norm(point)) if point[1] > 0 else
                                - np.arccos(point[0] / np.linalg.norm(point)) for point in
                                (data_target - data_origin)]))
    return heading_vector


def reverse_heading(heading, data_origin, vector_length=0.1):
    """Calculate the vector end coordinates based on heading and vector origin"""
    data_target = np.array([[np.cos(angle)*vector_length, np.sin(angle)*vector_length] for angle in
                           np.deg2rad(heading)] + data_origin)
    return data_target


def bin_angles(angles, number_angles=36):
    """Bin angles for polar plots"""
    # get the angle bins
    angle_bins = np.arange(0, 360, 360 / (number_angles + 1))
    digitized_array = np.digitize(angles, angle_bins)
    # calculate the bin values
    binned_array = np.array([np.sum(digitized_array == el + 1) for el in np.arange(number_angles)])
    polar_coord = np.concatenate((angle_bins[:-1].reshape((-1, 1)), binned_array.reshape((-1, 1))), axis=1)
    return polar_coord


def distance_calculation(data_1, data_2):
    """Calculate the euclidean distance between corresponding points in the 2 input data sets"""
    return np.array([np.linalg.norm(point_1 - point_2) for point_1, point_2 in zip(data_1, data_2)])


def circmean_deg(data_in, axis=0):
    """Wrapper in degrees for the scipy circmean function"""
    return np.rad2deg(circmean(np.deg2rad(data_in), axis=axis))


def circstd_deg(data_in, axis=0):
    """Wrapper in degrees for the scipy circstd function"""
    return np.rad2deg(circstd(np.deg2rad(data_in), axis=axis))


def circmean(samples, high=2*np.pi, low=0, axis=None, ):
    samples, ang = _circfuncs_common(samples, high, low)
    S = np.nansum(np.sin(ang), axis=axis)
    C = np.nansum(np.cos(ang), axis=axis)
    res = np.arctan2(S, C)
    mask = res < 0
    if mask.ndim > 0:
        res[mask] += 2 * np.pi
    elif mask:
        res += 2 * np.pi
    return res * (high - low) / 2.0 / np.pi + low


def circstd(samples, high=2*np.pi, low=0, axis=None):
    samples, ang = _circfuncs_common(samples, high, low)
    S = np.mean(np.sin(ang), axis=axis)
    C = np.mean(np.cos(ang), axis=axis)
    R = np.hypot(S, C)
    return ((high - low)/2.0/np.pi) * np.sqrt(-2*np.log(R))


def _circfuncs_common(samples, high, low):
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.nan

    ang = (samples - low)*2.*np.pi / (high - low)
    return samples, ang


def jump_killer(data_in, jump_threshold, discont=3.141592653589793):
    # unwrap the trace
    data_in = unwrap(data_in, discont=discont)
    # id the large jumps
    smooth_map = np.concatenate(([1], np.abs(np.diff(data_in)) < jump_threshold)) == 1
    # generate a vector with indexes
    index_vector = np.array(np.arange(data_in.shape[0]))
    # use interp_motive to interpolate
    interp_points = interp_motive(data_in[smooth_map], index_vector[smooth_map], index_vector[~smooth_map])
    # replace the target points in the data
    data_out = data_in.copy()
    data_out[index_vector[~smooth_map]] = interp_points
    return data_out


def rotate_points(data_in, angles):
    """Rotate the given points by the given angle in degrees"""

    # allocate memory for the angles
    rotated_x = np.zeros_like(angles)
    rotated_y = np.zeros_like(angles)
    for idx, angle in enumerate(angles):
        rotated_x[idx] = data_in[idx, 0] * np.cos(np.deg2rad(angle)) - data_in[idx, 1] * np.sin(np.deg2rad(angle))
        rotated_y[idx] = data_in[idx, 0] * np.sin(np.deg2rad(angle)) - data_in[idx, 1] * np.cos(np.deg2rad(angle))

    return np.concatenate((rotated_x, rotated_y), axis=1)


def accumulated_distance(data_in):
    """
    Calculated the Euclidean distance between two consecutive locations
    :param data_in: Array in the form [samples, points]
    :return:
    """
    distance = np.zeros(len(data_in))
    distance[1:] = distance_calculation(data_in[1:, :], data_in[:-1, :])
    return distance


def smooth_trace(data, jump=25, kernel_size=5, range=(0, 360), discont=2*np.pi):
    jump_killed = jump_killer(data, jump, discont=discont)
    jump_killed[jump_killed > range[-1]] = range[-1]
    jump_killed[jump_killed < range[0]] = range[0]
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(jump_killed, kernel, mode='same')
    return smoothed