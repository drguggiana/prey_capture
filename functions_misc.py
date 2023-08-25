import numpy as np
from sklearn.linear_model import LinearRegression as ols
from tkinter import Tk
from scipy.interpolate import interp1d
import cv2

# for slugify function
non_url_safe = ['"', '#', '$', '%', '&', '+',
                ',', '/', ':', ';', '=', '?',
                '@', '[', '\\', ']', '^', '`',
                '{', '|', '}', '~', "'"]
translate_table = {ord(char): u'' for char in non_url_safe}


def rolling_average(data_in, window_size):
    """Perform rolling average"""
    return np.convolve(data_in, np.ones((window_size,))/window_size, mode='same')


def rolling_ols(data_in, window_size):
    """Perform OLS regression on a rolling window"""
    # allocate memory for the output
    data_out = np.zeros_like(data_in)
    # get the subtraction to get to the bottom of the window
    window_bottom = np.int(np.floor(window_size / 2))

    for count in range(data_in[window_size:-window_size].shape[0]):
        # assemble the regression vector
        idx = count + window_bottom
        regression_vector = data_in[count:count+window_size]
        # fit the linear model
        linear_model = ols().fit(np.array(range(regression_vector.shape[0])).reshape(-1, 1), regression_vector)
        data_out[idx] = linear_model.coef_[0]
    # fill in the edges
    data_out = np.array(data_out)
    data_out[:window_size] = data_out[window_size]
    data_out[-window_size:] = data_out[-window_size]
    return data_out


def closest_point(highres_array, lowres_array):
    """Finds the closest point in a high resolution array to the ones in a low resolution one"""
    close_idx = np.array([(np.argmin(np.abs(el-highres_array)), count)
                         for count, el in enumerate(lowres_array)])
    return close_idx


def add_edges(data_in, points=10):
    """Adds interpolation edges to a time trace, using the average framerate as the interval and N points"""
    # calculate the average spacing between points
    average_interval = np.mean(np.diff(data_in))
    # use it to expand the original vector in both directions by points
    start_vector = np.arange(data_in[0]-average_interval*(points+1), data_in[0]-average_interval, average_interval)
    end_vector = np.arange(data_in[-1] + average_interval, data_in[-1] + average_interval * (points+1), average_interval)
    expanded_vector = np.concatenate((start_vector, data_in, end_vector))
    return expanded_vector


def error_logger(error_log, msg):
    """Logs errors and their arguments on a list and also prints them to the console"""
    error_log.append(msg + '\r\n')
    print(msg)
    return None


def tk_killwindow():
    """Prevents the appearance of the tk main window when using other GUI components"""
    # Create Tk root
    root = Tk()
    # Hide the main window
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    return None


def normalize_matrix(matrix, target=None, axis=None, background=None):
    """Normalize a matrix by the max and min, so to the range 0-1"""
    if axis is None:
        # normalize between 0 and 1
        out_matrix = (matrix - np.nanmin(matrix.flatten())) / (np.nanmax(matrix.flatten()) - np.nanmin(matrix.flatten()))
        # normalize to the range of a target matrix if provided
        if target is not None:
            out_matrix = out_matrix * (np.nanmax(target.flatten()) - np.nanmin(target.flatten())) + np.nanmin(
                target.flatten())
    else:
        assert target is None, "can't normalize to target when using a specific axis"
        if background is not None:
            # obtain the background trace along the given dimension
            background_signal = matrix.take(indices=range(background), axis=axis).mean(axis=axis).reshape(-1, 1)
            # calculate the background normalized trace
            out_matrix = (matrix - background_signal)/background_signal
        else:
            # normalize to 0-1 range along the desired dimension
            if axis == 0:
                out_matrix = (matrix - np.nanmin(matrix, axis=axis)) / (
                            np.nanmax(matrix, axis=axis) - np.nanmin(matrix, axis=axis))
            else:
                out_matrix = (matrix - np.nanmin(matrix, axis=axis).reshape(-1, 1)) / (
                        np.nanmax(matrix, axis=axis).reshape(-1, 1) - np.nanmin(matrix, axis=axis).reshape(-1, 1))
    return out_matrix


def interp_trace(y_known, x_known, x_target):
    """Interpolate a trace by building an interpolant"""
    # filter the values so the interpolant is trained only on sorted x points (required by the function)
    sorted_frames = np.hstack((True, np.invert(x_known[1:] <= x_known[:-1])))
    x_known = x_known[sorted_frames]
    # select frames based on the shape of the array
    if len(y_known.shape) > 1:
        y_known = y_known[sorted_frames, :]
        notnan = ~np.isnan(np.sum(y_known, axis=1))
        y_known = y_known[notnan, :].T
        axis = 1
    else:
        y_known = y_known[sorted_frames]
        notnan = ~np.isnan(y_known)
        y_known = y_known[notnan]
        axis = 0

    # also remove any NaN frames
    x_known = x_known[notnan]
    # create the interpolant
    interpolant = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value='extrapolate')
    return interpolant(x_target).T


def slugify(string_in):
    """Slugify the input string, taken from https://www.peterbe.com/plog/fastest-python-function-to-slugify-a-string"""
    string_out = string_in.translate(translate_table)
    string_out = u'_'.join(string_out.split()).lower()
    return string_out


def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def get_roi_stats(footprints):
    """Use opencv to extract the centroid, area and bounding box of an array of masks"""

    # allocate the output
    roi_info = []
    # for all the masks
    for roi in footprints:
        # binarize the image
        bin_roi = (roi > 0).astype(np.int8)
        # define the connectivity
        connectivity = 8
        # Perform the operation
        output = cv2.connectedComponentsWithStats(bin_roi, connectivity, cv2.CV_32S)
        # store the centroid x and y, the l, t, w, h of the bounding box and the area
        roi_info.append(np.hstack((output[3][1, :], output[2][1, :])))

    return np.vstack(roi_info)


def list_lists_to_array(list_of_lists):
    """ Converts a list of lists into a 2D array

    Parameters
    ----------
    list_of_lists : list

    Returns
    -------
    new_array : np.array
        Array where each row was an entry in the list of lists
    """

    max_length = max([len(sublist) for sublist in list_of_lists])
    new_array = np.empty((len(list_of_lists), max_length))
    new_array[:] = np.NaN

    for row, l in enumerate(list_of_lists):
        new_array[row, :len(l)] = l

    return new_array

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]