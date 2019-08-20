import numpy as np
from sklearn.linear_model import LinearRegression as ols


def rolling_average(data_in, window_size):
    return np.convolve(data_in, np.ones((window_size,))/window_size, mode='same')


def rolling_ols(data_in, window_size):
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
    close_idx = np.array([(np.argmin(np.abs(el-highres_array)), count)
                         for count, el in enumerate(lowres_array)])
    return close_idx


def add_edges(data_in, points=10):
    # calculate the average spacing between points
    average_interval = np.mean(np.diff(data_in))
    # use it to expand the original vector in both directions by points
    start_vector = np.arange(data_in[0]-average_interval*(points+1), data_in[0]-average_interval, average_interval)
    end_vector = np.arange(data_in[-1] + average_interval, data_in[-1] + average_interval * (points+1), average_interval)
    expanded_vector = np.concatenate((start_vector, data_in, end_vector))
    return expanded_vector
