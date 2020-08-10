import numpy as np
from scipy.ndimage.measurements import label
import datetime
from scipy.signal import medfilt
from functions_misc import interp_trace
import cv2
import functions_plotting as fp
import matplotlib.pyplot as plt
from quicksect import IntervalNode, Interval
from functools import reduce
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances


def remove_nans(files, tar_columns):
    """Basic function for the removal of NaNs"""
    # allocate memory for a per file list
    perfile_array = np.zeros_like(files)

    # for the mouse and the cricket columns
    for idx, animal in enumerate([tar_columns[1]]):
        # allocate a list to store the marked traces
        marked_traces = []
        col = animal[0]
        # get the data
        result = files[:, col].copy()

        # remove the NaN regions
        result[np.isnan(result) == 0] = 0
        result[np.isnan(result)] = 1

        # label the NaN positions
        result, number_regions = label(result)

        # run through the columns again
        for idx2, col in enumerate(animal):

            # copy the original vector
            target_vector = files[:, col]
            # for all the regions
            for segment in range(2, number_regions + 1):
                # get the edges of the labeled region
                indexes = np.nonzero(result == segment)[0]
                start = indexes[0] - 1
                end = indexes[-1] + 1
                # skip the segment if the end of the segment is also the end of the whole trace
                if end == target_vector.shape:
                    continue

                target_vector[result == segment] = np.interp(indexes, [start, end], target_vector[[start, end]])
            perfile_array[:, idx * idx2 + idx2] = target_vector

    return perfile_array


def trim_bounds(files, time, dates):
    """Based on rig specific heuristics, trim the trace"""
    # # get the time
    # time = [datetime.datetime.strptime(el[1][:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in files]
    # TODO: remove arbitrary thresholds
    # define the arena left boundary (based on the date of the experiment to be retrocompatible)
    if dates[0] > datetime.datetime(year=2019, month=11, day=10):
        left_bound_mouse = 0
        left_bound_cricket = 0
    else:
        left_bound_mouse = 110
        left_bound_cricket = 100

    # time = [(el - time[0]).total_seconds() for el in time]  # print the frame rate
    # print('Frame rate:' + str(1 / np.mean(np.diff(time))) + 'fps')

    # # get just the coordinate data
    # files = np.vstack(np.array([el[0] for el in files]))
    # eliminate any pieces of the trace until the mouse track doesn't have NaNs
    if sum(np.isnan(files[:, 0])) > 0:
        nan_pointer = np.max(np.argwhere(np.isnan(files[:, 0])))
        files = files[nan_pointer + 1:, :]
        time = time[nan_pointer + 1:]
    # eliminate any remaining pieces captured outside the arena
    if sum(files[:, 0] < left_bound_mouse) > 0:
        nan_pointer = np.max(np.argwhere(files[:, 0] < left_bound_mouse))
        files = files[nan_pointer + 1:, :]
        time = time[nan_pointer + 1:]
    # same as above with the cricket
    if sum(files[:, 2] < left_bound_cricket) > 0:
        nan_pointer = np.max(np.argwhere(files[:, 2] < left_bound_cricket))
        files = files[nan_pointer + 1:, :]
        time = time[nan_pointer + 1:]

    return files, time


def get_time(files):
    """Separate the Bonsai time and data"""
    # get the time
    dates = [datetime.datetime.strptime(el[1][:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in files]
    # convert to seconds
    time = [(el - dates[0]).total_seconds() for el in dates]
    # print the frame rate
    print('Frame rate:' + str(1 / np.mean(np.diff(time))) + 'fps')
    # get just the coordinate data
    files = np.vstack(np.array([el[0] for el in files]))

    return files, time, dates


def remove_discontinuities(files, tar_columns, max_step_euc):
    """Remove discontinuities in the trace via interpolation"""
    # for the mouse and the cricket columns
    for animal in tar_columns:
        # allocate a list to store the marked traces
        marked_traces = []

        for idx, col in enumerate(animal):
            # get the data
            curr_data = files[:, col].copy()

            # take the absolute derivative trace
            result = np.absolute(np.diff(curr_data[:]))

            result = np.hstack((0, result))

            # append the result to the list
            marked_traces.append(result)

        # combine the marked traces and rebinarize
        pre_result = []
        for idx, el in enumerate(marked_traces[0]):
            current_xy = np.array([el, marked_traces[1][idx]])
            pre_result.append(np.sqrt(current_xy[0] ** 2 + current_xy[1] ** 2))

        result = np.array(pre_result)
        # kill the remaining NaNs
        result[np.isnan(result)] = 0

        result[result < max_step_euc] = 0

        # label the relevant regions
        result, number_regions = label(result)

        # run through the columns again
        for col in animal:

            # copy the original vector
            target_vector = files[:, col]
            # for all the regions
            for segment in range(2, number_regions + 1):
                # get the edges of the labeled region
                indexes = np.nonzero(result == segment)[0]
                start = indexes[0] - 1
                end = indexes[-1] + 1
                # skip the segment if the end of the segment is also the end of the whole trace
                if end == target_vector.shape:
                    continue

                target_vector[result == segment] = np.interp(indexes, [start, end],
                                                             target_vector[[start, end]])
    return target_vector


def median_discontinuities(files, tar_columns, kernel_size):
    """Use a median filter to remove discontinuities in the trace"""
    # allocate memory for the output
    filtered_traces = files.copy()
    # for the mouse and the cricket columns
    for animal in tar_columns:
        # # for the x or y coordinate
        # for col in animal:
        #     filtered_traces[:, col] = medfilt(files[:, col], kernel_size=kernel_size)
        filtered_traces[animal] = medfilt(files[animal], kernel_size=kernel_size)

    return filtered_traces


def interpolate_segments(files, target_value):
    """Interpolate between the NaNs in a trace"""
    # allocate memory for the output
    interpolated_traces = files.copy()
    # for all the columns
    for col in np.arange(files.shape[1]):
        # get the target trace
        original_trace = files.iloc[:, col].to_numpy()
        # if the target is nan then search for nans, otherwise search for the target value
        if np.isnan(target_value):
            # check if the target value is present, otherwise skip
            if np.sum(np.isnan(original_trace)) == 0:
                interpolated_traces.iloc[:, col] = original_trace
                continue
            x_known = np.squeeze(np.argwhere(~np.isnan(original_trace)))
        else:
            # check if the target value is present, otherwise skip
            if np.sum(original_trace == target_value) == 0:
                interpolated_traces.iloc[:, col] = original_trace
                continue
            x_known = np.squeeze(np.argwhere(original_trace != target_value))

        # generate the x vectors as ranges
        x_target = np.arange(original_trace.shape[0])
        # get the known y vector
        y_known = np.expand_dims(original_trace[x_known], 1)
        # run the interpolation
        interpolated_traces.iloc[:, col] = np.squeeze(interp_trace(y_known, x_known, x_target))

    return interpolated_traces


def eliminate_singles(files):
    """Eliminate points from each column that have no neighbors"""
    # allocate memory for the output
    filtered_traces = files.copy()
    # for all the columns
    for col in np.arange(files.shape[1]):
        # get the target trace
        original_trace = files.iloc[:, col]
        # find the derivative of the nan trace
        nan_positions = np.diff(np.isnan(original_trace).astype(np.int32), n=2)
        # find the coordinates of the singles
        single_positions = np.argwhere(nan_positions == 2) + 1
        # single_positions = np.argwhere((nan_positions[:-1] == 1) & (nan_positions[1:] == -1))
        # nan the singles
        filtered_traces.iloc[single_positions, col] = np.nan

    return filtered_traces


def nan_large_jumps(files, tar_columns, max_step, max_length):
    """NaN discontinuities in the trace (for later interpolation)"""
    # allocate memory for the output
    corrected_trace = files.copy()
    # for the mouse and the cricket columns
    for animal in tar_columns:

        # for idx, col in enumerate(animal):
        # get the data
        curr_data = files[animal].copy()

        # take the derivative trace
        result = np.diff(curr_data[:])

        result = np.hstack((0, result))

        # find the places of threshold crossing
        jumps = np.argwhere(np.abs(result) > max_step)
        # get the distance between jumps
        distance_between = np.diff(jumps, axis=0)

        # go through each of the jumps
        for index, jump in enumerate(distance_between):
            # if the jump is smaller than the max_length allowed, NaN it (if bigger, that's a larger error in tracing
            # than can be fixed with just interpolation)
            if jump[0] < max_length:
                curr_data[jumps[index, 0]:jumps[index+1, 0]] = np.nan
        # ends = np.argwhere(result < -max_step)
        # # if they're empty, skip the iteration
        # if (starts.shape[0] == 0) | (ends.shape[0] == 0):
        #     continue
        # else:
        #     starts = starts[:, 0]
        #     ends = ends[:, 0]
        #
        # # match their sizes and order
        # if starts[0] > ends[0]:
        #     ends = ends[1:]
        # if ends.shape[0] == 0:
        #     continue
        # if starts[-1] > ends[-1]:
        #     starts = starts[:-1]
        # # NaN the in-betweens
        # # for all the starts
        # for start, end in zip(starts, ends):
        #     curr_data[start:end] = np.nan
        corrected_trace[animal] = curr_data
    return corrected_trace


def find_frozen_tracking(files, margin=0.5, stretch_length=10):
    """Find places where the trajectory is too steady (i.e. no single pixel movement) and NaN them since it's probably
    not actually tracking"""

    # create a copy of the data
    corrected_trace = files.copy()
    # get the column names
    column_names = corrected_trace.columns
    # run through the columns
    for column in column_names:
        # skip the index column
        if column == 'index':
            continue
        # get the derivative of the traces
        delta_trace = abs(np.diff(corrected_trace[column], axis=0))
        # find the places that don't pass the criterion
        no_movement, number_no_movement = label(delta_trace < margin)
        # add a zero at the beginning to match the size of corrected traces
        no_movement = np.hstack([0, no_movement])
        # go through the jumps
        for jumps in np.arange(1, number_no_movement):
            # if the jump passes the criterion
            if np.sum(no_movement == jumps) >= stretch_length:
                # nan them
                corrected_trace.loc[no_movement == jumps, column] = np.nan
        # no_movement = np.array([el[0] for el in np.argwhere(delta_trace < margin) + 1])
        # # if it's not empty
        # if no_movement.shape[0] > 0:
        #     # nan them
        #     corrected_trace.loc[no_movement, column] = np.nan

    return corrected_trace


def nan_jumps_dlc(files, max_jump=200):
    """Nan stretches in between large jumps, assuming most of the trace is correct"""
    # copy the data
    corrected_trace = files.copy()
    # get the column names
    column_names = corrected_trace.columns
    # run through the columns
    for column in column_names:
        # skip the index column if it's there
        if column == 'index':
            continue
        # find the jumps
        jump_length = np.diff(corrected_trace[column], axis=0)
        jump_location = np.argwhere(abs(jump_length) > max_jump)
        if jump_location.shape[0] == 0:
            continue
        jump_location = [el[0] for el in jump_location]
        # initialize a flag
        pair_flag = True
        # go through pairs of jumps
        for idx, jump in enumerate(jump_location[:-1]):
            # if this is the second member of a pair, skip
            if not pair_flag:
                # reset the pair flag
                pair_flag = True
                continue
            # if this jump and the next have the same sign, skip
            if (jump_length[jump]*jump_length[jump_location[idx+1]]) > 0:
                continue
            # nan the segment in between
            corrected_trace.loc[jump+1:jump_location[idx+1]+1, column] = np.nan
            # set the pair flag
            pair_flag = False

    return corrected_trace


def rescale_pixels(traces, db_data, reference):
    """Use OpenCV to find corners in the image and rescale the data"""

    # set up the looping flag
    valid_corners = False
    # loop until proper corners are found
    while not valid_corners:

        # get the corners
        corner_coordinates = find_corners(db_data['avi_path'], num_frames=50)

        # get the transformation between the reference and the real corners
        perspective_matrix = cv2.getPerspectiveTransform(corner_coordinates.astype('float32'),
                                                         np.array(reference).astype('float32'))
        # get the new corners
        new_corners = np.concatenate((corner_coordinates, np.ones((corner_coordinates.shape[0], 1))), axis=1)
        new_corners = np.matmul(perspective_matrix, new_corners.T).T
        new_corners = np.array([el[:2] / el[2] for el in new_corners])

        # get a single set of coordinates
        test_coordinates = traces[['mouse_x', 'mouse_y']].to_numpy()
        # add a vector of ones for the matrix multiplication
        test_coordinates = np.concatenate((test_coordinates, np.ones((test_coordinates.shape[0], 1))), axis=1)
        # transform
        test_data = np.matmul(perspective_matrix, test_coordinates.T).T
        test_data = np.array([el[:2] / el[2] for el in test_data])

        # # check if the data is within the boundaries
        # if (np.max(test_data[:, 0]) > np.max(new_corners[:, 0])+5 and
        #     np.min(test_data[:, 0]) < np.min(new_corners[:, 0])-5) or \
        #         (np.max(test_data[:, 1]) > np.max(new_corners[:, 1]) + 5 and
        #          np.min(test_data[:, 1]) < np.min(new_corners[:, 1]) - 5):






        valid_corners = True

    # copy the traces
    new_traces = traces.copy()
    # transform the traces

    # get the unique column names, excluding the letter at the end
    column_names = np.unique([el[:-1] for el in traces.columns])
    # for all the unique names
    for column in column_names:
        # if the name + x exists, transform
        if column+'x' in traces.columns:
            # get the x and y data
            original_data = traces[[column + 'x', column + 'y']].to_numpy()
            # add a vector of ones for the matrix multiplication
            original_data = np.concatenate((original_data, np.ones((original_data.shape[0], 1))), axis=1)
            # transform
            new_data = np.matmul(perspective_matrix, original_data.T).T
            new_data = np.array([el[:2]/el[2] for el in new_data])

            # replace the original data
            new_traces[[column + 'x', column + 'y']] = new_data[:, :2]

            # fp.plot_2d([[original_data[:, :2], corner_coordinates]], dpi=100)
            # plt.axis('equal')
            #
            # fp.plot_2d([[new_data, new_corners]], dpi=100)
            # plt.axis('equal')

    return new_traces, new_corners


def find_corners(video_path, num_frames=10):
    """Take the mode of a video to use the image to find corners"""
    # create the video object
    cap = cv2.VideoCapture(video_path)
    # allocate memory for the corners
    corners = []
    # # define sigma for the edge detection parameters
    # sigma = 0.2
    # get the frames to mode
    for frames in np.arange(num_frames):

        # read the image
        img = cap.read()[1]
        # blur to remove noise
        img2 = cv2.medianBlur(img, 5)

        im_median = np.median(img2)
        # find edges
        # img2 = cv2.Canny(img2, 30, 100)
        img2 = cv2.Canny(img2, 30, im_median)
        # plt.imshow(img2)

        # find the corners
        frame_corners = np.squeeze(np.int0(cv2.goodFeaturesToTrack(img2, 25, 0.0001, 100)))

        for c in frame_corners:
            x, y = c.ravel()
            cv2.circle(img2, (x, y), 10, 255, -1)

        plt.imshow(img2)

        # if there aren't 4 corners, skip
        if frame_corners.shape[0] > 4:
            frame_corners = exclude_non_corners(frame_corners, np.array(img2.shape))
            # if it comes out empty as not all 4 corners were found, skip
            if len(frame_corners) == 0:
                continue
        elif frame_corners.shape[0] < 4:
            continue

        # sort the rows
        sort_idx = np.argsort(frame_corners[:, 0], axis=0)
        # append to the list
        corners.append(frame_corners[sort_idx, :])

    # release the video file
    cap.release()
    # take the mode of the corners
    # corners, _ = mode(np.squeeze(corners), axis=0)

    # turn the corners list into an array
    corners = np.array(corners)
    # allocate memory for the output list
    corner_list = []
    # for all corners
    for corner in np.arange(4):
        # get the unique and the counts
        temp_corner, idx, inv, counts = np.unique(corners[:, corner, :],
                                                  axis=0, return_index=True, return_inverse=True, return_counts=True)
        # store the one with the most counts
        corner_list.append(temp_corner[np.argmax(counts)])

    return np.array(corner_list)


def exclude_non_corners(frame_corners, im_size, center_percentage=0.1):
    """Use quadrants and euclidean distances to exclude the non-corner extra points"""
    # allocate memory for the final set of points
    real_points = []
    # exclude points too close to the center of the image
    # get the center of the image
    image_center = im_size/2
    frame_corners = np.array([el for el in frame_corners if
                              (np.abs(el[0] - image_center[0]) > im_size[0]*center_percentage) and
                              (np.abs(el[1] - image_center[1]) > im_size[1]*center_percentage)])

    # calculate the middle of the image
    middle = np.mean(frame_corners, axis=0)
    # get the point angles
    angles = np.rad2deg(np.arctan2(frame_corners[:, 0] - middle[0], frame_corners[:, 1] - middle[1]))
    # determine the middle of the 4 corners via averaging
    for quadrants in np.arange(4):
        # get the points in this quadrant
        if quadrants == 0:
            low_bound = 0
            high_bound = 90
        elif quadrants == 1:
            low_bound = 90.1
            high_bound = 180
        elif quadrants == 2:
            low_bound = -90
            high_bound = 0
        else:
            low_bound = -180
            high_bound = -90.1
        # get the locations of the points
        point_locations = np.argwhere(np.logical_and(low_bound < angles, angles < high_bound))
        # if a point is missing, skip the whole frame
        if point_locations.shape[0] == 0:
            return []
        else:
            # get the points
            quadrant_points = point_locations[0]

        # if there's only 1 point, add it to the list
        if quadrant_points.shape[0] == 1:
            real_points.append(frame_corners[quadrant_points[0], :])
        # otherwise, get the euclidean distance
        else:
            target_points = frame_corners[quadrant_points, :]
            distances = np.linalg.norm(target_points - middle)
            # get the max distance as the point
            real_points.append(target_points[np.argmax(distances), :])

    # return the cleaned up corners
    return np.array(real_points)


def timed_event_finder(dframe_in, parameter, threshold, function, window=5):
    """This function will generate a dataframe with all the encountered windows of traces that pass the threshold
    in the target parameter"""

    # calculate the frame rate
    framerate = np.round(1 / np.mean(np.diff(dframe_in['time_vector']))).astype(int)
    # get the frames per window
    window_frames = int(window * framerate)
    # get the event triggers
    [event_labels, _] = label(function(dframe_in[parameter], threshold))

    # get the event onsets (where the threshold is crossed)
    event_onsets = np.argwhere(np.diff(event_labels) > 0) + 1
    # create a matrix with all the interval indexes
    event_matrix = [[el[0]-window_frames, el[0]+window_frames] for el in event_onsets
                    if ((el-window_frames) > 0) and ((el + window_frames) < event_labels.shape[0])]

    # if no events were found, skip
    if len(event_matrix) == 0:
        return []

    # filter the overlapping events
    nonoverlap_matrix = maximize_nonoverlapping_count(event_matrix)
    # get the output dataframe for each event
    output_events = [dframe_in.iloc[el[0]:el[1], :].copy() for el in nonoverlap_matrix]
    # for all events
    for idx, event in enumerate(output_events):

        # reset the index
        event.reset_index(inplace=True, drop=True)
        # reset the time
        event.loc[:, 'time_vector'] -= event.loc[:, 'time_vector'].to_numpy()[0]
        # add the event id
        event.loc[:, 'event_id'] = idx
    # concatenate and output
    return pd.concat(output_events)

# class and functions taken from https://stackoverflow.com/questions/16312871/python-removing-overlapping-lists


class IntervalSub(Interval, object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.removed = False


def maximize_nonoverlapping_count(intervals):
    intervals = [IntervalSub(start, end) for start, end in intervals]

    # sort by the end-point
    intervals.sort(key=lambda x: (x.end, (x.end - x.start)))   # O(n*log n)
    tree = build_interval_tree(intervals) # O(n*log n)
    result = []
    for smallest in intervals: # O(n) (without the loop body)
        # pop the interval with the smallest end-point, keep it in the result
        if smallest.removed:
            continue # skip removed nodes
        smallest.removed = True
        result.append([smallest.start, smallest.end]) # O(1)

        # remove (mark) intervals that overlap with the popped interval
        # tree.intersect(smallest.start, smallest.end, # O(log n + m)
        #                lambda x: setattr(x.other, 'removed', True))
        intersection = tree.intersect(smallest.start, smallest.end)
        [setattr(el, 'removed', True) for el in intersection]
    return result


def build_interval_tree(intervals):
    # root = IntervalNode(intervals[0].start, intervals[0].end,
    #                     other=intervals[0])
    root = IntervalNode(intervals[0])
    return reduce(lambda tree, x: tree.insert(x),
                  intervals[1:], root)


def parse_bonsai(path_in):
    """Parse bonsai files"""
    parsed_data = []
    last_nan = 0
    with open(path_in) as f:
        for ex_line in f:
            ex_list = ex_line.split(' ')
            ex_list.remove('\n')
            # TODO: check what happens with older data using the new line
            # if (float(ex_list[0]) < 110 or ex_list[0] == 'NaN') and last_nan == 0:
            if (ex_list[0] == 'NaN') and last_nan == 0:
                continue
            else:
                last_nan = 1

            timestamp = ex_list.pop()
            ex_list = [float(el) for el in ex_list]

            parsed_data.append([ex_list, timestamp])
    return parsed_data
