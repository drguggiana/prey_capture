import numpy as np
from scipy.ndimage.measurements import label
import datetime
from scipy.signal import medfilt
from functions_misc import interp_trace
import cv2
import matplotlib.pyplot as plt


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


def trim_bounds(files):
    """Based on rig specific heuristics, trim the trace"""
    # get the time
    time = [datetime.datetime.strptime(el[1][:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in files]

    # define the arena left boundary (based on the date of the experiment to be retrocompatible)
    if time[0] > datetime.datetime(year=2019, month=11, day=10):
        left_bound_mouse = 0
        left_bound_cricket = 0
    else:
        left_bound_mouse = 110
        left_bound_cricket = 100

    time = [(el - time[0]).total_seconds() for el in time]  # print the frame rate
    print('Frame rate:' + str(1 / np.mean(np.diff(time))) + 'fps')

    # get just the coordinate data
    files = np.vstack(np.array([el[0] for el in files]))
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


def rescale_pixels(traces, db_data):
    """Use OpenCV to find corners in the image and rescale the data"""
    # get the moded image
    corner_coordinates = find_corners(db_data['avi_path'])

    return []


def find_corners(video_path, mode_frames=10):
    """Take the mode of a video to use the image to find corners"""
    # create the video object
    cap = cv2.VideoCapture(video_path)
    # allocate memory for the frames
    corner_list = []
    corners = []
    # get the frames to mode
    for frames in np.arange(mode_frames):
        img = cap.read()[1]
        # # get the corners
        # corners = np.int0(cv2.goodFeaturesToTrack(current_frame[:, :, 0], 25, 0.001, 500))
        # dst = cv2.cornerHarris(current_frame[:, :, 0], 2, 3, 0.04)
        # corner_list.append(corners)
        #
        # for i in corners:
        #     x, y = i.ravel()
        #     cv2.circle(current_frame, (x, y), 10, 255, -1)

        # Initiate FAST object with default values
        fast = cv2.FastFeatureDetector_create(threshold=10, type=2)
        # find and draw the keypoints
        kp = fast.detect(img, None)
        img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        # Print all default params
        print("Threshold: {}".format(fast.getThreshold()))
        print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
        print("neighborhood: {}".format(fast.getType()))
        print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

        plt.imshow(img2)
    plt.show()

    # release the video file
    cap.release()

    return corners
