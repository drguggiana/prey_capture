import numpy as np
from sklearn.preprocessing import scale
from scipy.interpolate import interp1d
from scipy import signal
import cv2
from functions_misc import add_edges, interp_trace, normalize_matrix
import h5py
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import label


def align_traces_maxrate(frame_rate_1, frame_rate_2, data_1, data_2, sign_vector, frame_times, cricket, z=1):
    """Align traces temporally based on whichever one has the fastest frame rate"""
    # determine which rate is highest
    max_rate = np.argmax([frame_rate_1, frame_rate_2])
    # select it as the target for interpolation
    g = [frame_rate_1, frame_rate_2][max_rate]
    # # make both time vectors start at 0
    # frame_times = [el-el[0] for el in frame_times]

    # interpolate one trace and normalize the other depending on the max_rate
    if max_rate == 0:
        # expand the frames to interpolate from
        target_frames = add_edges(frame_times[0], points=100)
        y_motive = sign_vector[2] * data_1[:, sign_vector[0]]
        y_bonsai = interp_motive(sign_vector[3] * data_2[:, sign_vector[1]], frame_times[1], target_frames)
        y_cricket = interp_motive(sign_vector[3] * cricket[:, sign_vector[1]], frame_times[1], target_frames)
        if z == 1:
            height = data_1[:, 2]
        else:
            height = None

    else:
        target_frames = add_edges(frame_times[1], points=100)
        # target_frames = frame_times[1]
        y_motive = interp_motive(sign_vector[2] * data_1[:, sign_vector[0]], frame_times[0], target_frames)
        y_bonsai = sign_vector[3] * data_2[:, sign_vector[1]]
        y_cricket = sign_vector[3] * cricket[:, sign_vector[1]]
        if z == 1:
            height = interp_motive(data_1[:, 2], frame_times[0], target_frames)
        else:
            height = None
    y1 = scale(y_motive)
    y2 = scale(y_bonsai)
    # calculate the cross-correlation for analysis
    acor = np.correlate(y1, y2, mode='full')
    # return the shift, the matched traces and the rate
    return np.int(np.round(np.argmax(acor) - len(y2))), y_motive, y_bonsai, g, y_cricket, height, target_frames


def align_traces_motive(data_1, data_2, sign_vector, frame_times, cricket, z=1):
    """Align traces temporally based on the data_1, which is expected to be from motive"""
    # interpolate the bonsai trace
    # expand the frames to interpolate from
    target_frames = add_edges(frame_times[0], points=100)
    y_motive = sign_vector[2] * data_1[:, sign_vector[0]]
    y_bonsai = interp_motive(sign_vector[3] * data_2[:, sign_vector[1]], frame_times[1], target_frames)
    y_cricket = interp_motive(sign_vector[3] * cricket[:, sign_vector[1]], frame_times[1], target_frames)
    if z == 1:
        height = data_1[:, 2]
    else:
        height = None
    y1 = scale(y_motive)
    y2 = scale(y_bonsai)
    # calculate the cross-correlation for analysis
    acor = np.correlate(y1, y2, mode='full')
    # return the shift, the matched traces and the rate
    return np.int(np.round(np.argmax(acor) - len(y2))), y_motive, y_bonsai, y_cricket, height, frame_times[0]


def interp_motive(position, frame_times, target_times):
    """Interpolate a trace by building an interpolant"""
    # filter the values so the interpolant is trained only on sorted frame times
    sorted_frames = np.hstack((True, np.invert(frame_times[1:] <= frame_times[:-1])))
    frame_times = frame_times[sorted_frames]
    position = position[sorted_frames]
    # also remove any NaN frames
    notnan = ~np.isnan(position)
    frame_times = frame_times[notnan]
    position = position[notnan]
    # create the interpolant
    interpolant = interp1d(frame_times, position, kind='cubic', bounds_error=False, fill_value=np.mean(position))
    return interpolant(target_times)


def homography(from_data, to_data, target_data):
    """Compute the homography transformation between the data sets via opencv"""
    # find the homography transformation
    h, mask = cv2.findHomography(from_data, to_data, method=cv2.RANSAC)
    # make the transformed data homogeneous for multiplication with the affine
    transformed_data = np.squeeze(cv2.convertPointsToHomogeneous(target_data))
    # apply the homography matrix
    return np.matmul(transformed_data, h.T)


def partialaffine(from_data, to_data, target_data):
    """Compute the partial 2D affine transformation between the data sets via opencv"""
    # calculate an approximate affine
    affine_matrix, inliers = cv2.estimateAffinePartial2D(from_data, to_data, ransacReprojThreshold=1,
                                                         maxIters=20000, confidence=0.95,
                                                         refineIters=100, method=cv2.LMEDS)
    assert affine_matrix is not None, "Affine transform was not possible"
    # print('Percentage inliers used:' + str(np.sum(inliers)*100/from_data.shape[0]))
    # make the transformed data homogeneous for multiplication with the affine
    transformed_data = np.squeeze(cv2.convertPointsToHomogeneous(target_data))
    # apply the affine matrix
    return np.matmul(transformed_data, affine_matrix.T)


def affine(from_data, to_data, target_data):
    """Compute the 2D affine transformation between the data sets via opencv"""
    # calculate an approximate affine
    affine_matrix, inliers = cv2.estimateAffine2D(from_data, to_data, ransacReprojThreshold=3,
                                                  maxIters=20000, refineIters=0, method=cv2.LMEDS)
    print('Percentage inliers used:' + str(np.sum(inliers) * 100 / from_data.shape[0]))
    # make the transformed data homogeneous for multiplication with the affine
    transformed_data = np.squeeze(cv2.convertPointsToHomogeneous(target_data))
    # apply the affine matrix
    return np.matmul(transformed_data, affine_matrix.T)


def undistort(data_2d, data_3d, target_data):
    """Undistort points from a 2D camera based on matching 3D data via opencv"""
    # use the calibrateCamera function to get the camera matrix
    test_constant = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
                     cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
                     cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    # test_constant = 0
    # test_constant = cv2.CALIB_FIX_PRINCIPAL_POINT
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([data_3d.astype('float32')],
                                                       [data_2d.astype('float32')],
                                                       (1280, 1024), None, None,
                                                       flags=test_constant)
    # undistort the camera points
    return np.squeeze(cv2.undistortPoints(np.expand_dims(target_data, 1), mtx, dist))


def match_traces(data_3d, data_2d, frame_time_list, coordinate_list, cricket):
    """Match the traces given from motive and bonsai by aligning them temporally and then spatially, also providing
    the transformed cricket position data"""
    # allocate memory for the aligned traces
    aligned_traces = []
    # allocate memory for the cricket traces
    aligned_cricket = []
    # initialize the first shift very high in case there's an error finding the actual shift
    first_shift_idx = 1000
    # for all the coordinate sets (i.e. dimensions)
    for count, sets in enumerate(coordinate_list):
        shift_idx, ya, yb, cr, h, t = align_traces_motive(data_3d, data_2d, sets, frame_time_list, cricket, 1 - count)
        if count == 0:
            first_shift_idx = shift_idx
        # print('Temporal shift:' + str(shift_idx))
        # save the shifted traces

        # depending on the sign of the shift, choose which trace to shift
        if first_shift_idx > 0:

            ya = ya[first_shift_idx:]
            if count == 0:
                y = h[first_shift_idx:]
            shifted_time = t[first_shift_idx:]
        else:

            yb = yb[-first_shift_idx:]
            cr = cr[-first_shift_idx:]
            if count == 0:
                y = h
            shifted_time = t

        assert np.sign(first_shift_idx) == np.sign(shift_idx), 'shifts have a different sign'
        aligned_traces.append([ya, yb])
        aligned_cricket.append(cr)

    # use open cv to obtain a transformation matrix of the aligned traces
    # assemble the data to use for the calculation
    opencv_3d = np.array([aligned_traces[0][0], aligned_traces[1][0], y]).astype('float32')
    opencv_2d = np.array([aligned_traces[0][1], aligned_traces[1][1]]).astype('float32')
    opencv_cricket = np.array(aligned_cricket).astype('float32')
    # match their sizes
    min_size = np.min([opencv_3d.shape[1], opencv_2d.shape[1]])

    return opencv_3d[:, :min_size].T, opencv_2d[:, :min_size].T, shifted_time, opencv_cricket[:, :min_size].T


def match_calcium(calcium_path, sync_path, kinematics_data, frame_bounds, rig=None, trials=None):
    """Match the kinematic and calcium data provided based on the sync file provided"""

    # load the calcium data
    with h5py.File(calcium_path, mode='r') as f:
        calcium_data = np.array(f['calcium_data'])
        # if there are no ROIs, skip
        if (type(calcium_data) == np.ndarray) and (calcium_data == 'no_ROIs'):
            return

    # # get the time vector from bonsai
    # bonsai_time = filtered_traces.time
    # get the number of frames from the bonsai file
    # n_frames_bonsai_file = kinematics_data.shape[0]
    # n_frames_bonsai_file = frame_bounds[2]

    # load the sync data
    # sync_data = pd.read_csv(sync_path, names=['Time', 'mini_frames', 'bonsai_frames'])
    sync_data = pd.read_csv(sync_path)
    if rig in ['VR', 'VScreen', 'VTuning']:
        sync_data.columns = ['Time', 'projector_frames', 'bonsai_frames', 'optitrack_frames', 'mini_frames']

        # get the frame times from the sync file
        frame_times_bonsai_sync = sync_data.loc[
            np.concatenate(([0], np.diff(sync_data.projector_frames) > 0)) > 0, 'Time'].to_numpy()
    else:
        sync_data.columns = ['Time', 'mini_frames', 'bonsai_frames']

        # get the frame times from the sync file
        frame_times_bonsai_sync = sync_data.loc[
            np.concatenate(([0], np.diff(sync_data.bonsai_frames) > 0)) > 0, 'Time'].to_numpy()

    # get the number of miniscope frames on the sync file
    n_frames_mini_sync = np.sum(np.diff(sync_data.mini_frames) > 0)
    # match the sync frames with the actual miniscope frames
    frame_times_mini_sync = sync_data.loc[
        np.concatenate(([0], np.diff(sync_data.mini_frames) > 0)) > 0, 'Time'].to_numpy()

    # find the gaps between bonsai frames, take the frames only after the background subtraction gap
    # bonsai_ifi = np.argwhere(np.diff(sync_data.bonsai_frames) > 0)
    # bonsai_start_frame = bonsai_ifi[np.argwhere(np.diff(bonsai_ifi, axis=0) > 1000)[0][0] + 1][0]
    # n_frames_bonsai_sync = np.sum(np.diff(sync_data.bonsai_frames.to_numpy()[bonsai_start_frame:]) > 0)

    # compare to the frames from bonsai and adjust accordingly (if they don't match, show a warning)
    # plot_2d([[np.diff(frame_times_bonsai_sync[frame_bounds[0]:]),np.diff(kinematics_data['time_vector'].to_numpy())]])
    # if frame_times_bonsai_sync.shape[0] < n_frames_bonsai_file:
    # get the difference in frames between the full video and sync:
    delta_sync = frame_bounds.loc[0, 'original_length'] - frame_times_bonsai_sync.shape[0]
    # if frame_times_bonsai_sync.shape[0] < frame_bounds.loc[0, 'original_length']:
    # if the difference is higher than 0, trim the data from the end (not the sync)
    if delta_sync > 0:
        print('File %s has less sync frames than bonsai frames, trimmed bonsai from end' % sync_path)
        # n_frames_bonsai_file = frame_times_bonsai_sync.shape[0]
        # kinematics_data = kinematics_data[:n_frames_bonsai_file]

        kinematics_data = kinematics_data[:-(delta_sync - 1)]
        # frame_times_bonsai_sync = \
        #     frame_times_bonsai_sync[-(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'start']):
        #                             -(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'end']+1)-delta_sync]
    #     frame_times_bonsai_sync = \
    #         frame_times_bonsai_sync[-(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'start']):
    #                                 -(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'end']+1)]
    # else:
    #     # frame_times_bonsai_sync = frame_times_bonsai_sync[-n_frames_bonsai_file:]
    #     frame_times_bonsai_sync = \
    #         frame_times_bonsai_sync[-(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'start']):
    #                                 -(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'end']+1)]

    # determine the indexes to trim frame_times_bonsai_sync to match the trimming of the data
    trim_start = frame_bounds.loc[0, 'start']
    trim_end = frame_bounds.loc[0, 'end'] - 1
    # trim the sync frames to match the data from both ends (due to preprocessing here)
    frame_times_bonsai_sync = frame_times_bonsai_sync[trim_start:trim_end]

    # frame_times_bonsai_sync = \
    #     frame_times_bonsai_sync[-(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'start']):
    #                             -(frame_bounds.loc[0, 'original_length']-frame_bounds.loc[0, 'end'])]

    if trials is not None:
        # interpolate the bonsai traces to match the mini frames
        matched_bonsai = kinematics_data.drop(['time_vector'] + list(trials.columns), axis=1).apply(interp_trace,
                                                                                                    raw=False,
                                                                                                    args=(
                                                                                                        frame_times_bonsai_sync,
                                                                                                        frame_times_mini_sync))
        # deal with trial numbers
        # first reset the inter-stim intervals
        trial_nums = matched_bonsai.trial_num.to_numpy()
        trial_nums[trial_nums < -500] = 0

        # Find where trials occur, and reassign their index
        trials = np.argwhere(trial_nums != 0)[0]
        breaks = np.where(np.diff(trials) != 1)[0] + 1  # add 1 to compensate for the diff
        split_trials = np.array_split(trials, breaks)

        for trial_num, idxs in enumerate(split_trials):
            trial_nums[idxs] = trial_num + 1  # Compensate for zero indexing

        matched_bonsai.trial_num = trial_nums

        # now that the trials are reassigned, add the trial data
        matched_bonsai = assign_trial_parameters(matched_bonsai, trials)

    else:
        # interpolate the bonsai traces to match the mini frames
        matched_bonsai = kinematics_data.drop(['time_vector'], axis=1).apply(interp_trace, raw=False,
                                                                             args=(frame_times_bonsai_sync,
                                                                                   frame_times_mini_sync))
        # round the quadrant vector as it should be discrete
        quadrant_columns = [el for el in matched_bonsai.columns if ('_quadrant' in el)]
        for el in quadrant_columns:
            matched_bonsai[el] = np.round(matched_bonsai[el])

    # add the correct time vector from the interpolated traces
    matched_bonsai['time_vector'] = frame_times_mini_sync

    # if the calcium data has less frames than the ones detected during triggers, show a warning
    delta_frames = n_frames_mini_sync - calcium_data.shape[1]
    if delta_frames > 0:
        # show the warning
        print("File %s has %s calcium frames less than triggers detected" % (os.path.basename(calcium_path),
                                                                             str(delta_frames)))
        # trim matched bonsai
        matched_bonsai = matched_bonsai.iloc[:-delta_frames, :]

    # trim the data to the frames within the experiment
    calcium_data = calcium_data[:, :n_frames_mini_sync].T
    # also trim the data to the beginning of the video tracking
    # find the index of the closest timestamp to the beginning of the tracking file
    first_tracking_frame = np.argmin(np.abs(frame_times_bonsai_sync[0] - frame_times_mini_sync))
    calcium_data = calcium_data[first_tracking_frame:, :]
    matched_bonsai = matched_bonsai.iloc[first_tracking_frame:, :].reset_index(drop=True)

    # print a single dataframe with the calcium matched positions and timestamps
    calcium_dataframe = pd.DataFrame(calcium_data,
                                     columns=['_'.join(('cell', str(el))) for el in range(calcium_data.shape[1])])
    # concatenate both data frames
    full_dataframe = pd.concat([matched_bonsai, calcium_dataframe], axis=1)

    # reset the time vector
    old_time = full_dataframe['time_vector']
    full_dataframe['time_vector'] = np.array([el - old_time[0] for el in old_time])

    return full_dataframe


def match_cells(match_path):
    """Load the cell matching info if it exists"""
    try:
        with h5py.File(match_path, 'r') as f:
            # load the variables of interest
            assignments = np.array(f['assignments'])
            # f.create_dataset('matchings', data=np.array(matchings))
            date_list = np.array(f['date_list']).astype(str)
    except OSError:
        return empty_dataframe()

    # turn into a data frame
    cell_matches = pd.DataFrame(data=assignments, columns=date_list)
    return cell_matches


def empty_dataframe(column_label='empty'):
    """Return an empty dataframe"""
    return pd.DataFrame(data=[], columns=[column_label])


def match_motive(motive_traces, sync_path, kinematics_data):
    """Match the motive and video traces based on the sync file"""

    # # get the number of frames from the bonsai file
    # n_frames_bonsai_file = kinematics_data.shape[0]

    # load the sync data
    sync_data = pd.read_csv(sync_path, names=['Time', 'projector_frames', 'bonsai_frames',
                                              'optitrack_frames', 'mini_frames'])
    # # get the number of miniscope frames on the sync file
    # n_frames_motive_sync = np.sum(np.abs(np.diff(sync_data.projector_frames)) > 0)

    # match the sync frames with the actual miniscope frames
    frame_times_motive_sync = sync_data.loc[
        np.concatenate(([0], np.abs(np.diff(sync_data.projector_frames)) > 0)) > 0, 'Time'].to_numpy()
    # get the number of motive frames
    n_frames_motive_sync = frame_times_motive_sync.shape[0]
    # trim the trace to where the tracking starts
    first_frame = np.argwhere(motive_traces['time_m'])[0][0]
    trimmed_traces = motive_traces.iloc[first_frame:n_frames_motive_sync, :].reset_index(drop=True)
    # # get the number of frames in motive (assuming the extras in sync are at the end and therefore will be cropped)
    # n_frames_motive_sync = trimmed_traces.shape[0]
    # also trim the frame times (assuming frame 1 in both is the same)
    frame_times_motive_sync = frame_times_motive_sync[first_frame:n_frames_motive_sync + first_frame]
    # plot_2d([[sync_data['projector_frames']]], dpi=100)
    # plot_2d([[sync_data['projector_frames'], sync_data['bonsai_frames']]], dpi=100)
    # get the frame times for bonsai
    frame_times_bonsai_sync = sync_data.loc[
        np.concatenate(([0], np.diff(sync_data.bonsai_frames) > 0)) > 0,
        'Time'].to_numpy()  # [-n_frames_bonsai_file:]

    # trim the frame times to start the same time as motive
    bonsai_start = np.argmin(np.abs(frame_times_motive_sync[0] - frame_times_bonsai_sync))
    frame_times_bonsai_sync = frame_times_bonsai_sync[bonsai_start:]
    # get the number of frames from bonsai
    n_frames_bonsai_file = frame_times_bonsai_sync.shape[0]
    # check if there are extra bonsai frames (most likely at the end) and trim them if so
    if kinematics_data.shape[0] < n_frames_bonsai_file:
        n_frames_bonsai_file = kinematics_data.shape[0]
        frame_times_bonsai_sync = frame_times_bonsai_sync[:n_frames_bonsai_file]
    # trim the bonsai data accordingly (assumption is that the frames go all the way to the end)
    kinematics_data = kinematics_data.iloc[-n_frames_bonsai_file:].reset_index(drop=True)

    # interpolate the bonsai traces to match the mini frames
    matched_bonsai = kinematics_data.drop(['time_vector', 'mouse', 'datetime'],
                                          axis=1).apply(interp_trace, raw=False, args=(frame_times_bonsai_sync,
                                                                                       frame_times_motive_sync))

    # add the correct time vector from the interpolated traces
    matched_bonsai['time_vector'] = frame_times_motive_sync
    matched_bonsai['mouse'] = kinematics_data.loc[0, 'mouse']
    matched_bonsai['datetime'] = kinematics_data.loc[0, 'datetime']

    # trim the motive data

    # # if the motive data has less frames than the ones detected during triggers, show a warning
    # delta_frames = n_frames_motive_sync - motive_traces.shape[0]

    # concatenate both data frames
    full_dataframe = pd.concat([matched_bonsai, trimmed_traces.drop(['time_m'], axis=1)], axis=1)

    # reset the time vector
    old_time = full_dataframe['time_vector']
    full_dataframe['time_vector'] = np.array([el - old_time[0] for el in old_time])

    return full_dataframe


def assign_trial_parameters(motive_traces, trial_list):
    # add columns for the motive_traces df for the trial params
    param_cols = list(trial_list.columns)
    for pc in param_cols:
        motive_traces[pc] = -1000

    # Now assign the trial parameters to the motive_traces dataframe by matching trial numbers
    for index, row in trial_list.iterrows():
        trial_num = index + 1
        motive_traces.loc[motive_traces.trial_num == trial_num, param_cols] = row[param_cols].to_list()

    return motive_traces


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def align_nonaffine(input_traces):
    """For testing nonaffines transforms for automatic data alignment"""

    # copy the traces
    output_traces = input_traces.copy()

    # Grab the mouse data
    bonsai_position = input_traces[['mouse_x', 'mouse_y']].to_numpy()
    motive_position = input_traces[['mouse_z_m', 'mouse_x_m']].to_numpy()

    # Low-pass filter the data
    b, a = signal.butter(5, 0.01)
    filt_bp = signal.filtfilt(b, a, bonsai_position[120:-120], axis=0)
    filt_mp = signal.filtfilt(b, a, motive_position[120:-120], axis=0)

    # Apply manual shift seen from other traces
    shift = [0.085, -0.02]
    filt_bp_shift = filt_bp + shift
    # fig = plot_2d([[filt_mp, filt_bp_shift]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Manual shift')
    # plt.show()

    # Use the camera matrix and distortion coefficients from find_lens_distortion.py
    # to correct the trace
    camMtx = np.array([[1.0078195e+00, 0.0000000e+00, 9.8388788e-05],
                       [0.0000000e+00, 1.0078195e+00, 7.5000345e-05],
                       [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
    distCoeffs = np.array([0.00000000e+00, 0.00000000e+00, -6.54237602e-06, -9.86002297e-06, 0.00000000e+00])
    # filt_bp_udist = np.squeeze(cv2.undistortPoints(np.expand_dims(filt_bp, 1), camMtx, distCoeffs))
    # fig = plot_2d([[filt_mp, filt_bp_udist+shift]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Undistort + manual shift')
    # plt.show()

    # # Get spatial correlation between undistorted DLC tracking and motive tracking to automatically align.
    # # Constrain this to only the center parts of the image for best alignment
    # # TODO - make this better, is hacky
    # motive_sections = np.argwhere((((filt_mp[:, 0] >= -0.1) & (filt_mp[:, 0] <= 0.2)) &
    #                               ((filt_mp[:, 1] >= -0.2) & (filt_mp[:, 1] <= 0))))
    # # motive_subset_idxs = consecutive(np.squeeze(motive_sections))
    # filt_mp_sub = filt_mp[np.squeeze(motive_sections)]
    # filt_bp_udist_sub = filt_bp_udist[np.squeeze(motive_sections)]
    #
    # fig = plot_2d([[filt_mp_sub, filt_bp_udist_sub]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Undistorted subsets in central FOV')
    # plt.show()

    # corr = correlate2d(filt_mp_sub, filt_bp_udist_sub, mode='same')
    # # try way from scipy docs
    # row, col = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
    # shift = filt_mp_sub[row, :] - filt_bp_udist_sub[row, :]
    # # Find max in each column
    # shift_idx = np.argmax(corr, axis=0)
    # x_shift = filt_mp_sub[shift_idx[0], :] - filt_bp_udist_sub[shift_idx[0], :]
    # y_shift = filt_mp_sub[shift_idx[1], :] - filt_bp_udist_sub[shift_idx[1], :]
    # # shift = [x_shift, y_shift]
    #
    # # Align bonsai/DLC to motive coordinates based on spatial correlation
    # bonsai_position_new = filt_bp_udist.copy()
    # bonsai_position_new[:, 0] = bonsai_position_new[:, 0] + y_shift
    # bonsai_position_new[:, 1] = bonsai_position_new[:, 1] + x_shift
    #
    # fig = plot_2d([[filt_mp, filt_bp_udist+shift]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Undistort + spatial correlation shift')
    # plt.show()

    # # Apply partial affine transform to the lens corrected trace
    # filt_bp_udist_affine = partialaffine(filt_bp_udist, filt_mp, filt_bp_udist)
    # fig = plot_2d([[filt_mp, filt_bp_udist_affine]], rows=1, columns=1, dpi=100)
    # plt.show()

    # # Do partial affine transformation on the data
    # filt_bp_affine = partialaffine(filt_bp, filt_mp, filt_bp)
    #
    # fig = plot_2d([[filt_mp, filt_bp_affine]], rows=1, columns=1, dpi=100)
    # plt.show()

    # # Do affine transformation on the data
    # filt_bp_affine = affine(filt_bp, filt_mp, filt_bp)
    #
    # fig = plot_2d([[filt_mp, filt_bp_affine]], rows=1, columns=1, dpi=100)
    # plt.show()

    # # Try undistort
    # filt_mp3 = np.zeros((filt_mp.shape[0], filt_mp.shape[1]+1))
    # filt_mp3[:, :-1] = filt_mp
    # filt_bp_udist = undistort(filt_bp_affine, filt_mp3, filt_bp_affine)
    #
    # fig = plot_2d([[filt_mp, filt_bp_udist]], rows=1, columns=1, dpi=100)
    # plt.show()

    # # Do rotation to the data
    # theta = 7. * np.pi/180.
    # Rmat = np.array([[np.cos(theta), -np.sin(theta)],
    #                  [np.sin(theta), np.cos(theta)]])
    # filt_bp_rot = np.matmul(filt_bp_homog, Rmat)
    #
    # fig = plot_2d([[filt_mp, filt_bp_rot]], rows=1, columns=1, dpi=100)
    # plt.show()

    # Adjust all DLC traces to match motive
    # Note that unity traces (VR crickets) are already in scaled to motive space
    # get the unique column names, excluding the letter at the end
    column_names = np.unique([el[:-1] for el in input_traces.columns])
    # for all the unique names
    for column in column_names:
        # if the name + x exists, transform
        if column + 'x' in input_traces.columns:
            # get the x and y data
            new_data = input_traces[[column + 'x', column + 'y']].to_numpy()

            # Apply the undistortion
            new_data_udist = np.squeeze(cv2.undistortPoints(np.expand_dims(new_data, 1), camMtx, distCoeffs))

            # add the translation
            new_data_udist += shift

            # replace the original data
            output_traces[[column + 'x', column + 'y']] = new_data[:, :2]

    return output_traces


def align_spatial(input_traces):
    """Align the temporally aligned bonsai and motive traces in space"""

    # copy the traces
    output_traces = input_traces.copy()

    # Grab the mouse data
    bonsai_position = input_traces[['mouse_x', 'mouse_y']].to_numpy()
    motive_position = input_traces[['mouse_z_m', 'mouse_x_m']].to_numpy()

    # Low-pass filter the data
    b, a = signal.butter(5, 0.01)
    filt_bp = signal.filtfilt(b, a, bonsai_position[120:-120], axis=0)
    filt_mp = signal.filtfilt(b, a, motive_position[120:-120], axis=0)

    # # Plot filtered but not shifted data
    # fig = plot_2d([[filt_mp, filt_bp]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Manual shift')
    # plt.show()

    # Apply manual shift seen from other traces (calculated manually)
    shift = [0.085, -0.02]
    # fig = plot_2d([[filt_mp, filt_bp+shift]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Manual shift')
    # plt.show()

    # Use the camera matrix and distortion coefficients from find_lens_distortion.py
    # to correct the trace
    camMtx = np.array([[1.0078195e+00, 0.0000000e+00, 9.8388788e-05],
                       [0.0000000e+00, 1.0078195e+00, 7.5000345e-05],
                       [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
    distCoeffs = np.array([0.00000000e+00, 0.00000000e+00, -6.54237602e-06, -9.86002297e-06, 0.00000000e+00])
    # filt_bp_udist = np.squeeze(cv2.undistortPoints(np.expand_dims(filt_bp, 1), camMtx, distCoeffs))
    # fig = plot_2d([[filt_mp, filt_bp_udist+shift]], rows=1, columns=1, dpi=100)
    # fig.suptitle('Undistort + manual shift')
    # plt.show()

    # Adjust all DLC traces to match motive
    # Note that unity traces (VR crickets) are already in scaled to motive space
    # get the unique column names, excluding the letter at the end
    column_names = np.unique([el[:-1] for el in input_traces.columns])
    # for all the unique names
    for column in column_names:
        # if the name + x exists, transform
        if column + 'x' in input_traces.columns:
            # get the x and y data
            new_data = input_traces[[column + 'x', column + 'y']].to_numpy()

            # Apply the undistortion
            new_data_udist = np.squeeze(cv2.undistortPoints(np.expand_dims(new_data, 1), camMtx, distCoeffs))

            # add the translation
            new_data_udist += shift

            # replace the original data
            output_traces[[column + 'x', column + 'y']] = new_data[:, :2]

    return output_traces


def match_motive_2(motive_traces, sync_path, kinematics_data):
    """Match the motive and video traces based on the sync file, updated to second gen rig"""

    # find the first motive frame
    first_motive = np.argwhere(motive_traces.loc[:, 'trial_num'].to_numpy() == 0)[0][0]
    # exclude the last frame if it managed to include a single frame of 0
    last_motive = -1 if motive_traces.loc[motive_traces.shape[0] - 1, 'trial_num'] == 0 else motive_traces.shape[0]
    # trim the motive frames to the start and end of the experiment
    trimmed_traces = motive_traces.iloc[first_motive:last_motive, :].reset_index(drop=True)
    # TODO: remove this for regular trials, only here for 21.2.2022 ones
    if np.max(trimmed_traces.loc[:, 'color_factor']) > 81:
        trimmed_traces.loc[:, 'color_factor'] = trimmed_traces.loc[:, 'color_factor'] / 255
    # normalize the number to 0 1 2 3 range
    trimmed_traces.loc[:, 'color_factor'] = np.array([int('0b' + format(int(el) - 1, '#09b')[2] +
                                                          format(int(el) - 1, '#09b')[4], 2)
                                                      if el > 0 else 0 for el in trimmed_traces.loc[:, 'color_factor']])

    # load the sync data
    sync_data = pd.read_csv(sync_path, names=['Time', 'projector_frames', 'camera_frames',
                                              'sync_trigger', 'mini_frames', 'wheel_frames', 'projector_frames_2'],
                            index_col=False)
    # get the camera frames (as the indexes from sync_frames are referenced for the uncut sync_data, see match_dlc)
    frame_times_cam_sync = sync_data.loc[kinematics_data['sync_frames'].to_numpy(), 'Time'].to_numpy()

    # get the start and end triggers
    sync_start = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 1)[0][0] - 1
    sync_end = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 2)[0][0]

    # trim the sync data to the experiment
    sync_data = sync_data.iloc[sync_start:sync_end, :].reset_index(drop=True)

    # get the motive frame times
    # TODO: probs remove this later, since all trials should be on the new rig with the 2bit frame encoding
    if np.any(np.isnan(sync_data['projector_frames_2'])):
        # get the frame indexes
        idx_code = np.argwhere(np.abs(np.diff(np.round(sync_data.loc[:, 'projector_frames'] / 4))) > 0).squeeze() + 1
        # get the frame times
        frame_times_motive_sync = sync_data.loc[idx_code, 'Time'].to_numpy()
        # if the number of frames doesn't match, trim from the end
        if trimmed_traces.shape[0] > frame_times_motive_sync.shape[0]:
            trimmed_traces = trimmed_traces.iloc[-frame_times_motive_sync.shape[0]:, :]
        elif trimmed_traces.shape[0] < frame_times_motive_sync.shape[0]:
            frame_times_motive_sync = frame_times_motive_sync[-trimmed_traces.shape[0]:]

    else:
        # binarize both frame streams
        frames_0 = np.round(sync_data.loc[:, 'projector_frames'] / 4).astype(int) * 2
        frames_1 = np.round(sync_data.loc[:, 'projector_frames_2'] / 4).astype(int)
        # assemble the actual sequence
        frame_code = (frames_0 | frames_1).to_numpy()

        # TODO: turn this into a function
        fixed_code = frame_code.copy()
        # for all the frames
        for idx, frame in enumerate(frame_code[1:-1]):
            idx += 1
            # if it's the same number as before, skip
            if frame == fixed_code[idx - 1]:
                continue
            # if the numbers before and after are equal
            if fixed_code[idx - 1] == frame_code[idx + 1]:
                # replace this position by the repeated number cause it's likely a mistake
                fixed_code[idx] = frame_code[idx - 1]
                continue
            # if not, start filtering
            # first check for 0-2, cause 3 is a special case
            if fixed_code[idx - 1] in [0, 1, 2]:
                if frame != fixed_code[idx - 1] + 1:
                    fixed_code[idx] = fixed_code[idx - 1] + 1
                    continue
            else:
                if frame != 0:
                    fixed_code[idx] = 0
                    continue

        # get the motive-based frame code in sync
        idx_code = np.argwhere(np.abs(np.diff(fixed_code)) > 0).squeeze() + 1
        motive_code = fixed_code[idx_code]
        # if the frame numbers don't match, find the first motive color number and match that
        last_number = trimmed_traces.loc[trimmed_traces.shape[0] - 1, 'color_factor']
        # trim the idx based on the last appearance of the last_number in motive_code
        trim_idx = np.argwhere(motive_code == last_number)[-1][0] + 1
        idx_code = idx_code[-(trimmed_traces.shape[0] + 1):trim_idx]
        # if idx_code.shape[0] < trimmed_traces.shape[0]:
        #
        #     # get the difference in frames
        #     delta_frames = trimmed_traces.shape[0] - idx_code.shape[0]
        #     # get trimmed traces trimmed
        #     idx_code = idx_code[delta_frames:]
        # display_code = fixed_code[idx_code]
        # get the frame times
        frame_times_motive_sync = sync_data.loc[idx_code, 'Time'].to_numpy()
        # trim the motive frames to be contained within the camera frames
        if frame_times_motive_sync[0] < frame_times_cam_sync[0]:
            start_idx = np.argwhere(frame_times_motive_sync > frame_times_cam_sync[0])[0][0]
            frame_times_motive_sync = frame_times_motive_sync[start_idx:]
            idx_code = idx_code[start_idx:]
            trimmed_traces = trimmed_traces.iloc[start_idx:, :].reset_index(drop=True)
        if frame_times_motive_sync[-1] > frame_times_cam_sync[-1]:
            end_idx = np.argwhere(frame_times_motive_sync < frame_times_cam_sync[-1])[-1][0] + 1
            frame_times_motive_sync = frame_times_motive_sync[:end_idx]
            idx_code = idx_code[:end_idx]
            trimmed_traces = trimmed_traces.iloc[:end_idx, :].reset_index(drop=True)

        if trimmed_traces.shape[0] > frame_times_motive_sync.shape[0]:
            delta_frames = trimmed_traces.shape[0] - frame_times_motive_sync.shape[0]
            trimmed_traces = trimmed_traces.iloc[delta_frames:, :].reset_index(drop=True)

    # interpolate the camera traces to match the unity frames
    matched_camera = kinematics_data.drop(['time_vector', 'mouse', 'datetime', 'sync_frames'],
                                          axis=1).apply(interp_trace, raw=False, args=(frame_times_cam_sync,
                                                                                       frame_times_motive_sync))

    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # # ax.scatter(sync_data.loc[:, 'Time'], sync_data.loc[:, 'projector_frames'])
    # ax.scatter(sync_data.loc[:, 'Time'], sync_data.loc[:, 'camera_frames'])
    # ax.scatter(sync_data.loc[:, 'Time'], np.round(sync_data.loc[:, 'projector_frames']/4)*4)
    # ax.scatter(frame_times_motive_sync, np.ones_like(frame_times_motive_sync))
    #
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(211)
    # # ax.plot(np.diff(motive_traces.loc[:, 'time_m']))
    # ax.plot(frame_times_motive_sync[1:], np.diff(frame_times_motive_sync))
    #
    # fig3 = plt.figure()
    # ax = fig3.add_subplot(211)
    # ax.plot(sync_data.loc[1:, 'Time'], np.diff(frame_code))
    # ax.plot(sync_data.loc[1:, 'Time'], np.diff(fixed_code))
    #
    # fig4 = plt.figure()
    # ax = fig4.add_subplot(211)
    # # ax.plot(sync_data.loc[:, 'Time'], sync_data.loc[:, 'projector_frames'])
    # ax.plot(sync_data.loc[:, 'Time'], sync_data.loc[:, 'sync_trigger'])
    # ax.scatter(frame_times_motive_sync, np.ones_like(frame_times_motive_sync))
    #
    #
    # fig5 = plt.figure()
    # ax = fig5.add_subplot(211)
    # ax.plot(trimmed_traces.loc[:, 'time_m'], trimmed_traces.loc[:, 'trial_num'])
    # ax.plot(trimmed_traces.loc[:, 'time_m'], trimmed_traces.loc[:, 'sync_trigger'])
    #
    # fig6 = plt.figure()
    # ax = fig6.add_subplot(111)
    # ax.plot(np.diff(motive_code), marker='o')

    # add the correct time vector from the interpolated traces
    matched_camera['time_vector'] = frame_times_motive_sync
    matched_camera['mouse'] = kinematics_data.loc[kinematics_data.index[0], 'mouse']
    matched_camera['datetime'] = kinematics_data.loc[kinematics_data.index[0], 'datetime']
    # correct the frame indexes to work with the untrimmed sync file
    idx_code += sync_start
    matched_camera['sync_frames'] = idx_code

    # concatenate both data frames
    full_dataframe = pd.concat([matched_camera, trimmed_traces.drop(['time_m', 'color_factor'], axis=1)], axis=1)

    # reset the time vector
    old_time = full_dataframe['time_vector']
    full_dataframe['time_vector'] = np.array([el - old_time[0] for el in old_time])

    return full_dataframe


def match_calcium_2(calcium_path, sync_path, kinematics_data, trials=None):
    # load the calcium data (cells x time), transpose to get time x cells
    with h5py.File(calcium_path, mode='r') as f:
        calcium_data = np.array(f['calcium_data']).T

        # if there are no ROIs, skip
        if (type(calcium_data) == np.ndarray) and (calcium_data == 'no_ROIs'):
            return None, None
        roi_info = np.array(f['roi_info'])
    # check if there are nans in the columns, if so, also skip
    if kinematics_data.columns[0] == 'badFile':
        print(f'File {os.path.basename(calcium_path)} not matched due to NaNs')
        return None, None

    # load the sync data
    sync_data = pd.read_csv(sync_path, header=None)
    if sync_data.shape[1] == 3:
        sync_data.columns = ['Time', 'mini_frames', 'camera_frames']
    elif sync_data.shape[1] == 6:
        # TODO: only for files from 21.02.2022
        sync_data.columns = ['Time', 'projector_frames', 'camera_frames',
                             'sync_trigger', 'mini_frames', 'wheel_frames']
    else:
        sync_data.columns = ['Time', 'projector_frames', 'camera_frames',
                             'sync_trigger', 'mini_frames', 'wheel_frames', 'projector_frames_2']

    # get the camera frame times
    frame_idx_camera_sync = kinematics_data['sync_frames'].to_numpy().astype(int)
    frame_times_camera_sync = sync_data.loc[frame_idx_camera_sync, 'Time'].to_numpy()
    # get the miniscope frame indexes from the sync file
    frame_idx_mini_sync = np.argwhere(np.diff(np.round(sync_data.loc[:, 'mini_frames'])) > 0).squeeze() + 1
    # interpolate missing triggers (based on experience)
    frame_idx_mini_sync = np.round(interpolate_frame_triggers(frame_idx_mini_sync))
    # correct for the calcium starting before and/or ending after the behavior
    if frame_idx_mini_sync[0] < frame_idx_camera_sync[0]:
        start_idx = np.argwhere(frame_idx_mini_sync > frame_idx_camera_sync[0])[0][0]
        frame_idx_mini_sync = frame_idx_mini_sync[start_idx:]
        calcium_data = calcium_data[start_idx:, :]
    if frame_idx_mini_sync[-1] > frame_idx_camera_sync[-1]:
        end_idx = np.argwhere(frame_idx_mini_sync < frame_idx_camera_sync[-1])[-1][0] + 1
        frame_idx_mini_sync = frame_idx_mini_sync[:end_idx]
        calcium_data = calcium_data[:end_idx, :]
    # get the delta frames with the calcium
    delta_frames = frame_idx_mini_sync.shape[0] - calcium_data.shape[0]
    # remove extra detections coming from terminating the calcium mid frame (I think)
    if delta_frames > 0:
        print(f'There were {delta_frames} triggers more than frames on file {os.path.basename(calcium_path)}')
        frame_idx_mini_sync = frame_idx_mini_sync[:-delta_frames]
    elif delta_frames < 0:
        print(f'There were {-delta_frames} more frames than triggers on file {os.path.basename(calcium_path)}')
        calcium_data = calcium_data[:delta_frames, :]
    # trim calcium according to the frames left within the behavior
    calcium_data = calcium_data[frame_idx_mini_sync > frame_idx_camera_sync[0], :]
    # and then remove frames before the behavior starts
    frame_idx_mini_sync = frame_idx_mini_sync[frame_idx_mini_sync > frame_idx_camera_sync[0]]

    # get the actual mini times
    frame_times_mini_sync = sync_data.loc[frame_idx_mini_sync, 'Time'].to_numpy()

    # interpolate the bonsai traces to match the mini frames
    matched_bonsai = kinematics_data.drop(['time_vector', 'sync_frames', 'mouse', 'datetime'],
                                          axis=1).apply(interp_trace, raw=False, args=(frame_times_camera_sync,
                                                                                       frame_times_mini_sync))
    if trials is not None:

        # repair the trial_num column
        matched_bonsai.loc[:, 'trial_num'] = np.round(matched_bonsai.loc[:, 'trial_num'])

        # now that the trials are reassigned, add the trial data
        matched_bonsai = assign_trial_parameters(matched_bonsai, trials)

    else:
        # round the quadrant vector as it should be discrete
        quadrant_columns = [el for el in matched_bonsai.columns if ('_quadrant' in el)]
        for el in quadrant_columns:
            matched_bonsai[el] = np.round(matched_bonsai[el])
        # same for the hunt trace
        if 'hunt_trace' in matched_bonsai.columns:
            matched_bonsai.loc[:, 'hunt_trace'] = np.round(matched_bonsai.loc[:, 'hunt_trace'])

    # add the correct time vector from the interpolated traces, plus mouse and datetime
    matched_bonsai['time_vector'] = frame_times_mini_sync
    matched_bonsai['mouse'] = kinematics_data.loc[0, 'mouse']
    matched_bonsai['datetime'] = kinematics_data.loc[0, 'datetime']

    # print a single dataframe with the calcium matched positions and timestamps
    cell_column_names = ['_'.join(('cell', f'{el:04d}')) for el in range(calcium_data.shape[1])]
    calcium_dataframe = pd.DataFrame(calcium_data, columns=cell_column_names)
    # concatenate both data frames
    full_dataframe = pd.concat([matched_bonsai, calcium_dataframe], axis=1)

    # reset the time vector
    old_time = full_dataframe['time_vector']
    full_dataframe.loc[:, 'time_vector'] = np.array([el - old_time[0] for el in old_time])

    # turn the roi info into a dataframe
    roi_info = pd.DataFrame(roi_info, columns=['centroid_x', 'centroid_y',
                                               'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'area'])

    return full_dataframe, roi_info


def match_calcium_wf(calcium_path, sync_path, kinematics_data, trials=None):
    # load the calcium data (cells x time), transpose to get time x cells
    with h5py.File(calcium_path, mode='r') as f:
        calcium_data = np.array(f['calcium_data']).T
        fluor_data = np.array(f['fluor_data']).T

        # if there are no ROIs, skip
        if (type(calcium_data) == np.ndarray) and np.any(calcium_data.astype(str) == 'no_ROIs'):
            return None, None

        roi_info = np.array(f['roi_info'])

    # check if there are nans in the columns, if so, also skip
    if kinematics_data.columns[0] == 'badFile':
        print(f'File {os.path.basename(calcium_path)} not matched due to NaNs')
        return None, None

    # load the sync data
    sync_data = pd.read_csv(sync_path, header=None)
    if sync_data.shape[1] == 3:
        sync_data.columns = ['Time', 'mini_frames', 'camera_frames']
    
    elif sync_data.shape[1] == 6:
        # TODO: only for files from 21.02.2022
        sync_data.columns = ['Time', 'projector_frames', 'camera_frames',
                             'sync_trigger', 'mini_frames', 'wheel_frames']
    
    else:
        sync_data.columns = ['Time', 'projector_frames', 'camera_frames',
                             'sync_trigger', 'mini_frames', 'wheel_frames', 'projector_frames_2']

    # get the camera frame times
    frame_idx_camera_sync = kinematics_data['sync_frames'].to_numpy().astype(int)
    frame_times_camera_sync = sync_data.loc[frame_idx_camera_sync, 'Time'].to_numpy()
    
    # get the miniscope frame indexes from the sync file
    frame_idx_mini_sync = np.argwhere(np.diff(np.round(sync_data.loc[:, 'mini_frames'])) > 0).squeeze() + 1

    # correct for the calcium starting before and/or ending after the behavior
    if frame_idx_mini_sync[0] < frame_idx_camera_sync[0]:
        start_idx = np.argwhere(frame_idx_mini_sync > frame_idx_camera_sync[0])[0][0]
        frame_idx_mini_sync = frame_idx_mini_sync[start_idx:]
        calcium_data = calcium_data[start_idx:, :]
        fluor_data = fluor_data[start_idx:, :]

    if frame_idx_mini_sync[-1] > frame_idx_camera_sync[-1]:
        end_idx = np.argwhere(frame_idx_mini_sync < frame_idx_camera_sync[-1])[-1][0] + 1
        frame_idx_mini_sync = frame_idx_mini_sync[:end_idx]
        calcium_data = calcium_data[:end_idx, :]
        fluor_data = fluor_data[:end_idx, :]

    # get the delta frames with the calcium
    delta_frames = frame_idx_mini_sync.shape[0] - calcium_data.shape[0]

    # remove extra detections coming from terminating the calcium mid frame (I think)
    if delta_frames > 0:
        print(f'There were {delta_frames} triggers more than frames on file {os.path.basename(calcium_path)}')
        frame_idx_mini_sync = frame_idx_mini_sync[:-delta_frames]
    elif delta_frames < 0:
        print(f'There were {-delta_frames} more frames than triggers on file {os.path.basename(calcium_path)}')
        calcium_data = calcium_data[:delta_frames, :]
        fluor_data = fluor_data[:delta_frames, :]

    # trim calcium according to the frames left within the behavior
    calcium_data = calcium_data[frame_idx_mini_sync > frame_idx_camera_sync[0], :]
    # do the same with fluorescence data
    fluor_data = fluor_data[frame_idx_mini_sync > frame_idx_camera_sync[0], :]
    # and then remove frames before the behavior starts
    frame_idx_mini_sync = frame_idx_mini_sync[frame_idx_mini_sync > frame_idx_camera_sync[0]]

    # get the actual mini times
    frame_times_mini_sync = sync_data.loc[frame_idx_mini_sync, 'Time'].to_numpy()

    # interpolate the bonsai traces to match the mini frames
    matched_bonsai = kinematics_data.drop(['time_vector', 'sync_frames', 'mouse', 'datetime'],
                                          axis=1).apply(interp_trace, raw=False, args=(frame_times_camera_sync,
                                                                                       frame_times_mini_sync))
    if trials is not None:

        # repair the trial_num column
        real_trials = np.concatenate([[0], trials.index + 1])
        incorrect_trial_assignments = ~np.isin(matched_bonsai.loc[:, 'trial_num'], real_trials)
        incorrect_trial_idxs = matched_bonsai.index[incorrect_trial_assignments]
        
        # Check preceding or following frames for the correct trial number
        for idx in incorrect_trial_idxs:
            trial_val = matched_bonsai.loc[idx, 'trial_num']
            rounded_trial = np.round(trial_val)
            prec_trial = matched_bonsai.loc[idx - 1, 'trial_num']
            next_trial = matched_bonsai.loc[idx + 1, 'trial_num']
            
            if (rounded_trial == prec_trial) and (next_trial == 0):
                matched_bonsai.loc[idx, 'trial_num'] = prec_trial
            elif (rounded_trial == next_trial) and (prec_trial == 0):
                matched_bonsai.loc[idx, 'trial_num'] = next_trial
            elif (rounded_trial == prec_trial) and (rounded_trial == next_trial):
                matched_bonsai.loc[idx, 'trial_num'] = prec_trial
            else:
                matched_bonsai.loc[idx, 'trial_num'] = 0
        
        matched_bonsai.loc[:, 'trial_num'] = np.round(matched_bonsai.loc[:, 'trial_num'])

        # find indexes for each trial number > 0. If there are some that aren't consecutive, fix them
        # Seems to be the case the sometimes the transition is split across two frames
        for trial in matched_bonsai.trial_num.unique():
            indexes = matched_bonsai.index[matched_bonsai.trial_num == trial]
            if np.any(np.diff(indexes) != 1):
                where_bad = np.argwhere(np.diff(indexes) > 1).squeeze() + 1
                matched_bonsai.loc[indexes[where_bad], 'trial_num'] = 0

        # now that the trials are reassigned, add the trial data
        matched_bonsai = assign_trial_parameters(matched_bonsai, trials)

    else:
        # round the quadrant vector as it should be discrete
        quadrant_columns = [el for el in matched_bonsai.columns if ('_quadrant' in el)]

        for el in quadrant_columns:
            matched_bonsai[el] = np.round(matched_bonsai[el])

        # same for the hunt trace
        if 'hunt_trace' in matched_bonsai.columns:
            matched_bonsai.loc[:, 'hunt_trace'] = np.round(matched_bonsai.loc[:, 'hunt_trace'])

    # add the correct time vector from the interpolated traces, plus mouse and datetime
    matched_bonsai['time_vector'] = frame_times_mini_sync
    matched_bonsai['mouse'] = kinematics_data.loc[0, 'mouse']
    matched_bonsai['datetime'] = kinematics_data.loc[0, 'datetime']

    # print a single dataframe with the calcium matched positions and timestamps
    cell_column_names = ['_'.join(('cell', f'{el:04d}', 'spikes')) for el in range(calcium_data.shape[1])]
    calcium_dataframe = pd.DataFrame(calcium_data, columns=cell_column_names)

    cell_column_names = [col.replace('spikes', 'fluor') for col in cell_column_names]
    fluorescence_dataframe = pd.DataFrame(fluor_data, columns=cell_column_names)

    # concatenate both data frames
    full_dataframe = pd.concat([matched_bonsai, calcium_dataframe, fluorescence_dataframe], axis=1)

    # reset the time vector
    old_time = full_dataframe['time_vector']
    full_dataframe.loc[:, 'time_vector'] = np.array([el - old_time[0] for el in old_time])

    # turn the roi info into a dataframe
    roi_info = pd.DataFrame(roi_info, columns=['centroid_x', 'centroid_y',
                                               'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'area'])

    return full_dataframe, roi_info


def match_wheel(file_info, filtered_traces, wheel_diameter=16):
    """Get the wheel speed and acceleration on each frame"""
    # load the sync data
    sync_data = pd.read_csv(file_info['sync_path'], header=None)
    if sync_data.shape[1] == 6:
        sync_data.columns = ['Time', 'projector_frames', 'camera_frames',
                             'sync_trigger', 'mini_frames', 'wheel_frames']
    else:
        sync_data.columns = ['Time', 'projector_frames', 'camera_frames',
                             'sync_trigger', 'mini_frames', 'wheel_frames', 'projector_frames_2']
    #
    # sync_data = pd.read_csv(file_info['sync_path'], names=['Time', 'projector_frames', 'camera_frames',
    #                                                        'sync_trigger', 'mini_frames', 'wheel_frames'],
    #                         index_col=False)
    # get the wheel trace
    wheel_position = sync_data.loc[filtered_traces['sync_frames'], ['wheel_frames']]
    # convert the position to radians
    wheel_position = (wheel_position - wheel_position.min()) / (wheel_position.max() - wheel_position.min()) * 2 * np.pi
    # unwrap
    wheel_position = np.unwrap(wheel_position).flatten()
    # get the speed of the wheel
    wheel_speed = np.diff(wheel_position * np.pi * (wheel_diameter - 1) / 360)
    # get the wheel acceleration
    wheel_acceleration = np.diff(wheel_speed)
    # prepend zeros to speed and acceleration arrays
    wheel_speed = np.insert(wheel_speed, 0, 0, axis=0)
    wheel_acceleration = np.insert(wheel_acceleration, [0, 0], 0, axis=0)
    # save in the output frame
    filtered_traces['wheel_speed'] = wheel_speed
    filtered_traces['wheel_acceleration'] = wheel_acceleration

    return filtered_traces


def match_eye(filtered_traces, eye_model='sakatani+isa'):
    """Extract and process the eye tracking data"""
    filtered_traces.reset_index(drop=True, inplace=True)

    filtered_traces.reset_index(inplace=True, drop=True)

    # --- Blink detection --- #
    # create vectors between the eye nasal eye corner and eyelid top and bottom
    eyelid_top_vec = filtered_traces.loc[:, ['eyelid_top_x', 'eyelid_top_y']].to_numpy() - \
                     filtered_traces.loc[:, ['eye_corner_nasal_x', 'eye_corner_nasal_y']].to_numpy()
    eyelid_bottom_vec = filtered_traces.loc[:, ['eyelid_bottom_x', 'eyelid_bottom_y']].to_numpy() - \
                        filtered_traces.loc[:, ['eye_corner_nasal_x', 'eye_corner_nasal_y']].to_numpy()
    eyelid_top_vec_u = eyelid_top_vec / np.linalg.norm(eyelid_top_vec, axis=1)[:, np.newaxis]
    eyelid_bottom_vec_u = eyelid_bottom_vec / np.linalg.norm(eyelid_bottom_vec, axis=1)[:, np.newaxis]
    eyelid_angle = np.rad2deg(np.arccos(np.sum(eyelid_top_vec_u * eyelid_bottom_vec_u, axis=1)))
    filtered_traces.insert(loc=0, column='eyelid_angle', value=eyelid_angle)

    # --- Pupil tracking --- #
    pupil_columns = ['pupil_top', 'pupil_top_right', 'pupil_top_left',
                     'pupil_bottom', 'pupil_bottom_right', 'pupil_bottom_left',
                     'pupil_right', 'pupil_left']

    # Need to extract in y, x order because OpenCV defaults to row, column format
    pupil_points_yx = [np.column_stack(filtered_traces.loc[:, [point + '_y', point + '_x']].to_numpy().T) for point in
                       pupil_columns]

    # Gets the array so rows (first dimension) are points for each frame
    pupil_points_yx = np.array(pupil_points_yx).transpose((1, 0, 2))

    # Fit ellipse to pupil points - conversion to float needed for cv2
    ellipses = [cv2.fitEllipse(pupil_points_yx[row, :].astype(np.float32)) for
                row in np.arange(pupil_points_yx.shape[0])]

    # calculate diameter and center - take major axis to be pupil diameter
    # note the reversing of the first fit output because of reversed cv2 convention
    fit_columns = ['fit_pupil_center_x', 'fit_pupil_center_y', 'pupil_diameter', 'minor_axis', 'pupil_rotation']
    pupil_fit = pd.DataFrame([[*reversed(fit[0]), *fit[1], fit[-1]] for fit in ellipses], columns=fit_columns)
    filtered_traces = pd.concat([pupil_fit, filtered_traces], axis=1)

    # --- Gaze angle --- #
    # Get horizontal eye axis
    eye_horizontal_vector = filtered_traces.loc[:, ['eye_corner_nasal_x', 'eye_corner_nasal_y']].to_numpy() - \
                            filtered_traces.loc[:, ['eye_corner_temporal_x', 'eye_corner_temporal_y']].to_numpy()

    # horizontal midpoint becomes origin of eye coordinate system
    eye_axis_midpoint = filtered_traces.loc[:,
                        ['eye_corner_temporal_x', 'eye_corner_temporal_y']].to_numpy() + eye_horizontal_vector / 2

    # compute pupil position relative to origin of eye coordinate system
    pupiL_coord_ref = filtered_traces.loc[:, ['fit_pupil_center_x', 'fit_pupil_center_y']].to_numpy() - \
                      eye_axis_midpoint

    center_ref_cols = ['eye_horizontal_vector_x', 'eye_horizontal_vector_y', 'eye_midpoint_x', 'eye_midpoint_y',
                       'pupil_center_ref_x', 'pupil_center_ref_y']
    center_ref_pupil = pd.DataFrame(np.column_stack((eye_horizontal_vector, eye_axis_midpoint, pupiL_coord_ref)),
                                    columns=center_ref_cols)
    filtered_traces = pd.concat([center_ref_pupil, filtered_traces], axis=1)

    # TODO actually calculate gaze angle if needed

    return filtered_traces


def match_dlc(filtered_traces, file_info, file_date):
    """Match the DLC traces with the sync time"""

    # choose the timestamp mode depending on the date
    # (this is here mostly just in case, should be able to handle files before the sync file)
    if file_date <= datetime.datetime(year=2019, month=11, day=11, hour=20):
        # parse the bonsai file for the time stamps
        timestamp = []

        with open(file_info['bonsai_path']) as f:
            for ex_line in f:
                ex_list = ex_line.split(' ')
                ex_list.remove('\n')
                timestamp.append(ex_list.pop())

        # add the time stamps to the main dataframe
        time = np.array([datetime.datetime.strptime(el[:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in timestamp])
        time = np.array([el.total_seconds() for el in (time - time[0])])
        cam_idx = np.ones_like(time) * np.nan

    elif (file_date <= datetime.datetime(year=2021, month=12, day=14)) & \
            (file_date > datetime.datetime(year=2019, month=11, day=10)):

        # load the sync file
        sync_data = pd.read_csv(file_info['sync_path'], names=['Time', 'mini_frames', 'camera_frames'], index_col=False)
        # get the camera triggers
        cam_idx = np.argwhere(np.diff(sync_data.loc[:, 'camera_frames']) > 0).squeeze() + 1
        # get the actual frame times
        time = sync_data.loc[cam_idx, 'Time'].to_numpy()

    elif (file_date > datetime.datetime(year=2021, month=12, day=14)) & \
            (file_date <= datetime.datetime(year=2022, month=2, day=21)):
        # read the sync file
        sync_data = pd.read_csv(file_info['sync_path'], names=['Time', 'projector_frames', 'camera_frames',
                                                               'sync_trigger', 'mini_frames', 'wheel_frames'],
                                index_col=False)
        # get the cam frame times based on the triggers
        # get the camera triggers
        cam_idx = np.argwhere(np.diff(sync_data.loc[:, 'camera_frames']) > 1).squeeze() + 1
        # filter them by the sync trigger
        sync_start = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 1)[0]
        sync_end = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 2)[0]
        cam_idx = cam_idx[(cam_idx > sync_start) & (cam_idx < sync_end)]
        # get the actual frame times
        time = sync_data.loc[cam_idx, 'Time'].to_numpy()

    else:
        # read the sync file
        sync_data = pd.read_csv(file_info['sync_path'], names=['Time', 'projector_frames', 'camera_frames',
                                                               'sync_trigger', 'mini_frames', 'wheel_frames',
                                                               'projector_frames_2'], index_col=False)
        # get the cam frame times based on the triggers
        # get the camera triggers
        cam_idx = np.argwhere(np.diff(sync_data.loc[:, 'camera_frames']) > 1).squeeze() + 1
        # filter them by the sync trigger
        sync_start = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 1)[0]
        sync_end = np.argwhere(sync_data.loc[:, 'sync_trigger'].to_numpy() == 2)[0]
        cam_idx = cam_idx[(cam_idx > sync_start) & (cam_idx < sync_end)]
        # get the actual frame times
        time = sync_data.loc[cam_idx, 'Time'].to_numpy()

    # correct if there's a discrepancy and report
    if time.shape[0] != filtered_traces.shape[0]:
        delta_frame = time.shape[0] - filtered_traces.shape[0]
        print(f'Discrepancy of {delta_frame} frames between time and traces')
        if delta_frame < 0:
            # correct the traces
            filtered_traces = filtered_traces.iloc[-delta_frame:, :]
        else:
            # correct the time vector and camera idx
            time = time[delta_frame:]
            cam_idx = cam_idx[delta_frame:]

    # make a copy of filtered traces to bypass SettingWithCopyWarning, and return that
    filtered_traces_copy = filtered_traces.copy()

    # save the time and the frames
    filtered_traces_copy['time_vector'] = [el - time[0] for el in time]
    filtered_traces_copy['sync_frames'] = cam_idx

    return filtered_traces_copy


def interpolate_frame_triggers(triggers_in, threshold=1.5):
    """Interpolate missing triggers in the sequence based on the median interval"""
    # allocate the output
    triggers_out = list(triggers_in)
    # get the intervals
    intervals = np.diff(triggers_in)
    # get the median interval
    median_interval = np.median(intervals)
    # get the indexes of the intervals that violate threshold times the median or more (since they are continuous)
    long_interval_idx = np.argwhere(intervals > threshold * median_interval).flatten()
    # cycle through the intervals
    for idx in long_interval_idx:
        # determine the number of frames to interpolate
        frame_number = int(np.round(intervals[idx] / median_interval) - 1)
        # generate and add them to the list
        for idx2 in np.arange(frame_number):
            triggers_out.append(triggers_out[idx] + median_interval * (idx2 + 1))
    # sort the list and output
    triggers_out = np.sort(np.array(triggers_out))
    return triggers_out
