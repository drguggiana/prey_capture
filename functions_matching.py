from sklearn.preprocessing import scale
from scipy.interpolate import interp1d
from scipy import signal
import cv2
from functions_plotting import *
from functions_misc import add_edges, interp_trace, normalize_matrix
import h5py
import pandas as pd
import os
import time


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
    print('Percentage inliers used:' + str(np.sum(inliers)*100/from_data.shape[0]))
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
    aligned_cricket =[]
    # initialize the first shift very high in case there's an error finding the actual shift
    first_shift_idx = 1000
    # for all the coordinate sets (i.e. dimensions)
    for count, sets in enumerate(coordinate_list):
        shift_idx, ya, yb, cr, h, t = align_traces_motive(data_3d, data_2d, sets, frame_time_list, cricket, 1-count)
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


def match_calcium(calcium_path, sync_path, kinematics_data, frame_bounds):
    """Match the kinematic and calcium data provided based on the sync file provided"""

    # load the calcium data
    with h5py.File(calcium_path, mode='r') as f:
        calcium_data = np.array(f['calcium_data'])

    # # get the time vector from bonsai
    # bonsai_time = filtered_traces.time
    # get the number of frames from the bonsai file
    # n_frames_bonsai_file = kinematics_data.shape[0]
    # n_frames_bonsai_file = frame_bounds[2]

    # load the sync data
    sync_data = pd.read_csv(sync_path, names=['Time', 'mini_frames', 'bonsai_frames'])
    # get the number of miniscope frames on the sync file
    n_frames_mini_sync = np.sum(np.diff(sync_data.mini_frames) > 0)
    # match the sync frames with the actual miniscope frames
    frame_times_mini_sync = sync_data.loc[
        np.concatenate(([0], np.diff(sync_data.mini_frames) > 0)) > 0, 'Time'].to_numpy()

    # find the gaps between bonsai frames, take the frames only after the background subtraction gap
    # bonsai_ifi = np.argwhere(np.diff(sync_data.bonsai_frames) > 0)
    # bonsai_start_frame = bonsai_ifi[np.argwhere(np.diff(bonsai_ifi, axis=0) > 1000)[0][0] + 1][0]
    # n_frames_bonsai_sync = np.sum(np.diff(sync_data.bonsai_frames.to_numpy()[bonsai_start_frame:]) > 0)

    # get the frame times from the sync file
    frame_times_bonsai_sync = sync_data.loc[
                                  np.concatenate(([0], np.diff(sync_data.bonsai_frames) > 0)) > 0, 'Time'].to_numpy()
    # compare to the frames from bonsai and adjust accordingly (if they don't match, show a warning)
    # if frame_times_bonsai_sync.shape[0] < n_frames_bonsai_file:
    if frame_times_bonsai_sync.shape[0] < frame_bounds[2]:
        print('File %s has less sync frames than bonsai frames, trimmed bonsai from end' % sync_path)
        n_frames_bonsai_file = frame_times_bonsai_sync.shape[0]
        kinematics_data = kinematics_data[:n_frames_bonsai_file]
    else:
        # frame_times_bonsai_sync = frame_times_bonsai_sync[-n_frames_bonsai_file:]
        frame_times_bonsai_sync = frame_times_bonsai_sync[frame_bounds[0]:frame_bounds[1]]

    # interpolate the bonsai traces to match the mini frames
    matched_bonsai = kinematics_data.drop(['time_vector'], axis=1).apply(interp_trace, raw=False,
                                                                         args=(frame_times_bonsai_sync,
                                                                               frame_times_mini_sync))
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
    frame_times_motive_sync = frame_times_motive_sync[first_frame:n_frames_motive_sync+first_frame]
    # plot_2d([[sync_data['projector_frames']]], dpi=100)
    # plot_2d([[sync_data['projector_frames'], sync_data['bonsai_frames']]], dpi=100)
    # get the frame times for bonsai
    frame_times_bonsai_sync = sync_data.loc[
                                  np.concatenate(([0], np.diff(sync_data.bonsai_frames) > 0)) > 0,
                                  'Time'].to_numpy() # [-n_frames_bonsai_file:]

    # trim the frame times to start the same time as motive
    bonsai_start = np.argmin(np.abs(frame_times_motive_sync[0]-frame_times_bonsai_sync))
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


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


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
    distCoeffs = np.array([0.00000000e+00,  0.00000000e+00, -6.54237602e-06, -9.86002297e-06, 0.00000000e+00])
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
    distCoeffs = np.array([0.00000000e+00,  0.00000000e+00, -6.54237602e-06, -9.86002297e-06, 0.00000000e+00])
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
