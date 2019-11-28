from sklearn.preprocessing import scale
from scipy.interpolate import interp1d
import cv2
from functions_plotting import *
from functions_misc import add_edges


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
    h, mask = cv2.findHomography(from_data, to_data, method=cv2.LMEDS)
    # make the transformed data homogeneous for multiplication with the affine
    transformed_data = np.squeeze(cv2.convertPointsToHomogeneous(target_data))
    # apply the homography matrix
    return np.matmul(transformed_data, h.T)


def partialaffine(from_data, to_data, target_data):
    """Compute the partial 2D affine transformation between the data sets via opencv"""
    # calculate an approximate affine
    affine_matrix, inliers = cv2.estimateAffinePartial2D(from_data, to_data, ransacReprojThreshold=1,
                                                         maxIters=20000, refineIters=100, method=cv2.LMEDS)
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
                                                  maxIters=20000, refineIters=100, method=cv2.LMEDS)
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
    # test_constant = cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([data_3d], [data_2d], (1280, 1024), None, None,
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