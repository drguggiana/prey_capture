# imports
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error as mse
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.interpolate import interp1d


# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)


def align_traces_motive(frame_rate_1, frame_rate_2, data_1, data_2, sign_vector, frame_times):
    # determine which rate is highest
    max_rate = np.argmax([frame_rate_1, frame_rate_2])
    # select it as the target for interpolation
    g = [frame_rate_1, frame_rate_2][max_rate]
    # make both time vectors start at 0
    frame_times = [el-el[0] for el in frame_times]
    # interpolate one trace and normalize the other depending on the max_rate
    if max_rate == 0:
        y_motive = sign_vector[2] * data_1[:, sign_vector[0]]
        # y1 = sign_vector[2] * data_1[:, sign_vector[0]]
        y_bonsai = interp_motive(sign_vector[3] * data_2[:, sign_vector[1]], frame_times[1], frame_times[0])
        # y2 = interp_motive(sign_vector[3] * data_2[:, sign_vector[1]], frame_times[1], frame_times[0])
    else:
        y_motive = interp_motive(sign_vector[2] * data_1[:, sign_vector[0]], frame_times[0], frame_times[1])
        y_bonsai = sign_vector[3] * data_2[:, sign_vector[1]]
        # y2 = sign_vector[3] * data_2[:, sign_vector[1]]

    y1 = scale(y_motive)
    y2 = scale(y_bonsai)
    # y1 = y_motive
    # plot the aligned traces
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.plot(normalize_trace(Ya[first_shift_idx:]))
    # # ax.plot(normalize_trace(Yb))
    # ax.plot(y1)
    # ax.plot(y2)
    # plt.show()
    # calculate the cross-correlation for analysis
    acor = np.correlate(y1, y2, mode='full')
    # return the shift, the matched traces and the rate
    return np.int(np.round(np.argmax(acor) - len(y2))), y_motive, y_bonsai, g


def interp_motive(position, frame_times, target_times):
    # filter the values so the interpolant is trained only on sorted frame times
    sorted_frames = np.hstack((True, np.invert(frame_times[1:] <= frame_times[:-1])))
    frame_times = frame_times[sorted_frames]
    position = position[sorted_frames]
    # create the interpolant
    interpolant = interp1d(frame_times, position, kind='cubic', bounds_error=False, fill_value=np.mean(position))
    # # create the range for interpolation
    # target_frame_times = np.arange(frame_times[1], frame_times[-1], step=1 / target_rate)
    # return the interpolated position
    return interpolant(target_times)


def normalize_matrix(matrix, target=None):
    # normalize between 0 and 1
    out_matrix = (matrix - np.min(matrix.flatten())) / (np.max(matrix.flatten()) - np.min(matrix.flatten()))
    # normalize to the range of a target matrix if provided
    if target is not None:
        out_matrix = out_matrix * (np.max(target.flatten()) - np.min(target.flatten())) + np.min(target.flatten())
    return out_matrix


def parse_line(single_line):
    parsed_line = [float(number) if number != '' else np.nan for number in single_line[:-1].split(',')]
    # parsed_line = re.findall('[+-]?\d+[.]\d+',single_line)
    # parsed_line = [float(s) for s in re.findall('[+-]?\d+.\d+',single_line)]
    return parsed_line


base_path_bonsai = r'C:\Users\drguggiana\Documents\Bonsai_out'
# file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai)
file_path_bonsai = [r'C:\Users\drguggiana\Documents\Bonsai_out\08_16_2019_19_48_37_calibration.csv']

# define loading path and select file
# allocate a list for all the animals
animal_data_bonsai = []

for animal in file_path_bonsai:
    parsed_data = []
    last_nan = 0
    with open(file_path_bonsai[0]) as f:
        for count, ex_line in enumerate(f):
            ex_list = ex_line.split(' ')
            ex_list.remove('\n')
            if count == 0:
                bonsai_start = ex_list[-1]
            if ex_list[0] == 'NaN' and last_nan == 0:
                continue
            else:
                last_nan = 1

            timestamp = ex_list.pop()
            ex_list = [float(el) for el in ex_list]
            parsed_data.append([ex_list, timestamp])

    animal_data_bonsai.append(np.array(parsed_data))

# process the bonsai start time
bonsai_start = datetime.datetime.strptime(bonsai_start[:-7], '%Y-%m-%dT%H:%M:%S.%f')

bonsai_data = np.array([el[0] for el in animal_data_bonsai[0]])
# 2D mouse movement
fig = plt.figure()
ax = fig.add_subplot(111)
number_points = None
ax.plot(bonsai_data[:number_points, 0], bonsai_data[:number_points, 1])

plt.gca().invert_xaxis()

# ax.plot(target_data[:number_points,7], target_data[:number_points,9])
ax.autoscale()
ax.axis('equal')

timestamp = [datetime.datetime.strptime(el[1][:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in animal_data_bonsai[0]]
# store the absolute time for later use
bonsai_time_absolute = np.array(timestamp)
timestamp = np.array([(el - timestamp[0]).total_seconds() for el in timestamp])
# print the frame rate
print('Frame rate:' + str(1 / np.mean(np.diff(timestamp))) + 'fps')

# load the motive file

# define the path to the motive calibration file
motive_path = r'C:\Users\drguggiana\Desktop\Take 2019-08-16 07.48.34 PM.csv'

parsed_data = []
with open(motive_path) as f:
    for count, line in enumerate(f):
        if count == 0:
            motive_start = [str(number) for number in line[:-1].split(',')]
            continue
        elif count < 7:
            continue
        parsed_data.append(parse_line(line))

# turn the list into an array
parsed_data = np.array(parsed_data)
# now get the info of just the marker and also the timestamps
motive_time = parsed_data[:, 1]
motive_data = parsed_data[:, [20, 22, 21]]

# # get the start time
# motive_start = motive_start[3][5:-3]
# motive_start = datetime.datetime.strptime(motive_start, '%Y-%m-%d %H.%M.%S') + datetime.timedelta(hours=12)


# plot the motive data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(motive_data[:, 0], motive_data[:, 2], motive_data[:, 1])

# filter the points for nans
valid_bonsai = ~np.isnan(bonsai_data[:, 0])
timestamp = timestamp[valid_bonsai]
bonsai_data = bonsai_data[valid_bonsai, :]

valid_motive = ~np.isnan(motive_data[:, 0])
motive_time = motive_time[valid_motive]
motive_data = motive_data[valid_motive, :]

# scale the bonsai array to the motive array
bonsai_data = normalize_matrix(bonsai_data, motive_data[:, :2])
# bonsai_data = normalize_matrix(bonsai_data)
# motive_data[:, :2] = normalize_matrix(motive_data[:, :2])

# define the sets of dimensions for the matching coordinates (motive, then webcam, then signs)
coordinate_list = [[1, 0, 1, 1], [0, 1, 1, 1]]
# get the frame rates
frame_rate_motive = np.int(np.round(1 / np.mean(np.diff(motive_time))))
frame_rate_webcam = np.int(np.round(1 / np.median(np.diff(timestamp))))
# store the time vectors in a list
frame_time_list = [motive_time, timestamp]

# allocate memory for the aligned traces
aligned_traces = []
# initialize the first shift very high in case there's an error finding the actual shift
first_shift_idx = 1000
# for all the coordinate sets (i.e. dimensions)
for count, sets in enumerate(coordinate_list):
    # shift_idx, Ya, Yb, g = align_traces(frame_rate_motive, frame_rate_webcam,
    #                                     motive_data, webcam_data, sets)
    shift_idx, Ya, Yb, g = align_traces_motive(frame_rate_motive, frame_rate_webcam,
                                               motive_data, bonsai_data, sets, frame_time_list)
    if count == 0:
        first_shift_idx = shift_idx
    print('Temporal shift:' + str(shift_idx))
    # save the shifted traces
    # depending on the sign of the shift, choose which trace to shift
    if first_shift_idx > 0:
        Ya = Ya[first_shift_idx:-first_shift_idx]
        Yb = Yb[:-first_shift_idx]
        thirdD_shift = first_shift_idx

    else:
        Ya = Ya[:-first_shift_idx]
        Yb = Yb[-first_shift_idx:-first_shift_idx]
        thirdD_shift = None

    aligned_traces.append([Ya, Yb])
    # # plot the aligned traces
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.plot(normalize_trace(Ya[first_shift_idx:]))
    # # ax.plot(normalize_trace(Yb))
    # ax.plot(Ya)
    # ax.plot(Yb)

# check that the alignment shifts are not too different from each other
assert(np.abs(first_shift_idx-shift_idx) < 10)
# define the third dimension for the motive data
if thirdD_shift is not None:
    if frame_rate_motive != g:
        y = interp_motive(motive_data[:, 2], frame_time_list[0], frame_time_list[0])[thirdD_shift:-thirdD_shift]
    else:
        y = motive_data[:, 2][thirdD_shift:-thirdD_shift]

# use open cv to obtain a transformation matrix of the aligned traces
# assemble the data to use for the calculation
motive_opencv = np.array([aligned_traces[0][0], aligned_traces[1][0], y]).astype('float32')
# motive_opencv = np.vstack((motive_opencv, y1))
bonsai_opencv = np.array([aligned_traces[0][1], aligned_traces[1][1]]).astype('float32')
# match their sizes and scales
min_size = np.min([motive_opencv.shape[1], bonsai_opencv.shape[1]])
# motive_opencv = scale(motive_opencv[:, :min_size].T)
# bonsai_opencv = scale(bonsai_opencv[:, :min_size].T)
motive_opencv = motive_opencv[:, :min_size].T
bonsai_opencv = bonsai_opencv[:, :min_size].T
# motive_opencv = motive_opencv[:, :min_size].T
# motive_opencv[:, :2] = normalize_matrix(motive_opencv[:, :2])
# bonsai_opencv = normalize_matrix(bonsai_opencv[:, :min_size].T)

# # find the motive points most closely corresponding the bonsai points in time
# close_idx = np.array([(np.argmin(np.abs(el-motive_time_absolute)), count)
#                      for count, el in enumerate(bonsai_time_absolute)
#                      if np.min(np.abs(el-motive_time_absolute)).microseconds < 2000])

# motive_opencv = motive_data[close_idx[:, 0], :].astype('float32')
# bonsai_opencv = target_data[close_idx[:, 1], :].astype('float32')
# invert y on the bonsai data
# bonsai_opencv = np.vstack((bonsai_opencv[:, 1], -bonsai_opencv[:, 0])).T
# time_opencv = bonsai_time_absolute[close_idx[:, 1]]

# # use the calibrateCamera function to get the camera matrix
# test_constant = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
#                  cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
#                  cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
# # test_constant = 0
# # test_constant = cv2.CALIB_USE_INTRINSIC_GUESS
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([motive_opencv], [bonsai_opencv], (1280, 1024), None, None,
#                                                    flags=test_constant)
# # undistort the camera points
# transformed_data = np.squeeze(cv2.undistortPoints(np.expand_dims(bonsai_opencv, 1), mtx, dist))
# # get the rotation matrix from the rotation vector
# rotation_matrix, jacobian = cv2.Rodrigues(np.array(rvecs))

# use the original data instead of undistorting
transformed_data = bonsai_opencv
# # calculate an approximate affine
# affine_matrix, inliers = cv2.estimateAffinePartial2D(transformed_data, motive_opencv[:, :2], ransacReprojThreshold=3,
#                                                      maxIters=2000, refineIters=10)
# print('Percentage inliers used:' + str(np.sum(inliers)*100/transformed_data.shape[0]))
# # make the transformed data homogeneous for multiplication with the affine
# transformed_data = np.squeeze(cv2.convertPointsToHomogeneous(transformed_data))
# # apply the affine matrix
# transformed_data = np.matmul(transformed_data, affine_matrix.T)

# find the homography transformation
H, mask = cv2.findHomography(bonsai_opencv, motive_opencv[:, :2], method=cv2.RANSAC)
# make the transformed data homogeneous for multiplication with the affine
transformed_data = np.squeeze(cv2.convertPointsToHomogeneous(transformed_data))
# apply the homography matrix
transformed_data = np.matmul(transformed_data, H.T)

print('MSE_x:' + str(mse(transformed_data[:, 0], motive_opencv[:, 0])))
print('MSE_y:' + str(mse(transformed_data[:, 1], motive_opencv[:, 1])))

plot_data = (motive_opencv)
# plot_data = bonsai_opencv
# plot both timelines together
fig = plt.figure()
ax = fig.add_subplot(211)
ax1 = fig.add_subplot(212)

ax.plot(plot_data[:, 0])
ax.plot(bonsai_opencv[:, 0])
ax.plot((transformed_data[:, 0]))

ax1.plot(plot_data[:, 1])
ax1.plot(bonsai_opencv[:, 1])
ax1.plot((transformed_data[:, 1]))
plt.legend(['motive', 'bonsai', 'trans'])
# ax.plot(time_opencv, scale(motive_opencv[:, 0]), marker='o')
# ax.plot(time_opencv, scale(bonsai_opencv[:, 0]), marker='o')
# ax1.plot(time_opencv, scale(motive_opencv[:, 2]))
# ax1.plot(time_opencv, scale(bonsai_opencv[:, 1]))
# ax.plot(scale(motive_data[:, 0]), marker='o')
# ax1.plot(scale(target_data[:, 0]), marker='o')
# ax1.plot(scale(motive_data[:, 2]))
# ax1.plot(scale(target_data[:, 1]))

fig = plt.figure()
# ax = fig.add_subplot(111)
ax = fig.add_subplot(111, projection='3d')
# ax.plot(motive_time_absolute, motive_data[:, 0], marker='o')
# ax.plot(bonsai_time_absolute, target_data[:, 0], marker='o')
# ax.plot(time_opencv, transformed_data[:, 0], marker='o')
# ax.plot(motive_data[:, 0], motive_data[:, 2], marker='o')
# ax.plot(transformed_data[:, 0], transformed_data[:, 1], marker='o')
# ax.plot(plot_data[close_idx[:, 0], 0], plot_data[close_idx[:, 0], 2], plot_data[close_idx[:, 0], 1], marker='.')
ax.plot(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], marker='.')
# ax.plot(transformed_data[:, 0], transformed_data[:, 1], np.zeros_like(transformed_data[:, 0]), marker='o')
ax.plot(transformed_data[:, 0], transformed_data[:, 1], plot_data[:, 2], marker='.')

# ax.plot(plot_data[:, 0], plot_data[:, 1], marker='.')
# ax.plot(transformed_data[:, 0], transformed_data[:, 1], marker='o')
plt.show()
