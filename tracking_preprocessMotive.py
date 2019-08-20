# imports
from tkinter import filedialog
from tkinter import Tk
# import re
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.ndimage.measurements import label
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
import datetime
from os import listdir
from os.path import isfile, join, basename, split
from sklearn.preprocessing import scale
import time
import csv
import cv2

# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)


def load_preprocessed(file_path_in):
    # allocate a list for all the animals
    preproc_data = []

    for animal_in in file_path_in:
        temp_data = []
        with open(animal_in) as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for ex_line in reader:
                temp_data.append(np.array(ex_line))

            preproc_data.append(np.array(temp_data))

    return preproc_data


def align_traces(frame_rate_1, frame_rate_2, data_1, data_2, sign_vector):
    # get the common denominator between the 2 rates
    g = np.gcd(frame_rate_1, frame_rate_2)
    # interpolate both traces to the common frame rate
    y1 = normalize_trace(interp(sign_vector[2] * data_1[:, sign_vector[0]], frame_rate_2 / g))
    y2 = normalize_trace(interp(sign_vector[3] * data_2[:, sign_vector[1]], frame_rate_1 / g))
    # # plot the traces
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(y1)
    # ax1.plot(y2)
    acor = np.correlate(y1, y2, mode='full')

    return np.int(np.round(np.argmax(acor) - len(y2))), y1, y2, g


def align_traces_mod(frame_rate_1, frame_rate_2, data_1, data_2, sign_vector, frame_times):
    # get the common denominator between the 2 rates
    g = np.lcm(frame_rate_1, frame_rate_2)
    # interpolate both traces to the common frame rate
    y1 = normalize_trace(interp_mod(sign_vector[2] * data_1[:, sign_vector[0]], frame_times[0], g))
    y2 = normalize_trace(interp_mod(sign_vector[3] * data_2[:, sign_vector[1]], frame_times[1], g))
    # plot the traces
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.plot(y1)
    # ax1.plot(y2)
    acor = np.correlate(y1, y2, mode='full')

    return np.int(np.round(np.argmax(acor) - len(y2))), y1, y2, g


def align_traces_motive(frame_rate_1, frame_rate_2, data_1, data_2, sign_vector, frame_times):
    # determine which rate is highest
    max_rate = np.argmax([frame_rate_1, frame_rate_2])
    # select it as the target for interpolation
    g = [frame_rate_1, frame_rate_2][max_rate]
    # make both time vectors start at 0
    frame_times = [el-el[0] for el in frame_times]
    # interpolate one trace and normalize the other depending on the max_rate
    if max_rate == 0:
        y1 = normalize_trace(sign_vector[2] * data_1[:, sign_vector[0]])
        y2 = normalize_trace(interp_motive(sign_vector[3] * data_2[:, sign_vector[1]], frame_times[1], frame_times[0]))

    else:
        y1 = normalize_trace(interp_motive(sign_vector[2] * data_1[:, sign_vector[0]], frame_times[0], frame_times[1]))
        y2 = normalize_trace(sign_vector[3] * data_2[:, sign_vector[1]])
    # calculate the cross-correlation for analysis
    acor = np.correlate(y1, y2, mode='full')
    # return the shift, the matched traces and the rate
    return np.int(np.round(np.argmax(acor) - len(y2))), y1, y2, g


def parse_line(single_line):
    parsed_line = [float(number) for number in single_line[:-1].split(',')]
    # parsed_line = re.findall('[+-]?\d+[.]\d+',single_line)
    # parsed_line = [float(s) for s in re.findall('[+-]?\d+.\d+',single_line)]
    return parsed_line


def interp(ys, mul):
    # linear extrapolation for last (mul - 1) points
    ys = list(ys)
    ys.append(2*ys[-1] - ys[-2])
    # make interpolation function
    xs = np.arange(len(ys))
    fn = interp1d(xs, ys, kind="cubic")
    # call it on desired data points
    new_xs = np.arange(len(ys) - 1, step=1./mul)
    return fn(new_xs)


def interp_mod(position, frame_times, target_rate):
    # filter the values so the interpolant is trained only on sorted frame times
    sorted_frames = np.hstack((True, np.invert(frame_times[1:] <= frame_times[:-1])))
    frame_times = frame_times[sorted_frames]
    position = position[sorted_frames]
    # create the interpolant
    interpolant = interp1d(frame_times, position, kind='cubic')
    # create the range for interpolation
    target_frame_times = np.arange(frame_times[1], frame_times[-1], step=1/target_rate)
    # return the interpolated position
    return interpolant(target_frame_times)


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


def normalize_trace(trace):
    normalized_trace = (trace-np.nanmean(trace))/np.nanstd(trace)
    return normalized_trace


def dynamic_plot(x_points, y_points, x_lim, y_lim, interval=0.1):

    xdata = []
    ydata = []
    plt.show()

    axes = plt.gca()
    axes.set_xlim(x_lim[0], x_lim[1])
    axes.set_ylim(y_lim[0], y_lim[1])
    line, = axes.plot(xdata, ydata, 'r-')

    for i in range(x_points[0].shape[0]):
        xdata.append(x_points[i])
        ydata.append(y_points[i])
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(interval)
    return None


def animation_plotter(motivedata, webcamdata, xlim, ylim):
    # First set up the figure, the axis, and the plot element we want to animate
    fig0 = plt.figure()
    ax0 = plt.axes(xlim=xlim, ylim=ylim)
    line0, = ax0.plot([], [], lw=2)
    line1, = ax0.plot([], [], lw=2)

    # initialization function: plot the background of each frame
    def init():
        line0.set_data([], [])
        line1.set_data([], [])
        return line0, line1

    # animation function.  This is called sequentially
    def animate(i):
        # x = np.linspace(0, 2, 1000)
        # y = np.sin(2 * np.pi * (x - 0.01 * i))

        line0.set_data(motivedata[:i, 0], motivedata[:i, 1])
        line1.set_data(webcamdata[:i, 0], webcamdata[:i, 1])
        return line0, line1

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig0, animate, init_func=init,
                                   frames=motivedata.shape[0], interval=10, blit=True)

    plt.show()
    return anim


# define the outcome keyword to search for
outcome_keyword = 'all'
# define the condition keyword to search for
condition_keyword = ''
condition_list = ['dark', 'vr']
# load the data
base_path = r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed'
# file_path = [r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed\05_24_2019_16_34_35_DG_190417_c_succ_preproc.csv']
file_path = [r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed\08_01_2019_20_48_47_DG_190416_a_succ_preproc.csv']
# file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("preproc files", "*.csv"), ))

# define the figure save path
# figure_save = r'C:\Users\drguggiana\Dropbox\Bonhoeffer_things\Presentations\Figures'
figure_save = r'J:\Drago Guggiana Nilo\Prey_capture\Aligned traces'

# filter the results by outcome (only use all for performance plot though)
if outcome_keyword != 'all':
    file_path = [file for file in file_path if outcome_keyword in file]

# filter the files by the desired condition
if condition_keyword == '':
    file_path = [file for file in file_path if sum([1 for word in condition_list if word in file]) == 0]
elif condition_keyword != 'all':
    file_path = [file for file in file_path if condition_keyword in file]

# actually load the data
bonsai_data = load_preprocessed(file_path)
# find the matching motive files
# get the list of files in the motive folder
motive_path = r'J:\Drago Guggiana Nilo\Prey_capture\Motive'
motive_files = listdir(motive_path)
# get the times at which the files were produced
motive_times = [datetime.datetime.strptime(el[:14], '%Y%m%dT%H%M%S') for el in motive_files]
motive_files = [join(motive_path, el) for el in motive_files]


# # define loading path and select file
# file_path_motive = [r'J:\Drago Guggiana Nilo\Prey_capture\Motive\20190524T163441_DG_190417_c_succ.txt']


# for all the files
for file_idx, files in enumerate(bonsai_data):

    # get the creation time of the bonsai file
    bonsai_time = datetime.datetime.strptime(basename(file_path[file_idx])[:18], '%m_%d_%Y_%H_%M_%S')

    # find the closest motive time to the bonsai time
    motive_idx = np.argmin(np.abs([el-bonsai_time for el in motive_times]))
    animal = motive_files[motive_idx]

    # allocate a list for all the animals

    parsed_data = []
    with open(animal) as f:
        for line in f:
            if line[0] == '0':
                continue
            parsed_data.append(parse_line(line))
    target_data = np.array(parsed_data)

    # align the motive and webcam traces

    # get the corresponding data
    # if the origin date is from before 07062019, use the delta time calculation
    if datetime.datetime(2019, 6, 7, 0, 0, 0, 0) > motive_times[motive_idx]:
        timestamp = np.array([el - target_data[0, 0] for el in target_data[:, 0]]) * 1e-7
        # define the sets of dimensions for the matching coordinates (motive, then webcam, then signs)
        coordinate_list = [[1, 0, 1, -1], [0, 1, 1, -1]]

    else:
        timestamp = target_data[:, 0]
        # frame_rate_motive = np.int(1 / np.median(np.diff(timestamp)))
        # define the sets of dimensions for the matching coordinates (motive, then webcam, then signs)
        coordinate_list = [[1, 0, 1, 1], [0, 1, 1, -1]]

    # calculate the motive frame rate
    frame_rate_motive = np.int(np.round(1 / np.mean(np.diff(timestamp))))

    # get the motive data
    motive_data = target_data[:, [1, 3]]

    # get the bonsai time
    webcam_time = files[:, 4]
    # calculate delta time
    webcam_time = np.array([el-webcam_time[0] for el in webcam_time])
    # get the bonsai data
    webcam_data = files[:, 0:4]

    # save the frame times in a common list
    frame_time_list = [timestamp, webcam_time]
    # print the frame rates
    frame_rate_webcam = np.int(np.round(1/np.median(np.diff(webcam_time))))
    print('Motive frame rate:' + str(frame_rate_motive))
    print('Bonsai frame rate:' + str(frame_rate_webcam))
    # # plot the frame times
    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # ax.hist(np.diff(timestamp), bins=100)
    # ax = fig.add_subplot(212)
    # ax.hist(np.diff(webcam_time), bins=100)
    # plt.show()

    # allocate memory for the aligned traces
    aligned_traces = []
    # initialize the first shift very high in case there's an error finding the actual shift
    first_shift_idx = 1000
    # for all the coordinate sets (i.e. dimensions)
    for count, sets in enumerate(coordinate_list):
        # shift_idx, Ya, Yb, g = align_traces(frame_rate_motive, frame_rate_webcam,
        #                                     motive_data, webcam_data, sets)
        shift_idx, Ya, Yb, g = align_traces_motive(frame_rate_motive, frame_rate_webcam,
                                                   motive_data, webcam_data, sets, frame_time_list)
        if count == 0:
            first_shift_idx = shift_idx
        print(shift_idx)
        # save the shifted traces
        # depending on the sign of the shift, choose which trace to shift
        if first_shift_idx > 0:
            Ya = Ya[first_shift_idx:]
            Yb = Yb
            thirdD_shift = first_shift_idx

        else:
            Ya = Ya
            Yb = Yb[-first_shift_idx:]
            thirdD_shift = None

        aligned_traces.append([Ya, Yb])
        # plot the aligned traces
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # ax.plot(normalize_trace(Ya[first_shift_idx:]))
        # ax.plot(normalize_trace(Yb))
        ax.plot(Ya)
        ax.plot(Yb)

    # interpolate the third dimension for the motive data
    # y = scale(interp_mod(target_data[:, 2], frame_time_list[0], frame_rate_webcam / g)[thirdD_shift:])
    # if the target rate is the motive rate, don't interpolate, just normalize
    if frame_rate_motive != g:
        y = scale(interp_motive(target_data[:, 2], frame_time_list[0], frame_time_list[0])[thirdD_shift:])
    else:
        y = scale(target_data[:, 2][thirdD_shift:])

    # use open cv to obtain a transformation matrix of the aligned traces
    # assemble the data to use for the calculation
    motive_opencv = np.array([aligned_traces[0][0], aligned_traces[1][0], y]).astype('float32')
    # motive_opencv = np.vstack((motive_opencv, y1))
    webcam_opencv = np.array([aligned_traces[0][1], aligned_traces[1][1]]).astype('float32')
    # match their sizes and scales
    min_size = np.min([motive_opencv.shape[1], webcam_opencv.shape[1]])
    # motive_opencv = scale(motive_opencv[:, :min_size].T)
    # webcam_opencv = scale(webcam_opencv[:, :min_size].T)
    motive_opencv = motive_opencv[:, :min_size].T
    webcam_opencv = webcam_opencv[:, :min_size].T

    # opencv attempt at better aligning the traces, but it kinda makes it worse, probably given the fact that the
    # points are not actually equivalent between the 2 systems
    # test_constant = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
    #                                                    cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 |
    #                                                    cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([motive_opencv], [webcam_opencv], (1, 1), None, None,
    #                                                    flags=test_constant)
    #
    # transformed_data = scale(np.squeeze(cv2.undistortPoints(np.expand_dims(webcam_opencv, 1), mtx, dist)))
    motive_affine = motive_opencv[:, :2]
    # affine_matrix, inliers = cv2.estimateAffinePartial2D(webcam_opencv, motive_affine)
    affine_matrix, inliers = cv2.estimateAffine2D(webcam_opencv, motive_affine)
    # transformed_data = np.matmul(affine_matrix, np.hstack((webcam_opencv, np.ones((webcam_opencv.shape[0], 1)))).T).T
    transformed_data = np.matmul(np.hstack((webcam_opencv, np.ones((webcam_opencv.shape[0], 1)))), affine_matrix.T)
    # Plot the aligned sets of points in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(webcam_opencv[:, 0], webcam_opencv[:, 1])
    # ax.plot(motive_opencv[:, 0], motive_opencv[:, 1], motive_opencv[:, 2])

    # Plot the set of points in 2D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.plot(webcam_opencv[:, 0], webcam_opencv[:, 1])
    ax.plot(motive_opencv[:, 0], motive_opencv[:, 1])
    ax.plot(transformed_data[:, 0], transformed_data[:, 1])

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # dynamic_plot(motive_opencv[:, 0], motive_opencv[:, 1], [-4, 4], [-4, 4], interval=0.01)

    anim = animation_plotter(motive_opencv, webcam_opencv, (-4, 4), (-4, 4))
    # # plot the distance between the traces over time
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # traces = np.hstack((webcam_opencv, motive_opencv))
    # ax.plot(np.array([np.linalg.norm(el[[0, 1]] - el[[2, 3]]) for el in traces]))

    # fig.savefig(join(figure_save, basename(animal)[:-4]+'.png'), bbox_inches='tight')
    # plt.close()
    plt.show()





