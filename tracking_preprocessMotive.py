# imports
from tkinter import filedialog
from tkinter import Tk
# import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from scipy.interpolate import interp1d
import datetime
from os import listdir
from os.path import isfile, join, basename

import csv
import cv2

# define the outcome keyword to search for
outcome_keyword = 'succ'
# define the condition keyword to search for
condition_keyword = ''
condition_list = ['dark', 'vr']
# load the data
# base_path = r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed'
base_path = r'E:\Prey_capture\Pre_processed'
# file_path = [join(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f[:-4]+'.csv'))]
file_path = [r'E:\Prey_capture\Pre_processed\05_24_2019_16_34_35_DG_190417_c_succ_preproc.csv']
# define the figure save path
# figure_save = r'C:\Users\drguggiana\Dropbox\Bonhoeffer_things\Presentations\Figures'
figure_save = r'C:\Users\Drago\Dropbox\Bonhoeffer_things\Presentations\Figures'


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
    g = np.gcd(frame_rate_1, frame_rate_2)
    y1 = interp(sign_vector[2]*data_1[:, sign_vector[0]], frame_rate_2 / g)
    y2 = interp(sign_vector[3]*data_2[:, sign_vector[1]], frame_rate_1 / g)

    acor = np.correlate(normalize_trace(y1), normalize_trace(y2), mode='full')

    return np.int(np.round(np.argmax(acor) - len(y2))), y1, y2


# filter the results by outcome (only use all for performance plot though)
if outcome_keyword != 'all':
    file_path = [file for file in file_path if outcome_keyword in file]

# filter the files by the desired condition
if condition_keyword == '':
    file_path = [file for file in file_path if sum([1 for word in condition_list if word in file]) == 0]
elif condition_keyword != 'all':
    file_path = [file for file in file_path if condition_keyword in file]

bonsai_data = load_preprocessed(file_path)

# define loading path and select file
# base_path = r'C:\Users\drguggiana\Documents\Motive_test1\etc'
# file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("text files", "*.txt"), ))
file_path_motive = [r'E:\Prey_capture\Motive\20190524T163441_DG_190417_c_succ.txt']


# parse the file
def parse_line(single_line):
    parsed_line = [float(number) for number in single_line[:-1].split(',')]
    # parsed_line = re.findall('[+-]?\d+[.]\d+',single_line)
    # parsed_line = [float(s) for s in re.findall('[+-]?\d+.\d+',single_line)]
    return parsed_line


# allocate a list for all the animals
animal_data = []

for animal in file_path_motive:
    parsed_data = []
    with open(animal) as f:
        for line in f:
            if line[0] == '0':
                continue
            parsed_data.append(parse_line(line))

    animal_data.append(np.array(parsed_data))


# align the motive and webcam traces

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


def normalize_trace(trace):
    normalized_trace = (trace-np.nanmean(trace))/np.nanstd(trace)
    return normalized_trace


# for all the files
for file_idx, files in enumerate(bonsai_data):

    # get the corresponding data
    target_data = animal_data[file_idx]
    timestamp = np.array([el - target_data[0, 0] for el in target_data[:, 0]])

    # # convert the timestamp to date
    # print(datetime.timedelta(microseconds = timestamp[0]))
    # print('Frame rate motive:' + str(1/np.mean(np.diff(timestamp*1e-7)))+ ' fps')
    motive_data = target_data[:, [1, 3]]

    # get the time
    # webcam_time = [datetime.datetime.strptime(el[1][:-7],'%Y-%m-%dT%H:%M:%S.%f') for el in files]
    webcam_time = files[:, 4]
    # calculate delta time
    webcam_time = np.array([el-webcam_time[0] for el in webcam_time])
    webcam_data = files[:, 0:4]
    # time = range(len(time))
    # print the frame rate
    # print('Frame rate webcam:' + str(1/np.mean(np.diff(webcam_deltatime)))+ ' fps')

    # # get just the coordinate data
    # files = np.vstack(np.array([el[0] for el in files]))

    frame_rate_motive = np.int(np.round(1/np.mean(np.diff(timestamp*1e-7))))
    frame_rate_webcam = np.int(np.round(1/np.mean(np.diff(webcam_time))))
    print(frame_rate_motive)
    print(frame_rate_webcam)

    # define the sets of dimensions for the matching coordinates (motive, then webcam, then signs)
    coordinate_list = [[1, 0, 1, -1], [0, 1, 1, -1]]
    # allocate memory for the aligned traces
    aligned_traces = []
    # for all the coordinate sets
    for sets in coordinate_list:
        shift_idx, Ya, Yb = align_traces(frame_rate_motive, frame_rate_webcam,
                                         motive_data, webcam_data, sets)
        aligned_traces.append([Ya, Yb])
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(normalize_trace(Ya[shift_idx:]))
        # ax.plot(normalize_trace(Yb))

    # use open cv to obtain a transformation matrix of the aligned traces
    # assemble the data to use for the calculation
    motive_opencv = np.array([aligned_traces[0][0], aligned_traces[1][0]]).astype(float)
    webcam_opencv = np.array([aligned_traces[0][1], aligned_traces[1][1]]).astype(float)
    affine_matrix = cv2.getAffineTransform(motive_opencv, webcam_opencv)

    # animal_tri = np.array(animal_data[0][:, 1:3]).astype(float)
    # animal_tri[np.isnan(animal_tri)] = 0
    # bonsai_tri = np.vstack(np.array([el[0] for el in animal_data_bonsai[0]]))
    #
    # bonsai_tri = np.array(bonsai_tri[:, :1]).astype(float)
    # bonsai_tri[np.isnan(bonsai_tri)] = 0
    # transform_matrix = cv2.getAffineTransform(animal_tri, bonsai_tri)

    # g = np.gcd(frame_rate_motive, frame_rate_webcam)
    # Ya = interp(motive_data[:, 1], frame_rate_webcam/g)
    # Yb = interp(-webcam_data[:, 0], frame_rate_motive/g)
    # Yfs = frame_rate_motive*frame_rate_webcam/g
    #
    # acor = np.correlate(normalize_trace(Ya), normalize_trace(Yb), mode='full')
    # # time_shift = lag(acor == max(acor))/Yfs;
    #
    # # print(time_shift)
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.plot(acor)
    # ax = fig.add_subplot(122)
    # ax.plot(Ya)
    # ax.plot(Yb/100)
    # # get the actual shift
    # # shift_idx = np.int(np.round(np.argmax(acor)/Yfs))
    # # shift_idx = np.int(np.round(np.argmax(acor)/g))
    # # shift_idx = np.int(np.round(np.argmax(acor)/g-len(acor)/2))
    # shift_idx = np.int(np.round(np.argmax(acor)-len(Yb)))
    # # shift_idx = 0
    # print('shift:' + str(shift_idx))

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(normalize_trace(Ya[shift_idx:]))
    # ax.plot(normalize_trace(Yb))

    plt.show()
    # # get the proportionality factor
    # proportionality_factor = frame_rate_motive/frame_rate_webcam

    # # find the last point out of bounds for the motive trace
    # # define the limit in x (coordinate 3)
    # x_limit = 0.575
    # # find the last instance of a point breaking the threshold in the trace
    # first_inside = np.argwhere(target_data[:, 3]<x_limit)[0][0]
    # motive_start = motive_time[first_inside]
    # print(motive_time[first_inside])

    # # find that starting point in the webcam trace
    # webcam_start = np.abs(np.array([(motive_start - el).total_seconds() for el in webcam_time])).argmin()
    # print([(motive_start - el).total_seconds() for el in webcam_time])
    # print(motive_time[0])
    # print(webcam_time[0])
    # print(webcam_start)




