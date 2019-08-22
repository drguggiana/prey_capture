# imports
from tkinter import filedialog
import datetime
from sklearn.metrics import mean_squared_error as mse
from os.path import join, basename
from os import listdir
from scipy.signal import find_peaks
from matching_functions import *
from io_functions import *
from plotting_functions import *
from misc_functions import *
from paths import *


# prevent the appearance of the tk main window
tk_killwindow()

# define the outcome keyword to search for
outcome_keyword = 'all'
# define the condition keyword to search for
condition_keyword = 'all'
# load the data
base_path = pre_processed_path

file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("preproc files", "*.csv"),))

# define the figure save path
output_save = aligned_path

# parse the file names for the desired trait
file_parser(file_path, outcome_keyword, condition_keyword)

# actually load the data
bonsai_data_all = load_preprocessed(file_path)
# find the matching motive files
# get the list of files in the motive folder
motive_path = motive_path
motive_files = listdir(motive_path)
# get the times at which the files were produced
motive_times = [datetime.datetime.strptime(el[:14], '%Y%m%dT%H%M%S') for el in motive_files]
motive_files = [join(motive_path, el) for el in motive_files]

# allocate memory to store the error messages
error_log = []

# for all the files
for file_idx, files in enumerate(bonsai_data_all):

    # check whether the cricket was detected
    try:
        assert np.all(np.isnan(files[:, 2])) == 0, 'Cricket was not tracked'
        assert files.shape[0] > 100, 'Too few data points'
    except AssertionError as e:
        error_logger(error_log, basename(file_path[file_idx])[:-4] + '_Bonsai error_' + str(e.args))
        continue
    # get the creation time of the bonsai file
    bonsai_create_time = datetime.datetime.strptime(basename(file_path[file_idx])[:18], '%m_%d_%Y_%H_%M_%S')

    # find the closest motive time to the bonsai time
    # get the delta times with the motive files and this bonsai file
    delta_withmotive = np.abs([el - bonsai_create_time for el in motive_times])
    motive_idx = np.argmin(delta_withmotive)
    try:
        assert motive_idx is not np.ndarray, 'More than 1 matching file found'
        assert delta_withmotive[motive_idx].seconds < 100, 'Delay too long, probably unmatched file'
    except AssertionError as e:
        error_logger(error_log, basename(file_path[file_idx])[:-4] + '_File matching problem_' + str(e.args))
        continue
    animal = motive_files[motive_idx]

    # allocate a list for all the animals
    parsed_data = []
    # read the motive file
    with open(animal) as f:
        for line in f:
            if line[0] == '0':
                continue
            parsed_data.append(parse_line(line))
    target_data = np.array(parsed_data)

    # align the motive and bonsai traces

    # get the corresponding data
    # if the origin date is from before 07062019, use the delta time calculation
    if datetime.datetime(2019, 6, 7, 0, 0, 0, 0) > motive_times[motive_idx]:
        motive_time = np.array([el - target_data[0, 0] for el in target_data[:, 0]]) * 1e-7
        # define the sets of dimensions for the matching coordinates (motive, then webcam, then signs)
        coordinate_list = [[1, 0, 1, -1], [0, 1, 1, -1]]

    else:
        motive_time = np.array([el - target_data[0, 0] for el in target_data[:, 0]])
        # frame_rate_motive = np.int(1 / np.median(np.diff(timestamp)))
        # define the sets of dimensions for the matching coordinates (motive, then webcam, then signs)
        coordinate_list = [[1, 0, 1, 1], [0, 1, 1, -1]]

    # calculate the motive frame rate
    frame_rate_motive = np.int(np.round(1 / np.mean(np.diff(motive_time))))

    # get the motive data
    motive_data = target_data[:, [1, 3, 2]]
    motive_angles = target_data[:, [4, 6, 5]]

    # get the bonsai time
    bonsai_time = files[:, 4]
    # calculate delta time
    bonsai_time = np.array([el - bonsai_time[0] for el in bonsai_time])
    # get the bonsai data
    bonsai_data = files[:, :4]

    # print the frame rates
    frame_rate_bonsai = np.int(np.round(1 / np.median(np.diff(bonsai_time))))
    print('Motive frame rate:' + str(frame_rate_motive))
    print('Bonsai frame rate:' + str(frame_rate_bonsai))

    # remove NaNs
    # filter the points for nans
    valid_bonsai = ~np.isnan(bonsai_data[:, 0])
    bonsai_time = bonsai_time[valid_bonsai]
    bonsai_data = bonsai_data[valid_bonsai, :]

    valid_motive = ~np.isnan(motive_data[:, 0])
    motive_time = motive_time[valid_motive]
    motive_data = motive_data[valid_motive, :]

    # scale the bonsai data to the motive data
    bonsai_data = normalize_matrix(bonsai_data, motive_data[:, :2])

    # save the frame times in a common list
    frame_time_list = [motive_time, bonsai_time]
    # if the fr is less than 40 (i.e. older experiments) just take 1 segment
    if frame_rate_motive < 40:
        n_segments = 1
        peak_list = []
    else:
        # refine the existing alignment using steady frame rates
        # determine the number of segments (i.e. variable frame rates in motive)
        # get the frame rate change trace
        ols_window = 50
        average_window = 100
        fr_change = rolling_ols(rolling_average(np.diff(motive_time), average_window), ols_window)
        # time_plot = plot_2d([[motive_time]])
        # find the peaks that are not at the edges
        peak_list, properties = find_peaks(np.abs(fr_change), distance=average_window, prominence=(0.00002, 0.0007))
        # eliminate peaks at the beginning and end of the trace
        peak_list = peak_list[(peak_list > 300) & (peak_list < fr_change.shape[0] - 300)]
        # fig = plot_2d([[fr_change, np.array([peak_list, fr_change[peak_list]]).T]], markers=[['.', '*']])
        # run the transformation on a loop, taking segments of the trace instead of the whole thing
        # define the number of segments
        n_segments = peak_list.shape[0] + 1
    if n_segments > 1:
        print('Frame rate change detected:' + str(n_segments))
    # allocate memory for the data
    transformed_data = []
    # allocate memory for the original motive data
    original_motive = []
    # # allocate memory for the original bonsai data
    # original_bonsai = []
    # allocate memory for the time
    time_vector = []
    # allocate an index for the segments
    segment_index = 0
    # copy the bonsai data to use
    bonsai_use = bonsai_data.copy()
    # for all the segments
    for seg in range(n_segments):
        # get the segment range
        if seg + 1 < n_segments:
            segment = peak_list[seg]
        else:
            segment = motive_data.shape[0]
        # save the frame times in a common list
        frame_time_list[0] = motive_time[segment_index:segment]
        # calculate the motive frame rate
        frame_rate_motive = np.int(np.round(1 / np.mean(np.diff(motive_time[segment_index:segment]))))
        try:
            # match the bonsai and motive data temporally
            motive_opencv, bonsai_opencv, shifted_time, cricket_opencv = match_traces(
                motive_data[segment_index:segment, :],
                bonsai_use[:, :2], frame_time_list,
                coordinate_list,
                bonsai_use[:, 2:])
            assert motive_opencv.any(), 'motive_opencv is empty'
            assert bonsai_opencv.any(), 'bonsai_opencv is empty'
        except AssertionError as e:
            error_logger(error_log, basename(file_path[file_idx])[:-4] +
                         "_Fragment skipped alignment was not possible_" + str(e.args))
            continue

        # align traces spatially using opencv
        # transformed_data = homography(bonsai_opencv, motive_opencv[:, :2], bonsai_opencv)
        # transformed_data = affine(bonsai_opencv, motive_opencv[:, :2], bonsai_opencv)
        try:
            transformed_opencv = partialaffine(bonsai_opencv, motive_opencv[:, :2], bonsai_opencv)
        except AssertionError as e:
            error_logger(error_log, basename(file_path[file_idx])[:-4] +
                         "_Fragment skipped transform was not possible_" + str(e.args))
            continue
        transformed_cricket = partialaffine(bonsai_opencv, motive_opencv[:, :2], cricket_opencv)

        # assemble the original, interpolated traces
        original_motive.append(motive_opencv)
        # original_bonsai.append(bonsai_opencv)
        transformed_data.append(np.hstack((transformed_opencv, transformed_cricket)))
        time_vector.append(shifted_time)
        # update the index
        segment_index = segment
    try:
        # concatenate the segments
        transformed_data = np.concatenate(transformed_data)
        motive_opencv = np.concatenate(original_motive)
        # bonsai_opencv = np.concatenate(original_bonsai)
        time_opencv = np.concatenate(time_vector)
    except ValueError as e:
        error_logger(error_log, basename(file_path[file_idx])[:-4] + '_Concatenation error_' + str(e.args))
        continue

    # calculate the quality of the fit
    fit_mse = mse(transformed_data[:, :2].flatten(), motive_opencv[:, :2].flatten())

    plot_data = motive_opencv
    # plot both timelines together
    # fig_xymatch = plot_2d([[plot_data[:, 0], bonsai_opencv[:, 0], transformed_data[:, 0]],
    #                  [plot_data[:, 1], bonsai_opencv[:, 1], transformed_data[:, 1]]], rows=2)
    #
    # fig_3daligned = plot_3d([plot_data, np.vstack((transformed_data[:, 0], transformed_data[:, 1],
    #                                         np.zeros_like(transformed_data[:, 0]))).T,
    #                          np.vstack((transformed_data[:, 2], transformed_data[:, 3],
    #                                     np.zeros_like(transformed_data[:, 0]))).T])

    top_projection = plot_2d([[plot_data[:, :2], transformed_data[:, :2], transformed_data[:, 2:]]],
                             labels=[['motive', 'bonsai', 'cricket']])
    # anim = animation_plotter(motive_opencv, transformed_data[:, :2], transformed_data[:, 2:],
    #                         (-0.6, 0.6), (-0.35, 0.25), interval=0.5)
    # save the figures
    top_projection.savefig(join(output_save, basename(file_path[file_idx])[:-4] +
                                '_mse' + str(fit_mse)[:6] + '.png'), bbox_inches='tight')
    plt.close(fig='all')
    # plt.show()

    # save the data
    # assemble the file name
    save_file = join(output_save, basename(file_path[file_idx])[:-4] + '_mse' + str(fit_mse)[:6] + '_aligned.csv')
    with open(save_file, mode='w', newline='') as f:
        file_writer = csv.writer(f, delimiter=',')
        for m, b, d, t in zip(motive_opencv, motive_angles, transformed_data, time_opencv):
            file_writer.writerow(np.hstack((m, b, d, t)))


# save the error log
error_file = join(output_save, str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')) + '_errorlog.txt')
with open(error_file, mode='w', newline='') as f:
    f.writelines(error_log)
