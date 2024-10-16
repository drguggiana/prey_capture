"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & J. Kürsch & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""

import cv2 as cv
import numpy as np
import pandas as pd
import tqdm
import os
import h5py
import matplotlib.pyplot as plt
import time


# Returns cropped image using rect tuple
def crop_and_flip(rect, src, points, ref_index):
    # Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)
    # center_movie = tuple([1024-center[0]*1024/1000/40, center[1]*1280/1000/40])
    # M_movie = cv.getRotationMatrix2D(center_movie, theta, 1)

    # shift DLC points
    x_diff = center[0] - size[0] // 2
    y_diff = center[1] - size[1] // 2

    dlc_points_shifted = []

    for i in points:
        point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]

        point[0] -= x_diff
        point[1] -= y_diff

        dlc_points_shifted.append(point)

    # Perform rotation on src image
    # out = src
    dst = cv.warpAffine(src.astype('float32'), M, src.shape[:2])

    # plt.imshow(out)
    out = cv.getRectSubPix(dst, size, center)
    # print(np.max(out))
    # check if flipped correctly, otherwise flip again
    if dlc_points_shifted[ref_index[1]][0] >= dlc_points_shifted[ref_index[0]][0]:
        rect = ((size[0] // 2, size[0] // 2), size, 180)
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # Get rotation matrix
        M = cv.getRotationMatrix2D(center, theta, 1)

        # shift DLC points
        x_diff = center[0] - size[0] // 2
        y_diff = center[1] - size[1] // 2

        points = dlc_points_shifted
        dlc_points_shifted = []

        for i in points:
            point = cv.transform(np.array([[[i[0], i[1]]]]), M)[0][0]

            point[0] -= x_diff
            point[1] -= y_diff

            dlc_points_shifted.append(point)

        # Perform rotation on src image
        # center_movie = tuple([center[0] * 1024 / 1000 / 40, center[1] * 1280 / 1000 / 40])
        # M_movie = cv.getRotationMatrix2D(center_movie, theta, 1)
        dst = cv.warpAffine(out.astype('float32'), M, out.shape[:2])
        out = cv.getRectSubPix(dst, size, center)

    return out, dlc_points_shifted


# Helper function to return indexes of nans
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


# Interpolates all nan values of given array
def interpol(arr):
    y = np.transpose(arr)

    nans, x = nan_helper(y[0])
    y[0][nans] = np.interp(x(nans), x(~nans), y[0][~nans])
    nans, x = nan_helper(y[1])
    y[1][nans] = np.interp(x(nans), x(~nans), y[1][~nans])

    arr = np.transpose(y)

    return arr


def background(video_path, num_frames=1000):
    """
    Compute background image from fixed camera
    """
    import scipy.ndimage
    # video_path = os.path.join(path_to_file, 'videos', filename + file_format)

    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(video_path))

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()

    height, width, _ = frame.shape
    frames = np.zeros((height, width, num_frames))

    for i in tqdm.tqdm(range(num_frames),
                       disable=not True, desc='Compute background image for video %s' % os.path.basename(video_path)):
        rand = np.random.choice(frame_count, replace=False)
        capture.set(1, rand)
        ret, frame = capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frames[..., i] = gray

    print('Finishing up!')
    medFrame = np.median(frames, 2)
    background = scipy.ndimage.median_filter(medFrame, (5, 5))

    # np.save(path_to_file+'videos/'+'background/'+filename+'-background.npy',background)

    capture.release()
    return background


def align_mouse(path_to_file, filename, file_format, crop_size, pose_list, pose_ref_index,
                pose_flip_ref, bg, frame_count, use_video=True, interp_flag=True, vid_path=None):
    # returns: list of cropped images (if video is used) and list of cropped DLC points
    #
    # parameters:
    # path_to_file: directory
    # filename: name of video file without format
    # file_format: format of video file
    # crop_size: tuple of x and y crop size
    # dlc_list: list of arrays containg corresponding x and y DLC values
    # dlc_ref_index: indices of 2 lists in dlc_list to align mouse along
    # dlc_flip_ref: indices of 2 lists in dlc_list to flip mouse if flip was false
    # bg: background image to subtract
    # frame_count: number of frames to align
    # use_video: boolean if video should be cropped or DLC points only

    images = []
    points = []

    # interpolate if the flag is present
    if interp_flag:
        for i in pose_list:
            for j in i:
                if j[2] <= 0.8:
                    j[0], j[1] = np.nan, np.nan

        for i in pose_list:
            i = interpol(i)

    if use_video:
        # generate the video path
        if vid_path is None:
            video_path = os.path.join(path_to_file, 'videos', filename + file_format)
        else:
            video_path = vid_path
        # capture = cv.VideoCapture(path_to_file+'videos/'+filename+file_format)
        capture = cv.VideoCapture(video_path)

        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(video_path))

    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc='Align frames'):

        if use_video:
            # Read frame
            try:
                ret, frame = capture.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame - bg
                frame[frame <= 0] = 0
            except:
                print("Couldn't find a frame in capture.read(). #Frame: %d" % idx)
                continue
        else:
            frame = np.zeros((1, 1))

        # Read coordinates and add border
        pose_list_bordered = []

        for i in pose_list:
            pose_list_bordered.append((int(i[idx][0] + crop_size[0]), int(i[idx][1] + crop_size[1])))

        img = cv.copyMakeBorder(frame, crop_size[1], crop_size[1], crop_size[0], crop_size[0], cv.BORDER_CONSTANT, 0)

        punkte = []
        for i in pose_ref_index:
            coord = [pose_list_bordered[i][0], pose_list_bordered[i][1]]
            punkte.append(coord)
        punkte = [punkte]
        punkte = np.asarray(punkte)

        # calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)

        # change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)

        center, size, theta = rect

        # crop image
        out, shifted_points = crop_and_flip(rect, img, pose_list_bordered, pose_flip_ref)
        # print(np.max(out))
        images.append(out)
        points.append(shifted_points)

    time_series = np.zeros((len(pose_list) * 2, frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx:idx + 2, i] = points[i][j]
            idx += 2

    return images, points, time_series


# play aligned video
def play_aligned_video(a, n, frame_count):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (0, 0, 0), (255, 255, 255), (127, 0, 127), (0, 127, 127)]

    for i in range(frame_count):
        # Capture frame-by-frame
        ret, frame = True, a[i]
        if ret:

            # Display the resulting frame
            frame = cv.cvtColor(frame.astype('uint8'), cv.COLOR_GRAY2BGR)
            im_color = cv.applyColorMap(frame, cv.COLORMAP_JET)
            # im_color = frame

            for c, j in enumerate(n[i]):
                j[0] = j[0]*1024/40/1000
                j[1] = j[1]*1280/40/1000
                cv.circle(im_color, (j[0], j[1]), 5, colors[c], -1)

            cv.imshow('Frame', im_color)
            time.sleep(0.1)
            # plt.imshow(frame)
            # plt.show()

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    cv.destroyAllWindows()


def align_demo(path_to_dlc, path_to_file, filename, file_format,
               crop_size, use_video=False, check_video=False, vid_path=None):
    # read out data
    # data = pd.read_csv(path_to_file+'videos/pose_estimation/'+filename+'-DC.csv', skiprows = 2)

    # process files differentially depending on the extension
    if isinstance(path_to_dlc, pd.DataFrame):
        # the path input is actually the data
        data = path_to_dlc
        # get the column names
        column_list_all = list(data.columns)
        # # get only the columns with mouse information
        # column_list = [True if ('mouse' in el) and (('x' in el) or ('y' in el)) else False
        #                for el in column_list]
        # column_list = [True for el in column_list]
        column_list = [True if (('x' in el) or ('y' in el)) else False
                       for el in column_list_all]
        column_list_out = [el for el in column_list_all if (('x' in el) or ('y' in el))]

        # set interpolation flag
        interp_flag = False
        # define the multiplier for the coordinates
        coord_multiplier = 1000
    elif '_dlc.h5' in path_to_dlc:
        # data = pd.read_hdf(os.path.join(path_to_dlc, filename + '_dlc.h5'))
        data = pd.read_hdf(path_to_dlc)
        # get the column names
        column_list_all = [el[1] for el in np.array(data.columns)]
        # get only the columns with mouse information
        column_list = [True if 'mouse' in el else False
                       for el in column_list_all]
        column_list_out = [el for el in column_list_all if 'mouse' in el]
        # data_mat = data_mat[:, 1:]
        # set interpolation flag
        interp_flag = True
        # define the multiplier for the coordinates
        coord_multiplier = 1
    else:
        # data = pd.read_hdf(path_to_dlc, 'full_traces')
        with h5py.File(path_to_dlc, 'r') as f:
            values = np.array(f['full_traces/block0_values'])
            labels = np.array(f['full_traces/block0_items']).astype(str)
            data = pd.DataFrame(values, columns=labels)

        # get the column names
        column_list_all = list(data.columns)
        # # get only the columns with mouse information
        # column_list = [True if ('mouse' in el) and (('x' in el) or ('y' in el)) else False
        #                for el in column_list]
        # column_list = [True for el in column_list]
        column_list = [True if (('x' in el) or ('y' in el)) else False
                       for el in column_list_all]
        column_list_out = [el for el in column_list_all if (('x' in el) or ('y' in el))]
        # set interpolation flag
        interp_flag = False
        # define the multiplier for the coordinates
        coord_multiplier = 1000

    data = data.iloc[:, column_list]
    # convert to numpy and multiply by 1000 to avoid rounding artifacts
    data_mat = pd.DataFrame.to_numpy(data)*coord_multiplier
    # # get the filename from the dlc path
    # filename = os.path.splitext(os.path.basename(path_to_dlc))[0]
    # get the coordinates for alignment from data table
    pose_list = []

    # select the factor depending on the interpolation
    if interp_flag:
        factor = 3
    else:
        factor = 2

    for i in range(int(data_mat.shape[1] / factor)):
        pose_list.append(data_mat[:, i * factor:(i + 1) * factor])

        # list of reference coordinate indices for alignment
    # 0:snout, 1:barL, 2:barR, 3:head, 4:body, 5:body2. 6:body3, 7:base
    # 8:cricket_head, 9:cricket_body

    pose_ref_index = [0, 7]

    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = [0, 7]

    if use_video:
        if vid_path is None:
            video_path = os.path.join(path_to_file, 'videos', filename + file_format)
            bg = background(video_path)
        else:
            video_path = vid_path
            # compute background
            # bg = background(video_path)
            bg = 0
        capture = cv.VideoCapture(video_path)
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(video_path))

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    else:
        bg = 0
        frame_count = len(data)  # Change this to an arbitrary number if you first want to test the code

    a, n, time_series = align_mouse(path_to_file, filename, file_format, crop_size, pose_list, pose_ref_index,
                                    pose_flip_ref, bg, frame_count, use_video, interp_flag, vid_path=vid_path)

    if check_video:
        play_aligned_video(a, n, frame_count)
    # return dividing by 1000 to output in the same scale as input
    return time_series/coord_multiplier, a, column_list_out


def run_alignment(path_dlc, path_file, file_format, crop_size, use_video=False,
                  check_video=False, save_align=True, video_path=None):
    """Run the egocentric alignment on the target file and save the npy file to the target path"""

    # get the file_name
    file_name = os.path.splitext(os.path.basename(path_dlc))[0]
    # if '_preproc' in file_name:
    file_name = file_name.replace('_preproc', '')
    file_name = file_name.replace('_dlc', '')

    # call function and save into your VAME data folder
    egocentric_time_series, video_frames, column_list = align_demo(path_dlc, path_file, file_name, file_format,
                                                                   crop_size, use_video=use_video,
                                                                   check_video=check_video, vid_path=video_path)

    # define the output path
    output_path = os.path.join(path_file, 'data', file_name, file_name + '-PE-seq.npy')
    if save_align:
        np.save(output_path, egocentric_time_series)

    return egocentric_time_series, video_frames


if __name__ == '__main__':
    """ Happy aligning """
    # config parameters
    # path_dlc = r"J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment"
    # path_dlc = \
    #     r"J:\Drago Guggiana Nilo\Prey_capture\AnalyzedData\
    #     11_14_2019_17_24_28_miniscope_dg_190806_a_succ_nofluo_preproc.hdf5"
    # path_dlc = \
    #     r"J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\11_14_2019_17_24_28_miniscope_dg_190806_a_succ_nofluo_dlc.h5"

    # path_dlc = r"J:\Drago Guggiana Nilo\Prey_capture\AnalyzedData\09_08_2020_15_00_07_miniscope_DG_200701_a_succ_preproc.hdf5"
    # path_dlc = r"J:\Drago Guggiana Nilo\Prey_capture\AnalyzedData\11_11_2019_00_41_27_miniscope_DG_190806_a_fail_nomini_preproc.hdf5"
    # path_dlc = r"J:\Drago Guggiana Nilo\Prey_capture\AnalyzedData\12_16_2019_16_21_34_miniscope_MM_191108_a_fail_preproc.hdf5"
    path_dlc = r'J:\Drago Guggiana Nilo\Prey_capture\AnalyzedData\09_08_2020_15_26_21_miniscope_DG_200701_a_succ_preproc.hdf5'
    path_vame = r"F:\VAME_projects\VAME_prey-Dec1-2020"
    # video_path = path_dlc.replace('AnalyzedData', 'VideoExperiment').replace('_preproc.hdf5', '.avi')
    # video_path = path_dlc.replace('AnalyzedData', 'VideoExperiment').replace('_dlc.h5', '.avi')
    video_path = r"D:\temp_dlc_process\bounded.avi"
    # fname = r"03_13_2020_13_20_21_miniscope_MM_200129_a_succ"
    file_format = '.avi'
    crop_size = (200, 200)
    use_video = True
    check_video = True
    save_align = False

    egocentric_time_series = run_alignment(path_dlc, path_vame, file_format, crop_size,
                                           use_video=use_video, check_video=check_video,
                                           save_align=save_align, video_path=video_path)

    # beh_data = pd.read_hdf(path_dlc, 'full_traces')
    # beh_data = beh_data.iloc[15:-15, :].reset_index(drop=True)
    #
    # egocentric_time_series, video_frames = align_demo((beh_data+5)*20, path_vame, '', file_format, crop_size,
    #                                     use_video=use_video, check_video=check_video,
    #                                     vid_path=video_path)

    # test plot
    import matplotlib.pyplot as plt
    import functions_plotting as fp
    plt.plot(egocentric_time_series.T)
    plt.show()

    # fp.simple_animation(egocentric_time_series, interval=100)
    # fp.show()


