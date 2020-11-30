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


# Returns cropped image using rect tuple
def crop_and_flip(rect, src, points, ref_index):
    # Read out rect structures and convert
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix
    M = cv.getRotationMatrix2D(center, theta, 1)

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
    dst = cv.warpAffine(src.astype('float32'), M, src.shape[:2])
    out = cv.getRectSubPix(dst, size, center)

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


def background(path_to_file, filename, file_format='.mp4', num_frames=1000):
    """
    Compute background image from fixed camera
    """
    import scipy.ndimage
    video_path = os.path.join(path_to_file, 'videos', filename + file_format)
    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        raise Exception("Unable to open video file: {0}".format(video_path))

    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    ret, frame = capture.read()

    height, width, _ = frame.shape
    frames = np.zeros((height, width, num_frames))

    for i in tqdm.tqdm(range(num_frames), disable=not True, desc='Compute background image for video %s' % filename):
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
                pose_flip_ref, bg, frame_count, use_video=True):
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

    for i in pose_list:
        for j in i:
            if j[2] <= 0.8:
                j[0], j[1] = np.nan, np.nan

    for i in pose_list:
        i = interpol(i)

    if use_video:
        # generate the video path
        video_path = os.path.join(path_to_file, 'videos', filename + file_format)
        capture = cv.VideoCapture(video_path)
        # capture = cv.VideoCapture(path_to_file+'videos/'+filename+file_format)

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
              (0, 255, 255), (0, 0, 0), (255, 255, 255)]

    for i in range(frame_count):
        # Capture frame-by-frame
        ret, frame = True, a[i]
        if ret:

            # Display the resulting frame
            frame = cv.cvtColor(frame.astype('uint8'), cv.COLOR_GRAY2BGR)
            im_color = cv.applyColorMap(frame, cv.COLORMAP_JET)
            # im_color = frame

            for c, j in enumerate(n[i]):
                cv.circle(im_color, (j[0], j[1]), 5, colors[c], -1)

            cv.imshow('Frame', im_color)

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break
    cv.destroyAllWindows()


def align_demo(path_to_dlc, path_to_file, filename, file_format, crop_size, use_video=False, check_video=False):
    # read out data
    # data = pd.read_csv(path_to_file+'videos/pose_estimation/'+filename+'-DC.csv', skiprows = 2)

    # process files differentially depending on the extension
    if '_dlc.h5' in path_to_dlc:
        # data = pd.read_hdf(os.path.join(path_to_dlc, filename + '_dlc.h5'))
        data = pd.read_hdf(path_to_dlc)

        # get only the columns with mouse information
        column_list = [True if 'mouse' in el else False for el in [el[1] for el in np.array(data.columns)]]
        data = data.iloc[:, column_list]

        data_mat = pd.DataFrame.to_numpy(data)
        # data_mat = data_mat[:, 1:]
    else:
        data = pd.read_hdf(path_to_dlc, 'matched_calcium')

    # # get the filename from the dlc path
    # filename = os.path.splitext(os.path.basename(path_to_dlc))[0]
    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3:(i + 1) * 3])

        # list of reference coordinate indices for alignment
    # 0: snout, 1: forehand_left, 2: forehand_right,
    # 3: hindleft, 4: hindright, 5: tail

    pose_ref_index = [0, 2]

    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = [0, 2]

    if use_video:
        # compute background
        bg = background(path_to_file, filename, file_format=file_format)
        video_path = os.path.join(path_to_file, 'videos', filename + file_format)
        capture = cv.VideoCapture(video_path)
        if not capture.isOpened():
            raise Exception("Unable to open video file: {0}".format(video_path))

        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    else:
        bg = 0
        frame_count = len(data)  # Change this to an arbitrary number if you first want to test the code

    a, n, time_series = align_mouse(path_to_file, filename, file_format, crop_size, pose_list, pose_ref_index,
                                    pose_flip_ref, bg, frame_count, use_video)

    if check_video:
        play_aligned_video(a, n, frame_count)

    return time_series


def run_alignment(path_dlc, path_file, file_format, crop_size, use_video=False, check_video=False):
    """Run the egocentric alignment on the target file and save the npy file to the target path"""

    # get the file_name
    file_name = os.path.splitext(os.path.basename(path_dlc))[0]
    if '_preproc' in file_name:
        file_name = file_name.replace('_preproc', '')

    # call function and save into your VAME data folder
    egocentric_time_series = align_demo(path_dlc, path_file, file_name, file_format,
                                        crop_size, use_video=use_video, check_video=check_video)

    # define the output path
    output_path = os.path.join(path_file, 'data', file_name, file_name + '-PE-seq.npy')
    np.save(output_path, egocentric_time_series)

    return egocentric_time_series


if __name__ == '__main__':
    """ Happy aligning """
    # config parameters
    # path_dlc = r"J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment"
    path_dlc = \
        r"J:\Drago Guggiana Nilo\Prey_capture\AnalyzedData\03_13_2020_13_20_21_miniscope_MM_200129_a_succ_preproc.hdf5"
    path_vame = r"F:\VAME_projects\VAME_prey-Nov24-2020"
    # fname = r"03_13_2020_13_20_21_miniscope_MM_200129_a_succ"
    file_format = '.avi'
    crop_size = (1280, 1024)
    use_video = False
    check_video = False

    egocentric_time_series = run_alignment(path_dlc, path_vame, file_format, crop_size,
                                           use_video=use_video, check_video=check_video)

    # test plot
    import matplotlib.pyplot as plt

    plt.plot(egocentric_time_series.T)
    plt.show()


