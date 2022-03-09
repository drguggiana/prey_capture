import pandas as pd
import numpy as np
import paths
import processing_parameters
import os
import functions_vame as fv
import functions_bondjango as bd
import functions_io as fi
import cv2
import random


# Get the path to the involved files

# define the number of files to take
number_files = 20
# define the type of VAME
vame_type = 'prey_capture_15'
# define the number of frames to remove at beginning and end (due to VAME interval)
vame_interval = 15
# define the folder
target_folder = os.path.join(r'J:\Drago Guggiana Nilo\Prey_capture\temp_VAME', vame_type)

# load the sorting
motif_sort = np.array(processing_parameters.motif_sort)
motif_revsort = np.array(processing_parameters.motif_revsort)

# get a list of the result folders
result_list = os.listdir(os.path.join(target_folder, 'results'))
# load the search string
vame_vis_string = processing_parameters.vame_vis_string

# Load the matching prey capture data

# using the slug, perform serial calls to the database
# (super inefficient, but this is temporary as the VAME data should be included in the hdf5 file)

# for all the files

# define the search string
# query the database for data to plot
data_entries = bd.query_database('analyzed_data', vame_vis_string)

# take number_files random samples from the entries
data_sample = random.sample(data_entries, number_files)

# for the desired number of entries
for data_all in data_sample:
    data_path = data_all['analysis_path']
    data_vame_name = data_all['slug'].replace('_preprocessing', '')

    # load the data
    beh_data = pd.read_hdf(data_path, 'full_traces')
    beh_data = beh_data.iloc[vame_interval:-vame_interval, :].reset_index(drop=True)
    # load the frame bounds
    frame_bounds = pd.read_hdf(data_path, 'frame_bounds')
    # get the reference corners
    ref_corners = paths.arena_coordinates['miniscope']
    corner_points = pd.read_hdf(data_path, 'corners').to_numpy().T

    # if the file wasn't embedded, skip it
    if not os.path.isdir(os.path.join(target_folder, 'results', data_vame_name, 'VAME', 'kmeans-15')):
        continue

    # load the latent and labels
    label_list = motif_revsort[np.load(os.path.join(target_folder, 'results', data_vame_name, 'VAME',
                                                    'kmeans-15', '15_km_label_' + data_vame_name + '.npy'))]

    latent_list = np.load(os.path.join(target_folder, 'results', data_vame_name, 'VAME',
                                       'kmeans-15', 'latent_vector_' + data_vame_name + '.npy'))

    # # load the aligned data
    # data_list = np.load(os.path.join(target_folder, 'data', data_vame_name,
    #                                  data_vame_name + '-PE-seq.npy'))
    # data_list = data_list[:, vame_interval:-vame_interval]
    # print(beh_data.shape)

    # get the video
    # assemble the path
    video_path = os.path.join(paths.videoexperiment_path,
                              data_all['slug'].replace('_preprocessing', '.avi'))
    # create the video object
    cap = cv2.VideoCapture(video_path)
    # allocate memory for the corners
    frame_list = []
    # # define sigma for the edge detection parameters
    # sigma = 0.2
    # get the frames to mode
    for frames in np.arange(frame_bounds.loc[0, 'end']-1):

        # read the image
        frame_list.append(cap.read()[1])

    # release the capture
    cap.release()

    frame_list = frame_list[frame_bounds.loc[0, 'start']:]
    # keep this for the motif videos
    frames_formotif = frame_list.copy()
    # trim to interval
    frame_list = frame_list[vame_interval:-vame_interval]
    # print(len(frame_list))
    # print(frame_list[0].shape)
    # print(latent_list.shape)

    # Get the motif locations

    # get the motif number
    motif_number = latent_list.shape[1]
    # turn the movie into an array
    movie_array = np.array(frame_list)
    # allocate memory for all the locations
    location_perfile = []
    duration_perfile = []
    # for all the motifs
    for motif in np.arange(motif_number):

        # find all the starts and ends for this motif
        m_idx = (label_list == motif).astype(int)
        starts = np.argwhere(np.diff(np.pad(m_idx, (1, 1), mode='constant', constant_values=(0, 0))) == 1)
        ends = np.argwhere(np.diff(np.pad(m_idx, (1, 1), mode='constant', constant_values=(0, 0))) == -1)

        # skip if any of the arrays is empty
        if (starts.shape[0] == 0) or (ends.shape[0] == 0):
            duration_perfile.append(np.empty((0, 1)))
            location_perfile.append(np.empty((0, 1)))
            continue
        # trim the starts and ends based on ordering
        if starts[0] > ends[0]:
            if ends.shape[0] > 1:
                ends = ends[1:]
            else:
                duration_perfile.append(np.empty((0, 1)))
                location_perfile.append(np.empty((0, 1)))
                continue
        if starts[-1] > ends[-1]:
            if starts.shape[0] > 1:
                starts = starts[:-1]
            else:
                duration_perfile.append(np.empty((0, 1)))
                location_perfile.append(np.empty((0, 1)))
                continue
        # trim the starts or ends depending on size
        if starts.shape[0] > ends.shape[0]:
            starts = starts[:-1]
        if ends.shape[0] > starts.shape[0]:
            ends = ends[1:]
        # make sure the ends are always bigger than the starts
        try:
            assert np.all((ends - starts) > 0)
        except AssertionError:
            # print(str(idx) + '_' + str(motif))
            print(starts)
            print(ends)

        # save the locations for this motif
        location_perfile.append([el[0] for el in starts])
        duration_perfile.append([el[0] for el in ends - starts])

    # print(location_perfile[0])
    # print(duration_perfile[0])
    # print(label_list)
    # print(location_perfile)
    # print(duration_perfile)

    # create a distortion corrected video

    # define the path for saving the movies
    temp_path = paths.temp_path
    # clean the folder
    fi.delete_contents(temp_path)

    # create a bounded movie to align later
    # assemble the bounded movie path
    bounded_path = os.path.join(temp_path, 'bounded.avi')
    # new_mat = cv2.UMat(rot_matrix)

    # save the bounded movie
    # get the width and height
    width = frame_list[0].shape[1]
    height = frame_list[0].shape[0]

    # test = cv2.warpPerspective(frames_formotif[0].astype('float32'), rot_matrix.to_numpy(), (width, height))
    # current_matrix = rot_matrix.to_numpy()

    perspective_matrix = cv2.getPerspectiveTransform(corner_points.astype('float32'),
                                                     (np.array(ref_corners).astype('float32')+5)*20)

    # create the writer
    out = cv2.VideoWriter(bounded_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 1024))
    # save the movie
    for frames in frame_list:
        # apply the perspective matrix
        out_frame = cv2.warpPerspective(frames.astype('float32'), perspective_matrix, (1280, 1024))
        out.write(out_frame.astype('uint8'))

    out.release()

    # Align the video egocentrically

    # create the egocentric movie
    path_dlc = data_path
    path_vame = target_folder
    file_format = '.avi'
    crop_size = (1000, 1000)
    use_video = True
    check_video = False
    save_align = False

    scaled_data = (beh_data+5)*20
    # print(scaled_data.shape)

    _, egocentric_frames = fv.align_demo(scaled_data, path_vame, data_vame_name, file_format, crop_size,
                                         use_video=use_video, check_video=check_video,
                                         vid_path=bounded_path)

    # turn the list into an array
    egocentric_frames = np.array(egocentric_frames)
    # print(egocentric_frames.shape)

    # Create egocentric movies for all motifs

    # get the motifs present
    present_motifs = np.unique(label_list)
    # for all the motifs present
    for current_motif in present_motifs:
        # get the maximum duration
        max_duration = np.max(duration_perfile[current_motif])
        # get the start of the maximum duration
        max_location = location_perfile[current_motif][np.argmax(duration_perfile[current_motif])]

        # get the video frames
        frame_idx = np.array(np.arange(max_location, max_location + max_duration))
        motif_frames = egocentric_frames[frame_idx]
        # save the movie

        # assemble the bounded movie path
        # motif_path = os.path.join(temp_path, str(current_motif) + '_motif.avi')
        motif_path = os.path.join(target_folder, 'motif_videos', str(current_motif))
        # check if the folder exists, if not, create
        if not os.path.isdir(motif_path):
            os.makedirs(motif_path)
        # assemble the video path
        motif_video_path = os.path.join(motif_path,  '_'.join(('m'+str(current_motif), data_vame_name, '.avi')))

        # create the writer
        out2 = cv2.VideoWriter(motif_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                               5, (crop_size[0], crop_size[1]))
        # save the movie
        for frames in motif_frames:
            out_frame = np.repeat(np.expand_dims(frames, 2), 3, axis=2)
            out2.write(out_frame.astype('uint8'))

        out2.release()
