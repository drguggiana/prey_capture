# imports

import paths
import processing_parameters
import cv2
import pandas as pd
import numpy as np
import functions_bondjango as bd
import functions_loaders as fl
import functions_io as fi
import functions_vame as fv
import os


def save_video(frame_list, base_path, width, height, *args, frame_rate=10, frame_function=None, **kwargs):
    """Save a video with jpeg compression based on the input array"""

    # allocate memory for the output
    output_array = []
    # create the writer
    out = cv2.VideoWriter(base_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (width, height))
    # save the movie
    for frames in frame_list:
        # get the frame
        out_frame = frames.astype('float32')
        # apply the perspective matrix
        if frame_function is not None:
            out_frame = frame_function(out_frame, *args, **kwargs)
        # save in the output
        output_array.append(out_frame[:, :, 0])
        # save to file
        out.write(out_frame.astype('uint8'))

    out.release()
    return np.array(output_array).astype('uint8')


def load_video(video_path, frame_bounds):
    """Load an avi video into an array"""
    # create the video object
    cap = cv2.VideoCapture(video_path)
    # allocate memory for the corners
    frame_list = []

    # get the frames to mode
    for _ in np.arange(frame_bounds.loc[0, 'end']):
        # read the image
        frame_list.append(cap.read()[1])

    # release the capture
    cap.release()
    frame_list = frame_list[frame_bounds.loc[0, 'start']:]

    print(f'Frames after trimming bounds: {len(frame_list)}')
    return frame_list


def expand_frame_dims(frame_in):
    """Expand the dims of a single frame to 3 so it's compatible with color movies"""
    return np.repeat(np.expand_dims(frame_in, 2), 3, axis=2)


def find_motif_sequences(motif_vector):
    """Find the longest sequence of each motif present in the video"""

    present_motifs = np.unique(motif_vector)
    present_motifs = present_motifs[~np.isnan(present_motifs)]

    # allocate memory for all the locations
    location_perfile = []
    duration_perfile = []
    # for all the motifs
    for motif in present_motifs:

        # find all the starts and ends for this motif
        m_idx = (motif_vector == motif).astype(int)
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
        except AssertionError("End is located before start"):
            # print(str(idx) + '_' + str(motif))
            print(starts)
            print(ends)

        # save the locations for this motif
        location_perfile.append([el[0] for el in starts])
        duration_perfile.append([el[0] for el in ends - starts])

    return location_perfile, duration_perfile, present_motifs


def save_motif_videos(egocentric_frames, present_motifs, duration_perfile, location_perfile,
                      target_path, crop_size, slug):
    """Save the longest sequence of each motif present in the video, creating folders accordingly"""
    # for all the motifs present
    for idx, motif in enumerate(present_motifs):
        # get the maximum duration
        max_duration = np.max(duration_perfile[idx])
        # get the start of the maximum duration
        max_location = location_perfile[idx][np.argmax(duration_perfile[idx])]

        # get the video frames
        frame_idx = np.array(np.arange(max_location, max_location + max_duration))
        motif_frames = egocentric_frames[frame_idx]
        # save the movie

        # assemble the bounded movie path
        motif_folder = os.path.join(target_path, f'{int(motif):02d}')

        # check if the folder exists, if not, create
        if not os.path.isdir(motif_folder):
            os.makedirs(motif_folder)

        motif_path = os.path.join(motif_folder, '_'.join((f'{int(motif):02d}motif', slug, '.avi')))
        # save the motif movie
        save_video(motif_frames, motif_path, crop_size[0], crop_size[1], frame_function=expand_frame_dims)
    return


def create_vame_videos(search_query, video_root, temp_path, ref_corners, target_root,
                       shift_factor=5, scale_factor=20, crop_size=(200, 200), save_full_egocentric=False,
                       egocentric=True):
    """Create egocentric videos from the list of files passed
    search_query:

    """
    # TODO: fill up the docstring properly
    # LOAD THE PREPROCESSING FILE

    # get the paths from the database using search_list
    data_all = bd.query_database('analyzed_data', search_query)
    data_all = [el for el in data_all if '_preproc' in el['slug']][0]
    data_path = data_all['analysis_path']
    data_vame_name = data_all['slug'].replace('_preprocessing', '')

    data, _, _ = fl.load_preprocessing([data_path], [data_all], behavior_flag=True)
    data = data[0]

    dlc_corners = pd.read_hdf(data_path, 'corners')
    frame_bounds = pd.read_hdf(data_path, 'frame_bounds')

    # load the corresponding video

    # assemble the path
    video_path = os.path.join(video_root,
                              data_all['slug'].replace('_preprocessing', '.avi'))
    # get the video data
    frame_list = load_video(video_path, frame_bounds)

    # CREATE THE UNDISTORTED MOVIE

    # clean the folder
    fi.delete_contents(temp_path)

    # assemble the bounded movie path
    unwarped_path = os.path.join(temp_path, 'unwarped.avi')

    # get the width and height
    width = frame_list[0].shape[1]
    height = frame_list[0].shape[0]
    res_tuple = (width, height)

    # generate the perspective matrix to undistort
    perspective_matrix = cv2.getPerspectiveTransform(dlc_corners.to_numpy().T.astype('float32'),
                                                     (np.array(ref_corners).astype('float32') +
                                                      shift_factor)*scale_factor)
    # create and save the unwarped movie
    out_frames = save_video(frame_list, unwarped_path, width, height, perspective_matrix, res_tuple,
                            frame_function=cv2.warpPerspective)

    # ALIGN THE VIDEO EGOCENTRICALLY IF DESIRED
    if egocentric:

        # create the egocentric movie
        file_format = '.avi'
        use_video = True
        check_video = False

        # select the coordinate columns
        coordinate_columns = [el for el in data.columns if ('mouse_' in el) | ('cricket_' in el)]
        # scale the data so it maches the movie scaling (the 1000 factor comes from the alignment function, as it needs
        # more dynamic range to work with the real scale values instead of pixels)
        scaled_data = (data[coordinate_columns].fillna(0)+shift_factor)*scale_factor/1000
        # egocentrically crop the video data
        _, egocentric_frames, _ = fv.align_demo(scaled_data, [], data_vame_name, file_format, crop_size,
                                                use_video=use_video, check_video=check_video,
                                                vid_path=unwarped_path)

        # turn the list into an array
        egocentric_frames = np.array(egocentric_frames)

        # if egocentric movie is desired
        if save_full_egocentric:
            # create the egocentric movie

            # assemble the bounded movie path
            egocentric_path = os.path.join(temp_path, 'egocentric.avi')

            # get the width and height
            width = egocentric_frames[0].shape[1]
            height = egocentric_frames[0].shape[0]
            # save the egocentric movie
            save_video(egocentric_frames, egocentric_path, width, height, frame_function=expand_frame_dims)
        # overwrite the out frames
        out_frames = egocentric_frames
        res_tuple = crop_size

    # SAVE THE MOTIF-SPECIFIC VIDEOS

    # get the present motifs
    motif_vector = data['motifs'].to_numpy()
    # find the longest sequences
    location_perfile, duration_perfile, present_motifs = find_motif_sequences(motif_vector)
    # Create egocentric movies for all motifs
    save_motif_videos(out_frames, present_motifs, duration_perfile, location_perfile,
                      target_root, res_tuple, data_vame_name)


if __name__ == '__main__':

    # loop through the entries in the VAME search list
    for vid in processing_parameters.vame_video:
        create_vame_videos(vid, paths.videoexperiment_path, paths.temp_path,
                           paths.arena_coordinates['miniscope'], paths.vame_videos,
                           crop_size=(400, 400), save_full_egocentric=True, egocentric=False)

