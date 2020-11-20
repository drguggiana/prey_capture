# imports
from tkinter import filedialog
from os.path import join, basename
from functions_io import *
from functions_plotting import *
from functions_misc import *
from functions_kinematic import *
import paths
import pandas as pd


# def kinematic_calculations(name, data):
#     """Calculate basic kinematic parameters of mouse and cricket"""
#
#     # define which coordinates to use depending on the available data
#     # TODO: sort this out, should be a direct mapping between them
#     if 'mouse_x_m' in data.columns:
#         mouse_coord_hd = data.loc[:, ['mouse_x_m', 'mouse_y_m']].to_numpy()
#     else:
#         mouse_coord_hd = data.loc[:, ['mouse_x', 'mouse_y']].to_numpy()
#
#     # get the coordinates of the cricket
#     cricket_coord = data.loc[:, ['cricket_x', 'cricket_y']].to_numpy()
#     # get the mouse coordinates too (from bonsai)
#     mouse_coord = data.loc[:, ['mouse_x', 'mouse_y']].to_numpy()
#
#     # calculate headings and head directions depending on available data
#     if 'mouse_head_x' in data.columns:
#         mouse_heading = heading_calculation(data[['mouse_x', 'mouse_y']].to_numpy(),
#                                             data[['mouse_base_x', 'mouse_base_y']].to_numpy())
#     else:
#         mouse_heading = np.concatenate((heading_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]), [0]))
#
#     # zero the NaNs
#     mouse_heading[np.isnan(mouse_heading)] = 0
#
#     # calculate the heading to the cricket
#
#     # get the time
#     time_vector = data.time_vector.to_numpy()
#     # get the cricket angle to the mouse
#     cricket_heading = heading_calculation(cricket_coord, mouse_coord)
#     cricket_heading[np.isnan(cricket_heading)] = 0
#
#     # get the delta angle
#     delta_heading = cricket_heading - mouse_heading
#     # effectively wrap around -180 and 180 to put the center at 0
#     delta_heading[delta_heading > 180] += -180
#     delta_heading[delta_heading < -180] += 180
#
#     # get the mouse-cricket distance
#     mouse_cricket_distance = distance_calculation(cricket_coord, mouse_coord)
#
#     mouse_speed = np.concatenate(
#         ([0], distance_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]) /
#          (time_vector[1:] - time_vector[:-1])))
#     mouse_acceleration = np.concatenate(([0], np.diff(mouse_speed)))
#
#     cricket_speed = np.concatenate(([0], distance_calculation(cricket_coord[1:, :], cricket_coord[:-1, :])
#                                     / (time_vector[1:] - time_vector[:-1])))
#     cricket_acceleration = np.concatenate(([0], np.diff(cricket_speed)))
#
#     # save the traces to a variable
#     angle_traces = np.vstack((mouse_heading, cricket_heading, delta_heading, mouse_cricket_distance,
#                               mouse_speed, mouse_acceleration, cricket_speed, cricket_acceleration, time_vector)).T
#     # replace infinity values with NaNs (in the kinematic traces)
#     angle_traces[np.isinf(angle_traces)] = np.nan
#
#     # create a dataframe with the results
#     kine_data = pd.DataFrame(angle_traces, columns=['mouse_heading', 'cricket_heading', 'delta_heading',
#                                                     'mouse_cricket_distance', 'mouse_speed', 'mouse_acceleration',
#                                                     'cricket_speed', 'cricket_acceleration', 'time_vector'])
#
#     # if the motive flag is on, also calculate head direction
#     if 'mouse_x_m' in data.columns:
#         # get the head direction around the vertical axis
#         head_direction = data['mouse_zrot_m']
#         # get the head height
#         head_height = data['mouse_z_m']
#         # correct the angle trace for large jumps
#         # # define the jump threshold
#         # jump_threshold = 10
#         # head_direction = wrap(jump_killer(head_direction, jump_threshold))
#         # get the offset to correct the head direction due to the center of mass of the tracker
#         # (to single degree accuracy)
#
#         # get offset via correlation
#         angle_offset = np.argmax([np.corrcoef(np.vstack((head_direction - el, mouse_heading)))[0][1]
#                                   for el in range(360)])
#
#         head_direction = head_direction - angle_offset
#         delta_head = cricket_heading - head_direction
#         # correct the delta head to center on 0
#         delta_head[delta_head > 180] += -180
#         delta_head[delta_head < -180] += 180
#
#         # include the motive info in the final dataframe
#         kine_data['head_direction'] = head_direction
#         kine_data['head_height'] = head_height
#         kine_data['delta_head'] = delta_head
#
#     # if the head data from DLC is available, calculate head direction and delta_look
#     if 'mouse_head_x' in data.columns:
#         # get the head angle with respect to body
#         head_direction = heading_calculation(data[['mouse_head_x', 'mouse_head_y']].to_numpy(),
#                                              data[['mouse_x', 'mouse_y']].to_numpy())
#         # also get the cricket to head angle
#         head_cricket = heading_calculation(data[['cricket_x', 'cricket_y']].to_numpy(),
#                                            data[['mouse_head_x', 'mouse_head_y']].to_numpy())
#         # finally, get the delta_head angle
#         delta_head = head_cricket - head_direction
#         # effectively wrap around -180 and 180 to put the center at 0
#         delta_head[delta_head > 180] += -180
#         delta_head[delta_head < -180] += 180
#         # save in the dataframe
#         kine_data['head_direction'] = head_direction
#         kine_data['delta_head'] = delta_head
#
#     # if the head direction and delta_head were not added, pad with NaN
#     if 'delta_head' not in kine_data.columns:
#         kine_data['head_direction'] = np.nan
#         kine_data['delta_head'] = np.nan
#
#     kine_data = pd.concat([data.loc[:, ['mouse_x', 'mouse_y', 'cricket_x', 'cricket_y']], kine_data], axis=1)
#
#     # save the data to file
#     kine_data.to_hdf(name, key='full_traces', mode='w', format='table')
#
#     # save the dataframe
#     return kine_data


def cricket_processing(cricket_coord, data, mouse_coord, mouse_heading, kine_data, vr=False, cricket_name='cricket_0'):
    """Process the cricket related variables"""
    # get the coordinates of the cricket
    # cricket_coord = data.loc[:, ['cricket_x', 'cricket_y']].to_numpy()

    # get the cricket angle to the mouse
    cricket_heading = heading_calculation(cricket_coord, mouse_coord)
    cricket_heading[np.isnan(cricket_heading)] = 0

    # get the delta angle
    delta_heading = cricket_heading - mouse_heading
    # effectively wrap around -180 and 180 to put the center at 0
    delta_heading[delta_heading > 180] += -180
    delta_heading[delta_heading < -180] += 180

    # get the mouse-cricket distance
    mouse_cricket_distance = distance_calculation(cricket_coord, mouse_coord)

    cricket_speed = np.concatenate(([0], distance_calculation(cricket_coord[1:, :], cricket_coord[:-1, :]) /
                                    (kine_data.time_vector.to_numpy()[1:] - kine_data.time_vector.to_numpy()[:-1])))
    cricket_acceleration = np.concatenate(([0], np.diff(cricket_speed)))

    # assemble the array for the dataframe
    cricket_array = np.vstack((cricket_coord[:, 0], cricket_coord[:, 1], cricket_heading, cricket_speed,
                               cricket_acceleration, mouse_cricket_distance, delta_heading)).T

    # create the dataframe
    cricket_data = pd.DataFrame(cricket_array, columns=[cricket_name+'_x', cricket_name+'_y', cricket_name+'_heading',
                                                        cricket_name+'_speed', cricket_name+'_acceleration',
                                                        cricket_name+'_mouse_distance', cricket_name+'_delta_heading'])

    # if the motive flag is on, also calculate head direction
    if 'mouse_x_m' in data.columns:
        # calculate the delta head
        delta_head = cricket_heading - kine_data.head_direction
        # correct the delta head to center on 0
        delta_head[delta_head > 180] += -180
        delta_head[delta_head < -180] += 180
        # save in the data frame
        cricket_data[cricket_name+'_delta_head'] = delta_head

    # if the head data from DLC is available and there's no motive, calculate head direction and delta_look
    if 'mouse_head_x' in data.columns and 'mouse_x_m' not in data.columns and vr is False:

        # also get the cricket to head angle
        head_cricket = heading_calculation(data[['cricket_0_x', 'cricket_0_y']].to_numpy(),
                                           data[['mouse_head_x', 'mouse_head_y']].to_numpy())
        # finally, get the delta_head angle
        delta_head = head_cricket - kine_data.head_direction
        # effectively wrap around -180 and 180 to put the center at 0
        delta_head[delta_head > 180] += -180
        delta_head[delta_head < -180] += 180
        # save in the dataframe
        cricket_data[cricket_name+'_delta_head'] = delta_head

    return cricket_data


def kinematic_calculations(name, data):
    """Calculate basic kinematic parameters of mouse and cricket"""

    # define which coordinates to use depending on the available data
    # TODO: sort this out, should be a direct mapping between them
    if 'mouse_x_m' in data.columns:
        mouse_coord_hd = data.loc[:, ['mouse_x_m', 'mouse_y_m']].to_numpy()
    else:
        mouse_coord_hd = data.loc[:, ['mouse_x', 'mouse_y']].to_numpy()

    # get the mouse coordinates too (from bonsai)
    mouse_coord = data.loc[:, ['mouse_x', 'mouse_y']].to_numpy()

    # calculate headings and head directions depending on available data
    if 'mouse_head_x' in data.columns:
        mouse_heading = heading_calculation(data[['mouse_x', 'mouse_y']].to_numpy(),
                                            data[['mouse_base_x', 'mouse_base_y']].to_numpy())
    else:
        mouse_heading = np.concatenate((heading_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]), [0]))

    # zero the NaNs
    mouse_heading[np.isnan(mouse_heading)] = 0

    # calculate the heading to the cricket

    # get the time
    time_vector = data.time_vector.to_numpy()

    mouse_speed = np.concatenate(
        ([0], distance_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]) /
         (time_vector[1:] - time_vector[:-1])))
    mouse_acceleration = np.concatenate(([0], np.diff(mouse_speed)))

    # save the traces to a variable
    # angle_traces = np.vstack((mouse_heading, cricket_heading, delta_heading, mouse_cricket_distance,
    #                           mouse_speed, mouse_acceleration, cricket_speed, cricket_acceleration, time_vector)).T
    angle_traces = np.vstack((mouse_heading, mouse_speed, mouse_acceleration, time_vector)).T
    # replace infinity values with NaNs (in the kinematic traces)
    angle_traces[np.isinf(angle_traces)] = np.nan

    # create a dataframe with the results
    # kine_data = pd.DataFrame(angle_traces, columns=['mouse_heading', 'cricket_heading', 'delta_heading',
    #                                                 'mouse_cricket_distance', 'mouse_speed', 'mouse_acceleration',
    #                                                 'cricket_speed', 'cricket_acceleration', 'time_vector'])

    kine_data = pd.DataFrame(angle_traces, columns=['mouse_heading', 'mouse_speed', 'mouse_acceleration',
                                                    'time_vector'])

    # if the motive flag is on, also calculate head direction
    if 'mouse_x_m' in data.columns:
        # get the head direction around the vertical axis
        head_direction = data['mouse_zrot_m']
        # get the head height
        head_height = data['mouse_z_m']
        # correct the angle trace for large jumps
        # # define the jump threshold
        # jump_threshold = 10
        # head_direction = wrap(jump_killer(head_direction, jump_threshold))
        # get the offset to correct the head direction due to the center of mass of the tracker
        # (to single degree accuracy)

        # get offset via correlation
        angle_offset = np.argmax([np.corrcoef(np.vstack((head_direction - el, mouse_heading)))[0][1]
                                  for el in range(360)])

        head_direction = head_direction - angle_offset
        # delta_head = cricket_heading - head_direction
        # # correct the delta head to center on 0
        # delta_head[delta_head > 180] += -180
        # delta_head[delta_head < -180] += 180

        # include the motive info in the final dataframe
        kine_data['head_direction'] = head_direction
        kine_data['head_height'] = head_height
        # kine_data['delta_head'] = delta_head

    # if the head data from DLC is available and there's no motive, calculate head direction and delta_look
    if 'mouse_head_x' in data.columns and 'mouse_x_m' not in data.columns:
        # get the head angle with respect to body
        head_direction = heading_calculation(data[['mouse_head_x', 'mouse_head_y']].to_numpy(),
                                             data[['mouse_x', 'mouse_y']].to_numpy())
        # # also get the cricket to head angle
        # head_cricket = heading_calculation(data[['cricket_x', 'cricket_y']].to_numpy(),
        #                                    data[['mouse_head_x', 'mouse_head_y']].to_numpy())
        # # finally, get the delta_head angle
        # delta_head = head_cricket - head_direction
        # # effectively wrap around -180 and 180 to put the center at 0
        # delta_head[delta_head > 180] += -180
        # delta_head[delta_head < -180] += 180
        # save in the dataframe
        kine_data['head_direction'] = head_direction
        # kine_data['delta_head'] = delta_head

    # # if the head direction and delta_head were not added, pad with NaN
    # if 'delta_head' not in kine_data.columns:
    #     kine_data['head_direction'] = np.nan
    #     kine_data['delta_head'] = np.nan

    # kine_data = pd.concat([data.loc[:, ['mouse_x', 'mouse_y', 'cricket_x', 'cricket_y']], kine_data], axis=1)
    kine_data = pd.concat([data.loc[:, ['mouse_x', 'mouse_y']], kine_data], axis=1)

    # check for a real cricket
    if 'cricket_0_x' in data.columns:
        cricket_coord = data[['cricket_0_x', 'cricket_0_y']].to_numpy()
        # process the cricket related data
        cricket_data = cricket_processing(cricket_coord, data, mouse_coord, mouse_heading, kine_data)
        # concatenate the cricket data
        kine_data = pd.concat([kine_data, cricket_data], axis=1)
        # set the cricket number
        real_crickets = 1
    else:
        real_crickets = 0

    # check for vr crickets
    if 'vrcricket_0_x' or 'target_x_m' in data.columns:
        # get the number of vr crickets
        vr_cricket_list = np.unique([el[:11] for el in data.columns if 'vrcricket' in el])
        # If there is no vr_cricket, but this is instead a vr_target
        if vr_cricket_list.size == 0:
            vr_cricket_list = ['target']

        # for all the vr crickets
        for vr_cricket in vr_cricket_list:
            # get the coordinates
            try:
                cricket_coord = data[[vr_cricket+'_x', vr_cricket+'_y']].to_numpy()
            except KeyError:
                cricket_coord = data[[vr_cricket+'_x_m', vr_cricket+'_y_m']].to_numpy()
            # process the cricket related data
            cricket_data = cricket_processing(cricket_coord, data, mouse_coord_hd, mouse_heading,
                                              kine_data, vr=True, cricket_name=vr_cricket)
            # concatenate the cricket data
            kine_data = pd.concat([kine_data, cricket_data], axis=1)
        # set the cricket number
        vr_crickets = len(vr_cricket_list)
    else:
        vr_crickets = 0

    # save the data to file
    kine_data.to_hdf(name, key='full_traces', mode='w', format='table')

    # return the dataframe
    return kine_data, real_crickets, vr_crickets


# if __name__ == '__main__':
#     # prevent the appearance of the tk main window
#     tk_killwindow()
#
#     # define the outcome keyword to search for
#     outcome_keyword = 'all'
#     # define the condition keyword to search for
#     condition_keyword = 'all'
#     # define the mse threshold to save a file
#     mse_threshold = 0.018
#
#     # load the data
#     # base_path = aligned_path
#     base_path = paths.pre_processed_path
#     f_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("aligned files", "*.csv"),))
#     # file_path = test_aligned
#
#     # define the figure save path
#     s_path = paths.kinematics_path
#
#     # parse the file names for the desired trait
#     f_path = file_parser(f_path, outcome_keyword, condition_keyword, mse_threshold=mse_threshold)
#
#     # run the function
#     kinematic_data = kinematic_calculations(f_path, s_path)
