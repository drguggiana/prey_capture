# imports
from functions_kinematic import *
import functions_preprocessing as fp
import pandas as pd
import scipy as sc
import processing_parameters
from scipy.ndimage.measurements import label


def cricket_processing(cricket_coord, data, mouse_coord, mouse_heading, kine_data, vr=False, cricket_name='cricket_0'):
    """Process the cricket related variables"""

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

        # define the factor for cricket size (see onenote)
        cricket_proportion = 1.73
        cricket_to_head = 0.19
        cricket_to_tail = 0.57
        # define the width to length relationship to find the bounding box coordinates
        cricket_aspect_ratio = 0.288
        # calculate cricket size
        c_size = fp.cricket_size(data, cricket_proportion)
        # save in the dataframe
        cricket_data[cricket_name+'_size'] = c_size

        # get the mouse coordinates
        mouse_coord_numpy = data[['mouse_head_x', 'mouse_head_y']].to_numpy()

        # also get the cricket to head angle
        head_cricket = heading_calculation(data[['cricket_0_x', 'cricket_0_y']].to_numpy(),
                                           mouse_coord_numpy)
        # finally, get the delta_head angle
        delta_head = head_cricket - kine_data.head_direction
        # effectively wrap around -180 and 180 to put the center at 0
        delta_head[delta_head > 180] += -180
        delta_head[delta_head < -180] += 180
        # change nans to 0 as these happen when cricket and mouse overlap
        delta_head[np.isnan(delta_head)] = 0
        # save in the dataframe
        cricket_data[cricket_name+'_delta_head'] = delta_head

        # generate a vector with the FOV quadrant the cricket is in
        # 0 is binocular, 1 is left eye, 2 is right eye, 3 is out of view, assume 280 deg total FOV per eye
        # and 40 deg overlap (from Meyer et al. 2018)
        fov_quadrant = np.zeros_like(delta_head)
        # create a vector with the delta head data
        delta_head_array = delta_head.to_numpy()
        fov_quadrant[(-140 < delta_head_array) & (delta_head_array < -20)] = 1
        fov_quadrant[(140 > delta_head_array) & (delta_head_array > 20)] = 2
        fov_quadrant[np.abs(delta_head_array) >= 140] = 3
        # save the vector in the dataframe
        cricket_data[cricket_name+'_quadrant'] = fov_quadrant

        # calculate the visual angle subtended by the cricket
        # get the unit vector for the cricket body
        delta_vector = data[['cricket_0_head_x', 'cricket_0_head_y']].to_numpy() \
            - data[['cricket_0_x', 'cricket_0_y']].to_numpy()
        # get the length of the head point to body point vector
        distances = np.linalg.norm(delta_vector, axis=1)
        unit_vector = delta_vector/np.expand_dims(np.linalg.norm(delta_vector, axis=1), axis=1)
        # get the x and y coordinates of the edges of the cricket
        head_point = data[['cricket_0_head_x', 'cricket_0_head_y']].to_numpy() + \
            np.expand_dims(distances, axis=1)*cricket_to_head*unit_vector
        tail_point = data[['cricket_0_x', 'cricket_0_y']].to_numpy() - \
            np.expand_dims(distances, axis=1)*cricket_to_tail*unit_vector
        # get the perpendicular to the unit vector
        unit_ortho = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])@unit_vector.T
        unit_ortho = unit_ortho.T
        # now get the corners of the bounding box
        head_corner_1 = head_point + \
            np.expand_dims(distances, axis=1)*cricket_proportion*cricket_aspect_ratio*unit_ortho
        head_corner_2 = head_point - \
            np.expand_dims(distances, axis=1)*cricket_proportion*cricket_aspect_ratio*unit_ortho
        tail_corner_1 = tail_point + \
            np.expand_dims(distances, axis=1)*cricket_proportion*cricket_aspect_ratio*unit_ortho
        tail_corner_2 = tail_point - \
            np.expand_dims(distances, axis=1)*cricket_proportion*cricket_aspect_ratio*unit_ortho
        # put them on a list for for loops
        corner_list = [head_corner_1, head_corner_2, tail_corner_1, tail_corner_2]
        # get the angles
        angle_list = [heading_calculation(el, mouse_coord_numpy) + 360
                      for el in corner_list]
        # turn into an array
        angle_list = np.array(angle_list).T
        # get the angles
        visual_angle = np.max(np.array([sc.spatial.distance.pdist(np.expand_dims(el, axis=1)) for el in angle_list]),
                              axis=1)
        # filter the list by the quadrant (if not visible, discard)
        visual_angle[fov_quadrant == 3] = 0

        # visual_angle = np.apply_along_axis(sc.spatial.distance.pdist, 1, np.expand_dims(angle_list.T, axis=2))
        # # get the angles of these coordinates, including an offset to wrap
        # head_angle = heading_calculation(head_vector,
        #                                  data[['mouse_head_x', 'mouse_head_y']].to_numpy()) + 360
        # tail_angle = heading_calculation(tail_vector,
        #                                  data[['mouse_head_x', 'mouse_head_y']].to_numpy()) + 360
        # # get the delta angle, i.e. the visual angle
        # visual_angle = head_angle - tail_angle
        # save the coordinates in the dataframe
        cricket_data[cricket_name+'_visual_angle'] = visual_angle

        # get the delta time
        delta_time = np.concatenate(([1], np.diff(kine_data.time_vector.to_numpy())), axis=0)
        # calculate the derivatives of the distance, heading and visual angle
        cricket_direction = np.concatenate(([0], np.diff(delta_head+180)), axis=0)
        # wrap cricket direction
        cricket_direction[cricket_direction < -180] += 180
        cricket_direction[cricket_direction > 180] -= 180
        cricket_data[cricket_name + '_direction'] = cricket_direction/delta_time
        cricket_data[cricket_name + '_loom'] = np.concatenate(([0], np.diff(mouse_cricket_distance)), axis=0) / \
            delta_time
        cricket_data[cricket_name + '_delta_visual'] = np.concatenate(([0], np.diff(visual_angle)), axis=0) / \
            delta_time
    return cricket_data


def label_hunt(data, cricket):
    """Label the whole trial trace based on state of the hunt"""

    # get the mouse speed
    mouse_speed = data.loc[:, 'mouse_speed'].to_numpy()
    # get the approaches (according to Hoy et al.)
    speed_criterion = mouse_speed > processing_parameters.speed_threshold
    angle_criterion = (cricket.loc[:, 'cricket_0_delta_heading'] < processing_parameters.angle_threshold / 2) & \
                      (cricket.loc[:, 'cricket_0_delta_heading'] > - processing_parameters.angle_threshold / 2)
    range_rate = np.diff(cricket.loc[:, 'cricket_0_mouse_distance']) / np.diff(data.loc[:, 'time_vector'])
    range_criterion = np.hstack((np.zeros((1,)), range_rate)) < processing_parameters.range_threshold
    approach_vector = speed_criterion & angle_criterion & range_criterion
    # eliminate gaps that are too small
    [gap_vector, gap_num] = label(~approach_vector)
    # for all the gaps
    for gap in np.arange(1, gap_num):
        # check the gap length, if too short, eliminate it
        if data.loc[gap_vector == gap, 'time_vector'].sum() > processing_parameters.approach_break_threshold:
            approach_vector[gap_vector == gap] = 1

    # get the contact vector
    encounter_vector = (cricket.loc[:, 'cricket_0_mouse_distance'] < processing_parameters.encounter_threshold)
    # eliminate gaps that are too small (based on same threshold as approaches)
    [gap_vector, gap_num] = label(~encounter_vector)
    # for all the gaps
    for gap in np.arange(1, gap_num):
        # check the gap length, if too short, eliminate it
        if data.loc[gap_vector == gap, 'time_vector'].sum() > processing_parameters.approach_break_threshold:
            encounter_vector[gap_vector == gap] = 1

    # allocate the labeled vector
    labeled_hunt = np.zeros_like(mouse_speed)
    # fill it up
    labeled_hunt[approach_vector == 1] = 1
    labeled_hunt[encounter_vector == 1] = 2
    # turn into series
    labeled_hunt = pd.DataFrame(labeled_hunt, columns=['hunt_trace'])
    return labeled_hunt


def kinematic_calculations(data, kernel_size=11):
    """Calculate basic kinematic parameters of mouse and cricket"""

    # check if there are nans. if so, return a badFile dataframe
    coordinate_columns = [el for el in data.columns if ('_x' in el) | ('_y' in el)]

    # TODO: remove this gate once DLC networks are in place
    if 'trial_num' not in data.columns:
        if np.any(np.isnan(data[coordinate_columns])):
            real_crickets = 0
            vr_crickets = 0
            kine_data = pd.DataFrame(['badFile'], columns=['badFile'])

            return kine_data, real_crickets, vr_crickets

    # keep the mouse, datetime and syncframes columns (and trial_num if there)
    meta_columns = [el for el in ['mouse', 'datetime', 'sync_frames', 'trial_num', 'grating_phase']
                    if el in data.columns]

    # define which coordinates to use depending on the available data
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

    # get the time
    time_vector = data.time_vector.to_numpy()

    mouse_speed = np.concatenate(
        ([0], distance_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]) /
         (time_vector[1:] - time_vector[:-1])))
    mouse_acceleration = np.concatenate(([0], np.diff(mouse_speed)))

    # calculate the angular speed
    head_angular_speed = np.concatenate(([0], np.diff(mouse_heading + 180)), axis=0)
    delta_time = np.concatenate(([1], np.diff(time_vector)), axis=0)
    # wrap the angular speed
    head_angular_speed[head_angular_speed > 180] -= 360
    head_angular_speed[head_angular_speed < -180] += 360
    angular_speed = head_angular_speed/delta_time

    # save the traces to a variable
    angle_traces = np.vstack((mouse_heading, angular_speed, mouse_speed, mouse_acceleration, time_vector)).T
    # replace infinity values with NaNs (in the kinematic traces)
    angle_traces[np.isinf(angle_traces)] = np.nan

    # create a dataframe with the results
    kine_data = pd.DataFrame(angle_traces, columns=['mouse_heading', 'mouse_angular_speed', 'mouse_speed',
                                                    'mouse_acceleration', 'time_vector'])
    # add the meta columns
    kine_data = pd.concat([kine_data, data.loc[:, meta_columns]], axis=1)

    # if the motive flag is on, also calculate head direction
    if 'mouse_x_m' in data.columns:
        # Tidy up pitch, yaw, roll
        pitch = -wrap_negative(data.mouse_xrot_m.values)
        kine_data['pitch'] = smooth_trace(pitch, range=(-180, 180), kernel_size=10, discont=2*np.pi)

        yaw = wrap_negative(data.mouse_zrot_m.values)
        kine_data['yaw'] = smooth_trace(yaw, range=(-180, 180), kernel_size=10, discont=2*np.pi)

        roll = wrap_negative(data.mouse_yrot_m.values)
        kine_data['roll'] = smooth_trace(roll, range=(-180, 180), kernel_size=10, discont=2*np.pi)

        # get the head direction around the vertical axis
        head_direction = data['mouse_zrot_m']
        # get the head height
        head_height = data['mouse_z_m']

        # get the offset to correct the head direction due to the center of mass of the tracker
        # (to single degree accuracy)

        # get offset via correlation
        angle_offset = np.argmax([np.corrcoef(np.vstack((head_direction - el, mouse_heading)))[0][1]
                                  for el in range(360)])

        head_direction = head_direction - angle_offset

        # include the motive info in the final dataframe
        kine_data['head_direction'] = head_direction
        kine_data['head_height'] = head_height

    # if the head data from DLC is available and there's no motive, calculate head direction and delta_look
    if 'mouse_head_x' in data.columns and 'mouse_x_m' not in data.columns:
        # get the head angle with respect to body
        head_direction = heading_calculation(data[['mouse_head_x', 'mouse_head_y']].to_numpy(),
                                             data[['mouse_x', 'mouse_y']].to_numpy())

        # # calculate the angular speed
        # head_angular_speed = np.concatenate(([0], np.diff(head_direction+180)), axis=0)
        # delta_time = np.concatenate(([1], np.diff(kine_data.loc[:, 'time_vector'].to_numpy())), axis=0)
        # # wrap the angular speed
        # head_angular_speed[head_angular_speed > 180] -= 360
        # head_angular_speed[head_angular_speed < -180] += 360
        # save in the dataframe
        kine_data['head_direction'] = head_direction
        # kine_data['angular_speed'] = head_angular_speed/delta_time

    # add the raw tracks from DLC (already filtered in S1)
    track_list = [el for el in list(data.columns) if 'mouse_' in el]
    kine_data = pd.concat([data.loc[:, track_list], kine_data], axis=1)

    # check for a real cricket
    if 'cricket_0_x' in data.columns:
        cricket_coord = data[['cricket_0_x', 'cricket_0_y']].to_numpy()
        cricket_head = data[['cricket_0_head_x', 'cricket_0_head_y']]
        # process the cricket related data
        cricket_data = cricket_processing(cricket_coord, data, mouse_coord, mouse_heading, kine_data)
        # get the hunt trace
        hunt_trace = label_hunt(kine_data, cricket_data)
        # concatenate the cricket data
        kine_data = pd.concat([kine_data, cricket_data, cricket_head, hunt_trace], axis=1)
        # set the cricket number
        real_crickets = 1
    else:
        real_crickets = 0

    # check for vr crickets or vr targets
    if ('vrcricket_0_x' in data.columns) or ('target_x_m' in data.columns):
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
            # also grab the vr cricket states
            try:
                cricket_states = data[[vr_cricket+'_state', vr_cricket+'_motion', vr_cricket+'_encounter']]
            except KeyError:
                # There are no cricket animation in the VScreens trial, but we do
                # need the trial number data for later
                cricket_states = data[['trial_num']]
            # concatenate the cricket data
            kine_data = pd.concat([kine_data, cricket_data, cricket_states], axis=1)
        # set the cricket number
        vr_crickets = len(vr_cricket_list)
    else:
        vr_crickets = 0

    # define the columns the median
    target_columns = [el for el in kine_data.columns if el not in
                      ['time_vector', 'mouse', 'datetime', 'sync_frames', 'cricket_0_size', 'hunt_trace'] + track_list]
    # median filter the traces
    kine_data = fp.median_discontinuities(kine_data, target_columns, kernel_size)

    # return the dataframe
    return kine_data, real_crickets, vr_crickets
