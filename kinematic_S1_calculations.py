# imports
from tkinter import filedialog
from os.path import join, basename
from functions_io import *
from functions_plotting import *
from functions_misc import *
from functions_kinematic import *
import paths
import pandas as pd


def kinematic_calculations(file_path, data):
    """Calculate basic kinematic parameters of mouse and cricket"""

    kine_data = []
    # calculate heading direction, speed, acceleration
    for file_idx, name in enumerate(file_path):

        # load the dataframe
        # data = pd.read_csv(name, index_col=0)
        # data = pd.read_hdf(name, 'full_traces')

        # # get the field names
        # fields = data.columns

        # calculate the heading with the motive data if available
        if 'motive_x' in data.columns:
            mouse_coord_hd = data.loc[:, ['motive_x', 'motive_y']].to_numpy()
        else:
            mouse_coord_hd = data.loc[:, ['mouse_x', 'mouse_y']].to_numpy()

        mouse_heading = wrap(np.concatenate((heading_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]), [0])))

        # zero the NaNs and unwrap
        mouse_heading[np.isnan(mouse_heading)] = 0

        # calculate the heading to the cricket
        # first get the coordinates of the cricket with respect to the mouse
        cricket_coord = data.loc[:, ['cricket_x', 'cricket_y']].to_numpy()
        # get the mouse coordinates too (from bonsai)
        mouse_coord = data.loc[:, ['mouse_x', 'mouse_y']].to_numpy()
        # get the time
        time_vector = data.time.to_numpy()
        # get the cricket angle to the mouse
        cricket_heading = heading_calculation(cricket_coord, mouse_coord)
        cricket_heading[np.isnan(cricket_heading)] = 0

        # get the delta angle
        delta_heading = cricket_heading - mouse_heading

        # get the mouse-cricket distance
        mouse_cricket_distance = distance_calculation(cricket_coord, mouse_coord)

        mouse_speed = np.concatenate(
            ([0], distance_calculation(mouse_coord_hd[1:, :], mouse_coord_hd[:-1, :]) /
             (time_vector[1:] - time_vector[:-1])))
        mouse_acceleration = np.concatenate(([0], np.diff(mouse_speed)))

        cricket_speed = np.concatenate(([0], distance_calculation(cricket_coord[1:, :], cricket_coord[:-1, :])
                                        / (time_vector[1:] - time_vector[:-1])))
        cricket_acceleration = np.concatenate(([0], np.diff(cricket_speed)))

        # TODO: clean up preprocessing so this is not needed
        # mouse_cricket_distance[mouse_cricket_distance > 2] = np.nan
        # mouse_speed[mouse_speed > 20] = np.nan
        # cricket_speed[cricket_speed > 200] = np.nan

        # save the traces to a variable
        angle_traces = np.vstack((mouse_heading, cricket_heading, delta_heading, mouse_cricket_distance,
                                  mouse_speed, mouse_acceleration, cricket_speed, cricket_acceleration, time_vector)).T
        # replace infinity values with NaNs (in the kinematic traces)
        angle_traces[np.isinf(angle_traces)] = np.nan

        # create a dataframe with the results
        kine_data = pd.DataFrame(angle_traces, columns=['mouse_heading', 'cricket_heading', 'delta_heading',
                                                             'mouse_cricket_distance', 'mouse_speed', 'mouse_acceleration',
                                                             'cricket_speed', 'cricket_acceleration', 'time_vector'])

        # if the motive flag is on, also calculate head direction
        if 'motive_x' in data.columns:
            # get the head direction around the vertical axis
            head_direction = data.motive_ry
            # get the head height
            head_height = data.motive_z
            # correct the angle trace for large jumps
            # define the jump threshold
            jump_threshold = 10
            head_direction = wrap(jump_killer(head_direction, jump_threshold))
            # get the offset to correct the head direction due to the center of mass of the tracker
            # (to single degree accuracy)

            # get offset via correlation
            angle_offset = np.argmax([np.corrcoef(np.vstack((wrap(head_direction - el), mouse_heading)))[0][1]
                                      for el in range(360)])
            # print(angle_offset)

            head_direction = head_direction - angle_offset
            # print(np.corrcoef(np.vstack((head_direction, mouse_heading)))[0][1])
            # # calculate the arrow centers
            # arrow_centers = np.vstack((data[0, 0:2], (data[1:, 0:2] + data[:-1, 0:2]) / 2))
            # quiver_fig = plot_arrow(data[:, :2], arrow_centers, np.ones_like(arrow_centers),
            #                         np.ones_like(arrow_centers) * 0.5,
            #                         data[:, 8:10], angles=mouse_heading, angles2=head_direction)
            # animate_hunt(data[:, :2], mouse_heading, head_direction, data[:, 8:10], (-0.7, 0.6), (-0.35, 0.35),
            #              interval=80)
            delta_head = cricket_heading - head_direction

            # # save the quiver plot
            # quiver_fig.savefig(join(save_path, basename(file_path[file_idx])[:-12] + '.png'), bbox_inches='tight')
            # plt.close(fig='all')

            # include the motive info in the final dataframe
            kine_data['head_direction'] = head_direction
            kine_data['head_height'] = head_height
            kine_data['delta_head'] = delta_head

        # save the data to file
        # assemble the file name
        # save_file = join(save_path, basename(file_path[file_idx])[:-12] + '_kinematics.csv')
        kine_data.to_hdf(name, key='full_traces', mode='w', format='table')

        # save the dataframe
        # kine_data.to_csv(save_file)
    return kine_data
    # plt.show()


if __name__ == '__main__':
    # prevent the appearance of the tk main window
    tk_killwindow()

    # define the outcome keyword to search for
    outcome_keyword = 'all'
    # define the condition keyword to search for
    condition_keyword = 'all'
    # define the mse threshold to save a file
    mse_threshold = 0.018

    # load the data
    # base_path = aligned_path
    base_path = paths.pre_processed_path
    f_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("aligned files", "*.csv"),))
    # file_path = test_aligned

    # define the figure save path
    s_path = paths.kinematics_path

    # parse the file names for the desired trait
    f_path = file_parser(f_path, outcome_keyword, condition_keyword, mse_threshold=mse_threshold)

    # run the function
    kinematic_data = kinematic_calculations(f_path, s_path)
