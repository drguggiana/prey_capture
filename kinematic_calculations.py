# imports
from tkinter import filedialog
from os.path import join, basename
from io_functions import *
from plotting_functions import *
from misc_functions import *
from kinematics_functions import *
from paths import *

# prevent the appearance of the tk main window
tk_killwindow()

# define the outcome keyword to search for
outcome_keyword = 'all'
# define the condition keyword to search for
condition_keyword = 'all'
# define the mse threshold to save a file
mse_threshold = 0.018

# load the data
base_path = aligned_path
file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("aligned files", "*.csv"),))
# file_path = test_aligned

# define the figure save path
save_path = heading_path

# parse the file names for the desired trait
file_path = file_parser(file_path, outcome_keyword, condition_keyword, mse_threshold=mse_threshold)

# actually load the data
data_all = load_preprocessed(file_path)

# calculate heading direction, speed, acceleration
for file_idx, data in enumerate(data_all):
    # get rid of the NaNs
    # TODO: clean up preprocessing so skipping frames at this level is not necessary
    data = data[50:, :]
    # get the heading angle
    mouse_heading = wrap(np.concatenate(([0], heading_calculation(data[1:, :2], data[:-1, :2]))))
    # zero the NaNs and unwrap
    mouse_heading[np.isnan(mouse_heading)] = 0

    # get the head direction around the vertical axis
    head_direction = data[:, 5]

    # get the offset to correct the head direction due to the center of mass of the tracker (to single degree accuracy)
    angle_histogram = np.histogram(wrap(head_direction - mouse_heading), bins=72)
    angle_offset = np.round(angle_histogram[1][np.argmax(angle_histogram[0])])

    print(angle_offset)

    head_direction = head_direction - angle_offset

    # # prepare the polar heading plot
    # binned_angles = bin_angles(wrap(head_direction - mouse_heading))
    # plot_polar(binned_angles)

    # # calculate the arrow centers
    arrow_centers = np.vstack((data[0, 0:2], (data[1:, 0:2] + data[:-1, 0:2]) / 2))
    quiver_fig = plot_arrow(data[:, :2], arrow_centers, np.ones_like(arrow_centers), np.ones_like(arrow_centers) * 0.5, data[:,
               8:10], angles=mouse_heading, angles2=head_direction)
    # animate_hunt(data[:, :2], mouse_heading, head_direction, data[:, 8:10], (-0.7, 0.6), (-0.35, 0.35),
    #              interval=80)

    # plot_2d([[mouse_heading, head_direction]])

    # calculate the heading to the cricket
    # first get the coordinates of the cricket with respect to the mouse
    cricket_coord = data[:, 8:10]
    # get the cricket angle to the mouse
    cricket_heading = heading_calculation(cricket_coord, data[:, 6:8])
    cricket_heading[np.isnan(cricket_heading)] = 0

    # get the delta angle
    delta_angle = cricket_heading - mouse_heading
    # binned_angles = bin_angles(wrap(delta_angle))
    # plot_polar(binned_angles)

    # get the mouse-cricket distance
    mouse_cricket_distance = distance_calculation(cricket_coord, data[:, 6:8])
    mouse_speed = np.concatenate(([0], distance_calculation(data[1:, :2], data[:-1, :2])/(data[1:, -1] - data[0, -1])))
    mouse_acceleration = np.concatenate(([0], np.diff(mouse_speed)))

    # save the traces to a variable
    angle_traces = np.concatenate((mouse_heading, head_direction, cricket_heading, delta_angle))
    kinematic_traces = np.concatenate((mouse_cricket_distance, mouse_speed, mouse_acceleration))

    # save the data to file
    # assemble the file name
    save_file = join(save_path, basename(file_path[file_idx])[:-12] + '_kinematics.csv')
    with open(save_file, mode='w', newline='') as f:
        file_writer = csv.writer(f, delimiter=',')
        for a, k in zip(angle_traces, kinematic_traces):
            file_writer.writerow(np.hstack((a, k)))

    # save the quiver plot
    quiver_fig.savefig(join(save_path, basename(file_path[file_idx])[:-12] + '.png'), bbox_inches='tight')
    plt.close(fig='all')
