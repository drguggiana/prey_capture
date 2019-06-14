# from tkinter import filedialog
# from tkinter import Tk
# # Create Tk root
# root = Tk()
# # Hide the main window
# root.withdraw()
# root.call('wm', 'attributes', '.', '-topmost', True)
import numpy as np
import csv
from os import listdir
from os.path import isfile, join, basename
import matplotlib.pyplot as plt

# define the outcome keyword to search for
outcome_keyword = 'all'
# define the condition keyword to search for
condition_keyword = 'dark'
condition_list = ['dark', 'vr']
# load the data
base_path = r'J:\Drago Guggiana Nilo\Prey_capture\Pre_processed'
file_path = [join(base_path, f) for f in listdir(base_path) if isfile(join(base_path, f[:-4]+'.csv'))]

# define the figure save path
figure_save = r'C:\Users\drguggiana\Dropbox\Bonhoeffer_things\Presentations\Figures'

# filter the results by outcome
if outcome_keyword != 'all':
    file_path = [file for file in file_path if outcome_keyword in file]
# filter the results by condition

if condition_keyword == '':
    file_path = [file for file in file_path if sum([1 for word in condition_list if word in file]) == 0]
elif condition_keyword != 'all':
    file_path = [file for file in file_path if condition_keyword in file]

# file_path = filedialog.askopenfilenames(initialdir=base_path)
# define loading path and select file
# allocate a list for all the animals
animal_data = []

for animal in file_path:
    parsed_data = []
    with open(animal) as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for ex_line in reader:
            parsed_data.append(np.array(ex_line))

        animal_data.append(np.array(parsed_data))

# calculate the speed and acceleration for each animal
# allocate memory
kinematic_list = []
# define the length constant (measured approximately from the data)
m_px = 1 / 445
# for all the animals
for animal in animal_data:

    webcam_perFrame = np.diff(animal[:, 4])
    webcam_time = animal[:, 4]

    cricket_speed = np.linalg.norm(np.diff(animal[:, [2, 3]], axis=0), axis=1) * m_px / webcam_perFrame
    cricket_acceleration = np.diff(cricket_speed)

    mouse_speed = np.linalg.norm(np.diff(animal[:, [0, 1]], axis=0), axis=1) * m_px / webcam_perFrame
    mouse_acceleration = np.diff(mouse_speed)

    kinematic_list.append(np.vstack((mouse_speed, np.hstack((0, mouse_acceleration)),
                                     cricket_speed, np.hstack((0, cricket_acceleration)), webcam_time[1:])).T)

# calculate mouse to cricket distance

# allocate memory
distance_list = []
# for all the animals
for animal in animal_data:
    distance_list.append(np.array([np.linalg.norm(el[[0, 1]] - el[[2, 3]]) for el in animal]))

# plot distance histograms for the whole data set
# concatenate the entire distance array
all_distances = np.hstack(distance_list)*m_px

# produce the histogram
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(all_distances)
plt.xlabel('Mouse to cricket distance [m]')
fig.savefig(join(figure_save, 'distanceHistogram_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')

# plot all the distance traces
fig = plt.figure()
ax = fig.add_subplot(111)
# for all the animals
for idx, animal in enumerate(distance_list):
    ax.plot(kinematic_list[idx][:, -1], animal[1:]*m_px)
plt.title('Mouse to cricket distance [m]')
fig.savefig(join(figure_save, 'distanceTraces_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')


# plot all the speed traces for the mouse
fig = plt.figure()
ax = fig.add_subplot(111)
# for all the animals
for animal in kinematic_list:
    ax.plot(animal[:, -1], animal[:, 0])
ax.legend(range(len(animal_data)))
plt.title('Mouse speed [m/s]')
fig.savefig(join(figure_save, 'mouseSpeedTraces_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')


# plot all the speed traces for the cricket
fig = plt.figure()
ax = fig.add_subplot(111)
# for all the animals
for animal in kinematic_list:
    ax.plot(animal[:, -1], animal[:, 2])
plt.title('Cricket speed [m/s]')
fig.savefig(join(figure_save, 'cricketSpeedTraces_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')


# plot the trajectories of a given experiment
target_experiment = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(animal_data[target_experiment][:, 0], animal_data[target_experiment][:, 1])
ax.plot(animal_data[target_experiment][:, 2], animal_data[target_experiment][:, 3])
plt.gca().invert_yaxis()

# plot a histogram of the mouse speeds
all_kinematics = np.vstack(kinematic_list)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(all_kinematics[:, 0], log=True)
plt.xlabel('Mouse speed [m/s]')
ax = fig.add_subplot(122)
ax.hist(all_kinematics[:, 1], log=True)
plt.xlabel('Mouse acceleration [m/s^2]')
fig.savefig(join(figure_save, 'mouseKinematicHistograms_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(121)
ax.hist(all_kinematics[:, 2], log=True)
plt.xlabel('Cricket speed [m/s]')
ax = fig.add_subplot(122)
ax.hist(all_kinematics[:, 3], log=True)
plt.xlabel('Cricket acceleration [m/s^2]')

# plt.subplots_adjust(hspace=0.4)
fig.savefig(join(figure_save, 'cricketKinematicHistograms_'+outcome_keyword+'_'+condition_keyword+'.png'), bbox_inches='tight')


# TODO: plot performance
# TODO: plot time to capture
# TODO: plot latency to initiation
# TODO: plot distance triggered average
# TODO: remove points before both animals are tracked validly (within bounds)
# TODO: incorporate motive traces for the VR cricket

plt.show()
print('yay')
