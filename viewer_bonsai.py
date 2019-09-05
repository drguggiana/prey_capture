# imports
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import numpy as np
import datetime
# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)


base_path_bonsai = r'C:\Users\drguggiana\Documents\Bonsai_out'
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai)

# define loading path and select file
# allocate a list for all the animals
animal_data_bonsai = []

for animal in file_path_bonsai:
    parsed_data = []
    last_nan = 0
    with open(file_path_bonsai[0]) as f:
        for ex_line in f:
            ex_list = ex_line.split(' ')
            ex_list.remove('\n')
            if ex_list[0] == 'NaN' and last_nan == 0:
                continue
            else:
                last_nan = 1

            timestamp = ex_list.pop()
            ex_list = [float(el) for el in ex_list]
            parsed_data.append([ex_list, timestamp])

    animal_data_bonsai.append(np.array(parsed_data))


target_data = np.array([el[0] for el in animal_data_bonsai[0]])
# 2D mouse movement
fig = plt.figure()
ax = fig.add_subplot(111)
number_points = None
ax.plot(target_data[:number_points, 0], target_data[:number_points, 1])
ax.plot(target_data[:number_points, 2], target_data[:number_points, 3])

plt.gca().invert_xaxis()

# ax.plot(target_data[:number_points,7], target_data[:number_points,9])
ax.autoscale()
ax.axis('equal')

timestamp = [datetime.datetime.strptime(el[1][:-7], '%Y-%m-%dT%H:%M:%S.%f') for el in animal_data_bonsai[0]]
timestamp = [(el - timestamp[0]).total_seconds() for el in timestamp]    # print the frame rate
print('Frame rate:' + str(1 / np.mean(np.diff(timestamp))) + 'fps')

plt.show()
