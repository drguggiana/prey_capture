# imports
from tkinter import filedialog
from tkinter import Tk
import matplotlib.pyplot as plt
import numpy as np
# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

# define loading path and select file
base_path = r'C:\Users\drguggiana\Documents\Motive_test1\etc'
file_path = filedialog.askopenfilenames(initialdir=base_path)


# parse the file
def parse_line(single_line):
    parsed_line = [float(number) for number in single_line[:-1].split(',')]
    # parsed_line = re.findall('[+-]?\d+[.]\d+',single_line)
    # parsed_line = [float(s) for s in re.findall('[+-]?\d+.\d+',single_line)]
    return parsed_line


# allocate a list for all the animals
animal_data = []

for animal in file_path:
    parsed_data = []
    with open(file_path[0]) as f:
        for line in f:
            if line[0] == '0':
                continue
            parsed_data.append(parse_line(line))

    animal_data.append(np.array(parsed_data))

    # define target file
    target_file = 0

    # get the corresponding data
    target_data = animal_data[target_file]

    print(len(target_data))
    timestamp = target_data[:, 0]
    # print('Frame rate motive:' + str(1 / np.mean(np.diff(timestamp * 1e-7))) + ' fps')
    print('Frame rate motive:' + str(1 / np.mean(np.diff(timestamp))) + ' fps')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.diff(timestamp))

    # 2D mouse movement
    fig = plt.figure()
    ax = fig.add_subplot(111)
    number_points = None
    ax.plot(target_data[:number_points, 3], target_data[:number_points, 1])
    plt.gca().invert_xaxis()

    # ax.plot(target_data[:number_points,7], target_data[:number_points,9])
    ax.autoscale()
    ax.axis('equal')

plt.show()