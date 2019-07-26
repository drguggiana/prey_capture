# imports
from tkinter import filedialog
from tkinter import Tk
# Create Tk root
root = Tk()
# Hide the main window
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)
import matplotlib.pyplot as plt
import numpy as np

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