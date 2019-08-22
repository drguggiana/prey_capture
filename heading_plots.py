# imports
from tkinter import filedialog
from tkinter import Tk
import datetime
from sklearn.metrics import mean_squared_error as mse
from os.path import join, basename
from os import listdir
from scipy.signal import find_peaks
from matching_functions import *
from io_functions import *
from plotting_functions import *
from misc_functions import *


# prevent the appearance of the tk main window
tk_killwindow()

# define the outcome keyword to search for
outcome_keyword = 'all'
# define the condition keyword to search for
condition_keyword = 'all'
# load the data
base_path = r'J:\Drago Guggiana Nilo\Prey_capture\Aligned_traces'
file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("preproc files", "*.csv"),))

# define the figure save path
# figure_save = r'C:\Users\drguggiana\Dropbox\Bonhoeffer_things\Presentations\Figures'
figure_save = r'J:\Drago Guggiana Nilo\Prey_capture\Heading'

# parse the file names for the desired trait
file_parser(file_path, outcome_keyword, condition_keyword)

# actually load the data
data_all = load_preprocessed(file_path)