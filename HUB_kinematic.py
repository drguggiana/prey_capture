from tkinter import filedialog
from functions_misc import tk_killwindow
import paths

# get rid of the tk main window
tk_killwindow()

# load the data
base_path = paths.aligned_path
file_path = filedialog.askopenfilenames(initialdir=base_path, filetypes=(("aligned files", "*.csv"),))

# # run through the files
# for files in file_path:
