import subprocess
import paths
import functions_io
from functions_misc import tk_killwindow
from tkinter import filedialog


# get rid of the tk main window
tk_killwindow()

# prompt the user for file selection
# define the base loading path
base_path_bonsai = paths.testmini_path
# select the files to process
file_path_bonsai = filedialog.askopenfilenames(initialdir=base_path_bonsai, filetypes=(("tif files", "*.tif"), ))

# combine the selected files into a single tif
out_path = functions_io.combine_tif(file_path_bonsai)


# min1pipe_process = subprocess.Popen([paths.minpipe_path])