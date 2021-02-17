# imports
import matlab.engine
import sys

# get the path from the shell arguments
target_path = sys.argv[1]
# start the matlab engine
eng = matlab.engine.start_matlab()
# add paths to the matlab path with the required functions
# TODO: add these from a separate file
eng.addpath('D:\Code Repos\miniscope_processing')
eng.addpath('R:\Share\Simon\Drago_Volker_Simon\invivo_depth_GUI')
# run min1PIPE
path_out = eng.min1pipe_HPC_Python(target_path, nargout=1)
# show the generated path
print(path_out)

