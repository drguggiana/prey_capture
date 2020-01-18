import numpy as np
import csv
from os.path import basename
import datetime
from skimage import io
from skimage.external import tifffile as tif


def load_preprocessed(file_path_in):
    """Read a csv file"""
    # allocate a list for all the animals
    preproc_data = []

    for animal_in in file_path_in:
        temp_data = []
        with open(animal_in) as f:
            reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for ex_line in reader:
                temp_data.append(np.array(ex_line))

            preproc_data.append(np.array(temp_data))

    return preproc_data


def parse_line(single_line):
    """Parse lines to load only numbers and everything else as NaNs"""
    parsed_line = [float(number) if number != '' else np.nan for number in single_line[:-1].split(',')]
    return parsed_line


def file_parser(file_path, outcome_keyword, condition_keyword, mse_threshold=None):
    """Parse the file path to include or exclude desired terms"""
    # define the possible conditions
    condition_list = ['dark', 'vr', 'miniscope']

    # filter the results by outcome (only use all for performance plot though)
    if outcome_keyword != 'all':
        file_path = [file for file in file_path if outcome_keyword in file]

    # filter the files by the desired condition
    if condition_keyword == '':
        file_path = [file for file in file_path if sum([1 for word in condition_list if word in file]) == 0]
    elif condition_keyword != 'all':
        file_path = [file for file in file_path if condition_keyword in file]
    # if an mse threshold is provided, filter the files further
    # (unless there's no VR, since then there is not alignment mse)
    if (mse_threshold is not None) and ('mse' in file_path[0]):
        file_path = [file for file in file_path if np.float(file[file.find('mse')+3:file.find('mse')+9]) < mse_threshold]
    return file_path


def get_file_date(filename):
    """Extract the file date and time from a bonsai filename"""
    file_date = datetime.datetime.strptime(basename(filename)[:18], '%m_%d_%Y_%H_%M_%S')
    return file_date


def combine_tif(filenames):
    """Function to concatenate together several tif files supplied as a list of paths"""
    # based on https://stackoverflow.com/questions/47182125/how-to-combine-tif-stacks-in-python

    # read the first stack on the list
    im_1 = io.imread(filenames[0])
    # if it's 2d, expand 1 dimension
    if len(im_1.shape) == 2:
        im_1 = np.expand_dims(im_1, 2)
        im_1 = np.transpose(im_1, [2, 0, 1])
    # assemble the output path
    out_path = filenames[0].replace('.tif', '_CAT.tif')
    # run through the remaining files
    for i in range(1, len(filenames)):
        # load the next file
        im_n = io.imread(filenames[i])
        # if it's 2d, expand 1 dimension
        if len(im_n.shape) == 2:
            im_n = np.expand_dims(im_n, 2)
            im_n = np.transpose(im_n, [2, 0, 1])
        # concatenate it to the previous one
        im_1 = np.concatenate((im_1, im_n))
    # save the final stack
    tif.imsave(out_path, im_1.astype('uint16'), bigtiff=True)

    return out_path
