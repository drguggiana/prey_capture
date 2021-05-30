import numpy as np
import csv
import os
from os import path
import shutil
import datetime
from skimage import io
# from skimage.io import tifffile as tif
import pandas as pd
import psutil


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
    file_date = datetime.datetime.strptime(os.path.basename(filename)[:18], '%m_%d_%Y_%H_%M_%S')
    return file_date


def combine_tif(filenames, processing_path=None):
    """Function to concatenate together several tif files supplied as a list of paths"""
    # based on https://stackoverflow.com/questions/47182125/how-to-combine-tif-stacks-in-python

    # read the first stack on the list
    im_1 = io.imread(filenames[0])
    # allocate a list to store the original names and the number of frames
    frames_list = []
    # if it's 2d (i.e 1 frame), expand 1 dimension
    if len(im_1.shape) == 2:
        im_1 = np.expand_dims(im_1, 2)
        im_1 = np.transpose(im_1, [2, 0, 1])
    print(np.max(im_1))
    # save the file name and the number of frames
    frames_list.append([filenames[0], im_1.shape[0]])
    # assemble the output path
    if processing_path is not None:
        # get the basename
        base_name = os.path.basename(filenames[0])
        out_path_tif = os.path.join(processing_path, base_name.replace('.tif', '_CAT.tif'))
        out_path_log = os.path.join(processing_path, base_name.replace('.tif', '_CAT.csv'))
    else:
        out_path_tif = filenames[0].replace('.tif', '_CAT.tif')
        out_path_log = filenames[0].replace('.tif', '_CAT.csv')
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
        # save the file name and the number of frames
        frames_list.append([filenames[i], im_n.shape[0]])
    # scale the output to max and turn into uint8 (for MiniAn)
    max_value = np.max(im_1)
    for idx, frames in enumerate(im_1):
        im_1[idx, :, :] = ((frames/max_value)*255).astype('uint8')
    # save the final stack
    io.imsave(out_path_tif, im_1, bigtiff=True)
    # save the info about the files
    frames_list = pd.DataFrame(frames_list, columns=['filename', 'frame_number'])
    frames_list.to_csv(out_path_log)

    return out_path_tif, out_path_log, frames_list


def delete_contents(folder_path):
    """Delete all files and folders inside the target folder"""
    "taken from https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def parse_path(in_path):
    """Parse the input path into a dict"""
    path_parts = path.basename(in_path)[:-4].split('_')

    # check whether the rig is miniscope or social
    if path_parts[6] == 'miniscope':
        rig = 'miniscope'
        counter = 7
    elif path_parts[6] == 'social':
        rig = 'social'
        counter = 7
    elif path_parts[6] == 'VPrey':
        rig = 'VPrey'
        counter = 7
    elif path_parts[6] == 'VScreen':
        rig = 'VScreen'
        counter = 7
    else:
        rig = 'VR'
        counter = 6

    out_path = {'datetime': datetime.datetime.strptime('_'.join((path_parts[:6])), '%m_%d_%Y_%H_%M_%S'),
                'rig': rig,
                'animal': '_'.join((path_parts[counter:counter+3])),
                'result': path_parts[counter+3]}
    return out_path


def has_handle(fpath):
    """Check whether there's a process with a handle on this file"""
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False
