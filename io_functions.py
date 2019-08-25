import numpy as np
import csv


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
    condition_list = ['dark', 'vr']

    # filter the results by outcome (only use all for performance plot though)
    if outcome_keyword != 'all':
        file_path = [file for file in file_path if outcome_keyword in file]

    # filter the files by the desired condition
    if condition_keyword == '':
        file_path = [file for file in file_path if sum([1 for word in condition_list if word in file]) == 0]
    elif condition_keyword != 'all':
        file_path = [file for file in file_path if condition_keyword in file]
    # if an mse threshold is provided, filter the files further
    if mse_threshold is not None:
        file_path = [file for file in file_path if np.float(file[file.find('mse')+3:file.find('mse')+9]) < mse_threshold]
    return file_path
