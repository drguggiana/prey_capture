import numpy as np
import csv


def load_preprocessed(file_path_in):
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
    parsed_line = [float(number) if number != '' else np.nan for number in single_line[:-1].split(',')]
    return parsed_line
