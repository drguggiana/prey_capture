import os.path
from glob import glob, iglob
from os.path import join, sep
import numpy as np
import h5py


def find_matched_files(target_directory, animals, regex_pattern, exp_to_match):
    all_files = glob(join(target_directory, regex_pattern))
    keep_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]
    keep_file_basenames = [os.path.basename(file).split('.')[0] for file in keep_files]
    matched_experiments = {}

    for animal in animals:
        animal_filenames = [file for file in keep_file_basenames if animal in file]
        animal_filenames.sort()
        dates = [filename[:10] for filename in animal_filenames]
        values, counts = np.unique(dates, return_counts=True)
        matched_dates = values[counts == 2]

        control_files = [file for file in animal_filenames if 'control' in file]
        control_dates = [filename[:10] for filename in control_files]
        control_values, control_counts = np.unique(control_dates, return_counts=True)
        control_dates = control_values[control_counts == 2]

        matched_experiments[animal] = len(matched_dates) - len(control_dates)
        matched_experiments[f"{animal}_controls"] = len(control_dates)

    print('The following animals have matched experimental data for these many days')
    for key, value in matched_experiments.items():
        print(f"    {key}: {value}")


def find_unmatched_files(target_directory, animals, regex_pattern, exp_to_match):
    all_files = glob(join(target_directory, regex_pattern))
    keep_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]
    keep_file_basenames = [os.path.basename(file).split('.')[0] for file in keep_files]
    unmatched_experiments = []

    for animal in animals:
        animal_filenames = [file for file in keep_file_basenames if animal in file]
        animal_filenames.sort()
        dates = [filename[:10] for filename in animal_filenames]
        values, counts = np.unique(dates, return_counts=True)
        unmatched_dates = values[counts == 1]

        for date in unmatched_dates:
            unmatched = [file for file in animal_filenames if date in file]
            unmatched_experiments.extend(unmatched)

    print(f'The following {len(unmatched_experiments)} files are unmatched:')
    if len(unmatched_experiments) > 0:
        for exp in unmatched_experiments:
            print('    ' + exp)
    else:
        print('    No unmatched experiments!\n')


def find_extra_session_files(target_directory, animals, regex_pattern, exp_to_match):
    all_files = glob(join(target_directory, regex_pattern))
    keep_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]
    keep_file_basenames = [os.path.basename(file).split('.')[0] for file in keep_files]
    extra_experiments = []

    for animal in animals:
        animal_filenames = [file for file in keep_file_basenames if animal in file]
        animal_filenames.sort()
        dates = [filename[:10] for filename in animal_filenames]
        values, counts = np.unique(dates, return_counts=True)
        extra_dates = values[counts > 2]

        for date in extra_dates:
            extra = [file for file in animal_filenames if date in file]
            extra_experiments.extend(extra)

    print(f'The following {len(extra_experiments)} files need to be checked for validity:')
    if len(extra_experiments) > 0:
        for exp in extra_experiments:
            print('    ' + exp)
    else:
        print('    No excess experiments!\n')


def find_repeated_session_files(target_directory, animals, regex_pattern, exp_to_match):
    all_files = glob(join(target_directory, regex_pattern))
    keep_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]
    keep_file_basenames = [os.path.basename(file).split('.')[0] for file in keep_files]
    repeated_experiments = []

    for animal in animals:
        animal_filenames = [file for file in keep_file_basenames if animal in file]
        animal_filenames.sort()
        dates = [filename[:10] for filename in animal_filenames]
        date_values, _ = np.unique(dates, return_counts=True)

        for date in date_values:
            day_exps = [file for file in animal_filenames if date in file]
            rigs = [filename.split('_')[6] for filename in day_exps]
            values, counts = np.unique(rigs, return_counts=True)
            if len(values) > 1:
                pass
            else:
                repeated_experiments.extend(day_exps)

    print(f'The following {len(repeated_experiments)} files have repeated sessions:')
    if len(repeated_experiments) > 0:
        for exp in repeated_experiments:
            print('    ' + exp)
    else:
        print('    No excess experiments!\n')


def find_files_with_missing_ca(target_directory, regex_pattern, exp_to_match):

    # Get files without calcium tif
    all_files = glob(join(target_directory, regex_pattern))
    keep_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]
    keep_file_basenames = [os.path.basename(file).split('.')[0] for file in keep_files]

    files_with_ca = glob(join(target_directory, r'*.tif'))
    keep_ca_files = [file for file in files_with_ca if any(exp in file for exp in exp_to_match)]
    keep_with_ca_basenames = [os.path.basename(file).split('.')[0] for file in keep_ca_files]
    files_without_ca = [file for file in keep_file_basenames if file not in keep_with_ca_basenames]

    print(f'\nThe following {len(files_without_ca)} files do not have calcium data:')
    if len(files_without_ca) > 0:
        for exp in files_without_ca:
            print('    ' + exp)
    else:
        print('    No files with missing calcium data!\n')


def find_empty_ROIS(target_directory, regex_pattern, exp_to_match):

    all_files = glob(join(target_directory, regex_pattern))
    keep_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]

    no_ROI_exps = []

    for calcium_path in keep_files:
        with h5py.File(calcium_path, mode='r') as f:
                calcium_data = np.array(f['calcium_data']).T

                # if there are no ROIs, skip
                if (type(calcium_data) == np.ndarray) and np.any(calcium_data.astype(str) == 'no_ROIs'):
                    no_ROI_exps.append(calcium_path)

    print(f'\nThe following {len(no_ROI_exps)} files do not have ROIS:')
    if len(no_ROI_exps) > 0:
        for exp in no_ROI_exps:
            print('    ' + exp)
    else:
        print('    No files with empty ROIs!\n')


def list_conflict_slugs(target_directory, regex_pattern, exp_to_match):

    all_files = glob(join(target_directory, regex_pattern))
    conflict_files = [file for file in all_files if any(exp in file for exp in exp_to_match)]
    conflict_file_basenames = [os.path.basename(file).split('.')[0] for file in conflict_files]

    print(f'\nThe following {len(conflict_file_basenames)} files have conflicts:')
    if len(conflict_file_basenames) > 0:
        for exp in conflict_file_basenames:
            print('    ' + exp)
    else:
        print('    No conflict files!\n')


base_dir = r'Z:\Prey_capture\VRExperiment'
conflict_dir = r"Z:\Prey_capture\conflict_files\VTuningWF"
animals = ["MM_220915_a", "MM_220928_a", "MM_221109_a", "MM_221110_a",
           'MM_230518_b', 'MM_230705_b', 'MM_230706_a', 'MM_230706_b']
rigs = ['VWheelWF', 'VTuningWF']

# Get session with repeated fixed or free recordings
find_repeated_session_files(base_dir, animals, r'*.avi', rigs)

# Get matched files and counts per animal
find_matched_files(base_dir, animals, r'*.avi', rigs)

# Get unmatched files
find_unmatched_files(base_dir, animals, r'*.avi', rigs)

# Find experiments more than one of the same kind of session
find_extra_session_files(base_dir, animals, r'*.avi', rigs)

# Find files with missing calcium tif
find_files_with_missing_ca(base_dir, r'*.avi', rigs)

list_conflict_slugs(conflict_dir, r'*.avi', rigs)

# Find experiments where no ROIs were found
# find_empty_ROIS(base_dir, r'*_calcium.hdf5', rigs)

print("done!")