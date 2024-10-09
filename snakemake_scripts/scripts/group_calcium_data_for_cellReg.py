import os
import h5py
import shutil
from scipy import io

import processing_parameters
import functions_bondjango as bd
import functions_data_handling as fdh


def load_footprints_save_mat(file_path, save_dir):
    """
    Load the footprints from the file and save them as a .mat file
    :param file_path: str, path to the file
    :param save_dir: str, path to the directory to save the file
    :return: None
    """

    try:
        # load the footprints
        with h5py.File(file_path, 'r') as f:

            # Check if there are footprints in this file. If not, return a fail flag
            footprints = f['A'][()]

        # save the footprints as a .mat file
        save_name = os.path.basename(file_path).replace('.hdf5', '.mat')
        io.savemat(os.path.join(save_dir, save_name), {'footprints': footprints})
        return 'success'

    except KeyError:
        return 'fail'


if __name__ == '__main__':
    try:
        # get the input
        tc_file = snakemake.input[0]
        dummy_out_file = snakemake.output[0]

        basename = os.path.basename(tc_file)
        day = basename[:10]
        mouse = '_'.join(basename.split('_')[7:10])
        result = basename.split('_')[10]
        rig = basename.split('_')[6]
        lighting = basename.split('_')[11]

    except NameError:

        data_all = bd.query_database('analyzed_data', processing_parameters.search_string)
        parsed_search = fdh.parse_search_string(processing_parameters.search_string)
        day = parsed_search['slug']
        mouse = parsed_search['mouse']

        # get the paths to the files
        tc_data = [el for el in data_all if '_tcday' in el['slug'] and
                        (parsed_search['mouse'].lower() in el['slug'])]
        tc_file = tc_data[0]['analysis_path']
        result = tc_data[0]['result']
        rig = tc_data[0]['rig']
        lighting = tc_data[0]['lighting']

        dummy_out_file = tc_file.replace('_preproc.hdf5', '_cellReg_setup_dummy.txt')

    # Make sure the result flag is correct
    if result not in ['control', 'repeat', 'fullfield'] and rig in ['VWheel', 'VWheelWF', 'VTuning', 'VTuningWF']:
        result = 'multi'

    if lighting not in ['normal', 'dark'] and rig in ['VWheel', 'VWheelWF', 'VTuning', 'VTuningWF']:
        lighting = 'normal'

    # Path to the base_folder
    reg_basepath = r"Z:\Prey_capture\WF_cell_matching_cellreg"

    # Search for the cell matching file
    calcium_data = bd.query_database('analyzed_data', f'slug:{day}, analysis_type:calciumraw')
    calcium_data = [el for el in calcium_data if mouse.lower() in el['slug']]

    # Check if at least two files are present for matching
    if len(calcium_data) < 2:
        raise ValueError(f'Insufficient or no calcium data found for {mouse} on {day}. Skipping...')
    else:
        print(f'Found {len(calcium_data)} files for {mouse} on {day}')

        # Get the Ca2+ file paths
        calcium_data_paths = [data['analysis_path'] for data in calcium_data]
        calcium_data_slugs = [data['slug'] for data in calcium_data]

        # Create the experiment directory, if it doesn't exist already
        this_exp_dir = os.path.join(reg_basepath, result, lighting, f'{day}_{mouse}')
        if not os.path.isdir(this_exp_dir):
            os.makedirs(this_exp_dir)
            all_present = False

        else:
            # Check if the footprints are already saved
            present_files = [os.path.basename(file).lower().split('.')[0] for
                             file in os.listdir(this_exp_dir) if
                             file.endswith('.mat')]
            is_slug = [slug in present_files for slug in calcium_data_slugs]
            all_present = all(is_slug)

        if all_present:
            # Skip if already saved
            print(f'Footprints already saved for {mouse} on {day}. Skipping...')
        else:
            # Load the ROI footprints and save to the new directory for the cell matching
            for i, file in enumerate(calcium_data_paths):
                flag = load_footprints_save_mat(file, this_exp_dir)

                if flag == 'success':
                    continue
                else:
                    print(f'No footprints found in {file}. Deleting folder and skipping...')
                    shutil.rmtree(this_exp_dir)

            if i == len(calcium_data_paths):
                print(f'Saved footprints for {mouse} on {day} to {this_exp_dir}')

    # Write the dummy file
    with open(dummy_out_file, 'w') as f:
        f.writelines(calcium_data_paths)

    print('Done!')
