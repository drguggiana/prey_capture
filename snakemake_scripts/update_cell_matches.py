import os
import pandas as pd

import processing_parameters
import functions_bondjango as bd
import functions_data_handling as fdh
from functions_matching import match_cells


def check_if_updated(file_path, updated_matches):
    with pd.HDFStore(file_path, mode='r+') as h:

        for key in h.keys():
            if key == '/no_ROIs':
                empty_flag = True
                break
            elif key == '/cell_matches':
                this_file_cell_matches = h[key]
            else:
                continue

        if not updated_matches.equals(this_file_cell_matches):
            h.remove('cell_matches')
            h.put('cell_matches', updated_matches)

    return


if __name__ == '__main__':
    try:
        # get the input
        preproc_file = snakemake.input[0]
        tc_file = snakemake.input[1]
        dummy_out_file = snakemake.output[0]

        basename = os.path.basename(preproc_file)
        day = basename[:10]
        mouse = '_'.join(basename.split('_')[7:10])

    except NameError:

        data_all = bd.query_database('analyzed_data', processing_parameters.search_string)
        parsed_search = fdh.parse_search_string(processing_parameters.search_string)
        day = parsed_search['slug']
        mouse = parsed_search['mouse']

        # get the paths to the files
        preproc_data = [el for el in data_all if '_preproc' in el['slug'] and
                        (parsed_search['mouse'].lower() in el['slug'])]
        preproc_file = preproc_data[0]['analysis_path']

        tc_data = [el for el in data_all if ('_tcday' in el['slug']) and
                   (parsed_search['mouse'].lower() in el['slug'])]
        tc_file = tc_data[0]['analysis_path']

        dummy_out_file = tc_file.replace('_tcday.hdf5', '_update_dummy.txt')

    # Search for the cell matching file
    cell_match_data = bd.query_database('analyzed_data', f'mouse:{mouse}, slug:{day}, analysis_type:cellmatching')
    cell_match_data = [el for el in cell_match_data if mouse.lower() in el['slug']]
    cell_match_file = cell_match_data[0]['analysis_path']

    # Load the cell matching file
    cell_matches = match_cells(cell_match_file)

    # Check if the preproc file has been updated
    check_if_updated(preproc_file, cell_matches)

    # Check if the tc file has been updated
    check_if_updated(tc_file, cell_matches)

    with open(dummy_out_file, 'w') as f:
        f.writelines([cell_match_file, preproc_file, tc_file])

    print(f'Updated {os.path.basename(preproc_file)} and {os.path.basename(tc_file)} \n'
          f'with {os.path.basename(cell_match_file)}')


