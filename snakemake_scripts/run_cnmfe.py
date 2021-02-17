import shutil
import os
import paths
from snakemake_scripts.cnmfe import cnmfe_function
import sys
import json
import functions_bondjango as bd
import functions_io as fi
from copy import deepcopy
from cnmfe_params import online_dict


if __name__ == "__main__":
    try:
        # get the target video path
        video_path = sys.argv[1]
        out_path = sys.argv[2]
        video_data = json.loads(sys.argv[3])
    except IndexError:
        # define the search string
        # search_string = 'result:succ, lighting:normal, rig:miniscope, imaging:doric'
        search_string = 'slug:08_06_2020_18_07_32_miniscope_DG_200701_a_succ'
        # query the database for data to plot
        data_all = bd.query_database('video_experiment', search_string)
        video_data = data_all[0]
        video_path = video_data['tif_path']
        # assemble the output path
        out_path = video_path.replace('.tif', '_calcium.hdf5')

    # delete the folder contents
    fi.delete_contents(paths.temp_path)

    # define the temp path
    temp_video_path = os.path.join(paths.temp_path, os.path.basename(video_path))
    # copy the file to the processing folder
    shutil.copyfile(video_path, temp_video_path)

    # run cnmfe
    cnmfe_out = cnmfe_function([temp_video_path], out_path, online_dict)

    # get the target model
    if video_data['rig'] == 'miniscope':
        target_model = 'video_experiment'
    else:
        target_model = 'vr_experiment'

    # update the bondjango entry (need to sort out some fields)
    ori_data = deepcopy(video_data)
    ori_data['fluo_path'] = out_path
    mouse = ori_data['mouse']
    ori_data['mouse'] = '/'.join((paths.bondjango_url, 'mouse', mouse, ''))
    ori_data['experiment_type'] = '/'.join((paths.bondjango_url, 'experiment_type', 'Free_behavior', ''))
    # fill in the preprocess_file field if present
    if len(ori_data['preproc_files']) > 0:
        # for all the elements there
        for idx, el in enumerate(ori_data['preproc_files']):
            ori_data['preproc_files'][idx] = \
                '/'.join((paths.bondjango_url, 'analyzed_data', el, ''))

    update_url = '/'.join((paths.bondjango_url, target_model, ori_data['slug'], ''))
    output_entry = bd.update_entry(update_url, ori_data)

    print(output_entry.status_code)
