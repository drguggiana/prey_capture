import sys
import os
import json
from skimage import io

import paths
import processing_parameters
import functions_bondjango as bd
import functions_denoise_calcium as fdn


if __name__ == "__main__":

    try:
        # get the target video path
        video_path = sys.argv[1]
        out_path = sys.argv[2]
        data_all = json.loads(sys.argv[3])
        # get the parts for the file naming
        name_parts = os.path.basename(out_path).split('_')
        day = '_'.join(name_parts[0:3])
        animal = '_'.join([name_parts[3].upper()] + name_parts[4:6])
        rig = name_parts[6]

        print(video_path)
        print(out_path)
        print(day, animal, rig)

    except IndexError:
        # get the search string
        # search_string = processing_parameters.search_string
        animal = processing_parameters.animal
        day = processing_parameters.day
        rig = processing_parameters.rig
        search_string = 'rig:%s, imaging:wirefree, mouse:%s, slug:%s' % (rig, animal, day)

        # query the database for data to plot
        data_all = bd.query_database('vr_experiment', search_string)
        # video_data = data_all[0]
        video_path = data_all[0]['tif_path']
        # video_path = [el['tif_path'] for el in data_all]
        # overwrite data_all with just the urls
        data_all = {os.path.basename(el['bonsai_path'])[:-4]: el['url'] for el in data_all}

    print(f"Denoising {video_path} ...")
    denoised_stack = fdn.denoise_stack(video_path)

    # allocate a list to store the original names and the number of frames
    frames_list = []
    print(f"Number of frames: {denoised_stack.shape[0]}")
    # save the file name and the number of frames
    frames_list.append([os.path.basename(video_path), denoised_stack.shape[0]])

    # Handle file renaming for denoised file
    out_path_tif = os.path.join(paths.temp_path, os.path.basename(video_path).replace('.tif', '_denoised.tif'))
    out_path_log = os.path.join(paths.temp_path, os.path.basename(video_path).replace('.tif', '_denoised.csv'))

    # Save the denoised stack
    io.imsave(out_path_tif, denoised_stack, plugin="tifffile", bigtiff=True)

    print("Done!\n")





