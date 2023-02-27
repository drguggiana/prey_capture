import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import h5py
import sys
import json
from skimage.io import imread, imsave

import paths
import functions_bondjango as bd
import functions_io as fi
import functions_misc as fm
import functions_denoise_calcium as fdn
import processing_parameters
from snakemake_scripts.run_MiniAn_wirefree import minian_main


if __name__ == "__main__":

    try:
        # get the target video path
        video_path = sys.argv[1]

        # read the output path and the input file urls
        out_path = sys.argv[2]
        data_all = json.loads(sys.argv[3])

        # get the parts for the file naming
        name_parts = os.path.basename(out_path).split('_')
        day = '_'.join(name_parts[0:3])
        animal = '_'.join([name_parts[3].upper()] + name_parts[4:6])
        rig = name_parts[6]

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
        # overwrite data_all with just the urls
        data_all = {os.path.basename(el['bonsai_path'])[:-4]: el['url'] for el in data_all}
        # assemble the output path
        out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'calciumraw.hdf5')))

    # Here is some stupidity to deal with how Minian expects the directory to be formatted
    save_path = os.path.join(paths.temp_path, animal, rig)

    # denoise the video and save the tif in the modified temp path
    denoised_path = os.path.join(save_path, os.path.basename(video_path).replace('.tif', '_denoised.tif'))
    # delete the folder contents
    fi.delete_contents(paths.temp_minian)
    fi.delete_contents(paths.temp_path)
    os.makedirs(save_path)
    stack = fdn.denoise_stack(video_path)
    # allocate a list to store the original names and the number of frames
    frames_list = []
    print(f"Number of frames: {stack.shape[0]}")
    # save the file name and the number of frames
    frames_list.append([os.path.basename(video_path), stack.shape[0]])
    frames_list = pd.DataFrame(frames_list, columns=['filename', 'frame_number'])
    # Handle file renaming for denoised file
    out_path_tif = os.path.join(save_path, os.path.basename(video_path).replace('.tif', '_denoised.tif'))
    # Save the denoised stack
    imsave(out_path_tif, stack, plugin="tifffile", bigtiff=True)
    del stack

    try:
        print("starting minian")
        minian_out = minian_main(rig, animal, override_dpath=save_path)
        minian_out['processed_frames'] = np.array(minian_out['f'].frame)

        # custom save the output to include the frames list
        for idx, el in enumerate(frames_list.iloc[:, 0]):
            # parse the line
            frames_list.iloc[idx, 0] = '_'.join(os.path.basename(el).split('_')[:6])

        # save in an hdf5 file
        with h5py.File(out_path, 'w') as f:
            # save the calcium data
            for key, value in minian_out.items():
                f.create_dataset(key, data=np.array(value))
            # save the frames list
            f.create_dataset('frame_list', data=frames_list.values.astype('S'))

        # produce the contour figure
        calcium_pic = np.sum(minian_out['A'] > 0, axis=0)
        # plt.imshow(calcium_pic)

        # assemble the pic path
        pic_path = out_path.replace('_calciumraw.hdf5', '_calciumpic.tif')
        # also save a figure with the contours
        plt.savefig(pic_path, dpi=200)

    except (ValueError, np.linalg.LinAlgError):
        print(f'File {video_path} contained no ROIs')
        # save as a hdf5 file
        with h5py.File(out_path, 'w') as f:
            # save an empty
            f.create_dataset('frame_list', data='no_ROIs')
        # define pic_path as empty
        pic_path = ''

    # assemble the entry data
    entry_data = {
        'analysis_type': 'calciumraw',
        'analysis_path': out_path,
        'date': '',
        'pic_path': pic_path,
        'result': 'multi',
        'rig': rig,
        'lighting': 'multi',
        'imaging': 'multi',
        'slug': fm.slugify(os.path.basename(out_path)[:-5]),
        'video_analysis': [el for el in data_all.values() if 'miniscope' in el],
        'vr_analysis': [el for el in data_all.values() if 'miniscope' not in el],
    }

    # check if the entry already exists, if so, update it, otherwise, create it
    update_url = '/'.join((paths.bondjango_url, 'analyzed_data', entry_data['slug'], ''))
    output_entry = bd.update_entry(update_url, entry_data)
    if output_entry.status_code == 404:
        # build the url for creating an entry
        create_url = '/'.join((paths.bondjango_url, 'analyzed_data', ''))
        output_entry = bd.create_entry(create_url, entry_data)

    print('The output status was %i, reason %s' %
          (output_entry.status_code, output_entry.reason))
    if output_entry.status_code in [500, 400]:
        print(entry_data)

    print('<3')
