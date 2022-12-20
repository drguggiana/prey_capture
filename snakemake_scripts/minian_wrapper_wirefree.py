import os
import paths
from snakemake_scripts.run_MiniAn import minian_main
import sys
import json
import functions_bondjango as bd
import functions_io as fi
import functions_misc as fm
import functions_denoise_calcium as fdn
from matplotlib import pyplot as plt
import re
import processing_parameters
import h5py
import numpy as np


if __name__ == "__main__":

    try:
        # get the target video path
        video_path = sys.argv[1]
        # find the occurrences of .tif terminators
        ends = [el.start() for el in re.finditer('.tif', video_path)]
        # allocate the list of videos
        video_list = []
        count = 0
        # read the paths
        for el in ends:
            video_list.append(video_path[count:el+4])
            count = el + 5

        video_path = video_list
        # read the output path and the input file urls
        out_path = sys.argv[2]
        data_all = json.loads(sys.argv[3])
        # get the parts for the file naming
        name_parts = out_path.split('_')
        day = name_parts[0]
        animal = name_parts[1]
        rig = name_parts[2]

    except IndexError:
        # get the search string
        # search_string = processing_parameters.search_string_calcium
        animal = processing_parameters.animal
        day = processing_parameters.day
        rig = processing_parameters.rig
        # search_string = 'rig:%s, imaging:doric, mouse:%s, slug:%s' % (rig, animal, day)
        search_string = 'imaging:doric, mouse:%s, slug:%s' % (animal, day)
        # query the database for data to plot
        # data_all = bd.query_database('video_experiment', search_string)
        data_all = bd.query_database('vr_experiment', processing_parameters.search_string)
        # video_data = data_all[0]
        # video_path = video_data['tif_path']
        video_path = [el['tif_path'] for el in data_all]
        # overwrite data_all with just the urls
        data_all = {os.path.basename(el['bonsai_path'])[:-4]: el['url'] for el in data_all}
        # assemble the output path
        out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, 'calciumday.hdf5')))

    for video in video_path:
        # delete the folder contents
        fi.delete_contents(paths.temp_path)

        # denoise the video
        out_path_tif, _, frames_list = fdn.denoise_stack(video, paths.temp_path)

        # # combine the selected files into a single tif
        # out_path_tif, _, frames_list = fi.combine_tif(video_path, paths.temp_path)

        # # get the extraction parameters
        # try:
        #     online_dict = processing_parameters.mouse_parameters[animal]
        # except KeyError:
        #     print(f'mouse {animal} not found, using default')
        #     online_dict = processing_parameters.mouse_parameters['default']

        try:
            # run minian
            minian_out = minian_main()

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
            plt.imshow(calcium_pic)

            # assemble the pic path
            pic_path = out_path.replace('_calciumday.hdf5', '_calciumpic.tif')
            # also save a figure with the contours
            plt.savefig(pic_path, dpi=200)

        except (ValueError, np.linalg.LinAlgError):
            print(f'File {video_path} contained no ROIs')
            # save in an hdf5 file
            with h5py.File(out_path, 'w') as f:
                # save an empty
                f.create_dataset('frame_list', data='no_ROIs')
            # define pic_path as empty
            pic_path = ''



     # What to do here, as we don't do a calcium day file

    # assemble the entry data
    entry_data = {
        'analysis_type': 'calciumday',
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
