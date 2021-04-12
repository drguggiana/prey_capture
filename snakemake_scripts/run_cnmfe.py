import os
import paths
from snakemake_scripts.cnmfe import cnmfe_function
import sys
import json
import functions_bondjango as bd
import functions_io as fi
# from cnmfe_params import online_dict
import functions_misc as fm
from matplotlib import pyplot as plt
from skimage.transform import resize
import re
import processing_parameters


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
        search_string = processing_parameters.search_string_cnmfe
        animal = processing_parameters.animal
        day = processing_parameters.day
        rig = processing_parameters.rig
        # query the database for data to plot
        data_all = bd.query_database('video_experiment', search_string)
        # video_data = data_all[0]
        # video_path = video_data['tif_path']
        video_path = [el['tif_path'] for el in data_all]
        # overwrite data_all with just the urls
        data_all = {os.path.basename(el['bonsai_path'])[:-4]: el['url'] for el in data_all}
        # assemble the output path
        out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'calciumday.hdf5')))

    # delete the folder contents
    fi.delete_contents(paths.temp_path)
    # raise IndexError('stop here')
    # combine the selected files into a single tif
    out_path_tif, _, frames_list = fi.combine_tif(video_path, paths.temp_path)

    # get the extraction parameters
    try:
        online_dict = processing_parameters.mouse_parameters[animal]
    except KeyError:
        print(f'mouse {animal} not found, using default')
        online_dict = processing_parameters.mouse_parameters['default']

    # run cnmfe
    cnmfe_out, _ = cnmfe_function([out_path_tif], out_path, online_dict, save_output=False)

    # custom save the output to include the frames list
    for idx, el in enumerate(frames_list.iloc[:, 0]):
        # parse the line
        frames_list.iloc[idx, 0] = '_'.join(os.path.basename(el).split('_')[:6])

    cnmfe_out.frame_list = frames_list.values.tolist()
    # save the output
    cnmfe_out.save(out_path)

    # produce the contour figure
    img = cnmfe_out.estimates.corr_img
    img = resize(img, (cnmfe_out.estimates.dims[0], cnmfe_out.estimates.dims[1]))
    cnmfe_out.estimates.plot_contours(img=img)
    # assemble the pic path
    pic_path = out_path.replace('_calciumday.hdf5', '_calciumpic.tif')
    # also save a figure with the contours
    plt.savefig(pic_path, dpi=200)
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
