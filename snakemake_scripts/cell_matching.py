import functions_bondjango as bd
import paths
import os
from caiman.base.rois import register_multisession

import h5py
import numpy as np
import functions_misc as fm
import processing_parameters
import sys
import re
import json
import functions_plotting as fplot

# Main script
try:
    # get the target video path
    video_path = sys.argv[1]
    # find the occurrences of .tif terminators
    ends = [el.start() for el in re.finditer('.hdf5', video_path)]
    # allocate the list of videos
    video_list = []
    count = 0
    # read the paths
    for el in ends:
        video_list.append(video_path[count:el + 5])
        count = el + 6
    print(video_list)
    calcium_path = video_list
    # read the output path and the input file urls
    out_path = sys.argv[2]
    # data_all = json.loads(sys.argv[3])
    # get the parts for the file naming
    name_parts = out_path.split('_')
    animal = name_parts[:3]
    rig = name_parts[3]

except IndexError:

    # get the search string
    animal = processing_parameters.animal
    rig = processing_parameters.rig
    search_string = 'slug:%s, analysis_type:calciumday' % (fm.slugify(animal))
    # query the database for data to plot
    data_all = bd.query_database('analyzed_data', search_string)
    # get the paths to the files
    calcium_path = [el['analysis_path'] for el in data_all if 'miniscope' not in el['slug']]
    # # for testing, filter calcium path
    # calcium_path = [el for el in calcium_path if ('03_24' in el) or ('03_23' in el) or ('03_29' in el)]
    # assemble the output path
    # out_path = os.path.join(paths.analysis_path, '_'.join((animal, rig, 'cellMatch.hdf5')))
    out_path = os.path.join(paths.analysis_path, '_'.join((animal, 'cellMatch.hdf5')))

# load the data for the matching
footprint_list = []
size_list = []
template_list = []
# # also store the frame lists
# frame_lists = []
# also store the date for each file
date_list = []
# load the calcium data
for files in calcium_path:
    with h5py.File(files, mode='r') as f:
        try:
            calcium_data = np.array(f['A'])
        except KeyError:
            continue

        # if there are no ROIs, skip
        if (type(calcium_data) == np.ndarray) and (calcium_data == 'no_ROIs'):
            continue
        # clear the rois that don't pass the size criteria
        areas = fm.get_roi_stats(calcium_data)[:, -1]
        keep_vector = (areas > processing_parameters.roi_parameters['area_min']) & \
                      (areas < processing_parameters.roi_parameters['area_max'])
        calcium_data = calcium_data[keep_vector, :, :]
        # format and masks and store for matching
        footprint_list.append(np.moveaxis(calcium_data, 0, -1).reshape((-1, calcium_data.shape[0])))
        size_list.append(calcium_data.shape[1:])
        template_list.append(np.zeros(size_list[0]))
        # template_list.append(np.array(f['max_proj']))
        date_list.append(os.path.basename(files)[:10])
        # frame_lists.append(np.array(f['frame_list']))

try:
    # run the matching software
    spatial_union, assignments, matchings = register_multisession(
        A=footprint_list, dims=size_list[0], templates=template_list, thresh_cost=0.9)
except Exception:
    # generate an empty array for saving
    assignments = np.ones((len(date_list), 1))

# fplot.plot_image([spatial_union[:, 0].reshape((630, 630))])
# save the matching results
with h5py.File(out_path, 'w') as f:
    # # save the calcium data
    # for key, value in minian_out.items():
    #     f.create_dataset(key, data=np.array(value))
    f.create_dataset('assignments', data=assignments)
    # f.create_dataset('matchings', data=np.array(matchings))
    f.create_dataset('date_list', data=np.array(date_list).astype('S10'))

# create the appropriate bondjango entry
entry_data = {
    'analysis_type': 'cellmatching',
    'analysis_path': out_path,
    'date': '',
    'pic_path': '',
    'result': 'multi',
    'rig': rig,
    'lighting': 'multi',
    'imaging': 'multi',
    'slug': fm.slugify(os.path.basename(out_path)[:-5]),
    # 'video_analysis': [el for el in data_all.values() if 'miniscope' in el],
    # 'vr_analysis': [el for el in data_all.values() if 'miniscope' not in el],
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
