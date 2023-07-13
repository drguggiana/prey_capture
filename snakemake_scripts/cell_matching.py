import numpy as np
import os
import sys
import re
import h5py
from caiman.base.rois import register_multisession


import paths
import processing_parameters
import functions_bondjango as bd
import functions_misc as fm
import functions_plotting as fplot

# Main script
try:
    # get the target video path
    video_path = sys.argv[1]
    # read the output path and the input file urls
    out_path = sys.argv[2]

    # find the occurrences of .hdf5 terminators
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

except IndexError:

    # get the search string
    animal = processing_parameters.animal
    day = processing_parameters.day
    rig = processing_parameters.rig

    search_string = 'slug:%s, analysis_type:calciumraw' % (fm.slugify(animal))
    # query the database for data to plot
    data_all = bd.query_database('analyzed_data', search_string)
    # get the paths to the files
    calcium_path = [el['analysis_path'] for el in data_all if 'miniscope' not in el['slug']]
    calcium_path.sort()

    # assemble the output path
    # out_path = os.path.join(paths.analysis_path, '_'.join((animal, rig, 'cellMatch.hdf5')))
    # out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, 'cellMatch.hdf5')))
    out_path = os.path.join(paths.analysis_path, '_'.join((animal, 'cellMatch.hdf5')))

# load the data for the matching
footprint_list = []
size_list = []
template_list = []
# # also store the frame lists
# frame_lists = []
# also store the date for each file
date_list = []
# store the rig for each file
rig_list = []
# load the calcium data
for files in calcium_path:
    with h5py.File(files, mode='r') as f:

        try:
            calcium_data = np.array(f['A'])
        except KeyError:
            continue

        # if there are no ROIs, skip
        if (type(calcium_data) == np.ndarray) and np.any(calcium_data.astype(str) == 'no_ROIs'):
            continue
        # clear the rois that don't pass the size criteria
        areas = fm.get_roi_stats(calcium_data)[:, -1]
        keep_vector = (areas > processing_parameters.roi_parameters['area_min']) & \
                      (areas < processing_parameters.roi_parameters['area_max'])

        if np.all(keep_vector == False):
            continue

        calcium_data = calcium_data[keep_vector, :, :]

        # format and masks and store for matching

        footprint_list.append(np.moveaxis(calcium_data, 0, -1).reshape((-1, calcium_data.shape[0])))
        size_list.append(calcium_data.shape[1:])
        template_list.append(np.zeros(size_list[0]))
        # template_list.append(np.array(f['max_proj']))

        date = os.path.basename(files)[:10]
        rig = os.path.basename(files).split('_')[6]
        if rig in ['VTuning', 'VWheel', 'VTuningWF', 'VWheelWF']:
            trial = re.findall(r'fixed\d', files) + re.findall(r'free\d', files)
            trial = trial[0]
            date_list.append('_'.join((date, rig, trial)))
            rig_list.append(rig)
        else:
            date_list.append(date)
        # frame_lists.append(np.array(f['frame_list']))

try:
    # run the matching software
    spatial_union, assignments, matchings = register_multisession(
        A=footprint_list, dims=size_list[0], templates=template_list, thresh_cost=0.7)
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
    f.create_dataset('date_list', data=np.array(date_list).astype('S30'))

# Check if there are unique rigs or not
if len(set(rig_list)) == 1:
    rig = rig_list[0]
else:
    rig = 'multi'

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
