import numpy as np
import os
import sys
import re
import h5py
import cv2
from caiman.base.rois import register_multisession

# Insert the cwd for local imports
os.chdir(os.getcwd())
sys.path.insert(0, os.getcwd())

import paths
import processing_parameters
import functions_bondjango as bd
import functions_misc as fm
import functions_plotting as fplot
import matplotlib.pyplot as plt


def get_footprint_contours(calcium_data):
    contour_list = []
    contour_stats = []
    for frame in calcium_data:
        frame = frame * 255.
        frame = frame.astype(np.uint8)
        thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # get contours and filter out small defects
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # Only take the largest contour
        cntr = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cntr)
        perimeter = cv2.arcLength(cntr, True)
        compactness = 4 * np.pi * area / (perimeter + 1e-16) ** 2

        contour_list.append(cntr)
        contour_stats.append((area, perimeter, compactness))

    return contour_list, np.array(contour_stats)


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
    # calcium_path = [el['analysis_path'] for el in data_all if ('miniscope' not in el['slug'])]
    calcium_path = [el['analysis_path'] for el in data_all if ('miniscope' not in el['slug']) and
                    (day in el['slug'])]
    calcium_path.sort()

    # assemble the output path
    # out_path = os.path.join(paths.analysis_path, '_'.join((animal, rig, 'cellMatch.hdf5')))
    out_path = os.path.join(paths.analysis_path, '_'.join((day, animal, 'cellMatch.hdf5')))
    # out_path = os.path.join(paths.analysis_path, '_'.join((animal, 'cellMatch.hdf5')))

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

    date = os.path.basename(files)[:10]
    rig = os.path.basename(files).split('_')[6]

    with h5py.File(files, mode='r') as f:

        try:
            calcium_data = np.array(f['A'])
            max_proj = np.array(f['max_proj'])
        except KeyError:
            # Likely because there are no ROIs
            calcium_data = np.array(f['frame_list'])
            continue

        if rig in ['VTuning', 'VWheel', 'VTuningWF', 'VWheelWF']:
            trial = re.findall(r'fixed\d', files) + re.findall(r'free\d', files)
            trial = trial[0]
            date_list.append('_'.join((date, rig, trial)))
            rig_list.append(rig)
        else:
            date_list.append(date)

        # if there are no ROIs, skip
        if (type(calcium_data) == np.ndarray) and np.any(calcium_data.astype(str) == 'no_ROIs'):
            continue

        # clear the rois that don't pass the size criteria
        roi_stats = fm.get_roi_stats(calcium_data)
        contours, contour_stats = get_footprint_contours(calcium_data)

        if len(roi_stats.shape) == 1:
            roi_stats = roi_stats.reshape(1, -1)
            contour_stats = contour_stats.reshape(1, -1)

        areas = roi_stats[:, -1]
        compactness = contour_stats[:, -1]

        keep_vector = (areas > processing_parameters.roi_parameters['area_min']) & \
                      (areas < processing_parameters.roi_parameters['area_max']) & \
                      (compactness > processing_parameters.roi_parameters['compactness'])

        if np.all(keep_vector == False):
            continue

        calcium_data = calcium_data[keep_vector, :, :]

        # format and masks and store for matching
        footprint_list.append(np.moveaxis(calcium_data, 0, -1).reshape((-1, calcium_data.shape[0])))
        size_list.append(calcium_data.shape[1:])
        template_list.append(np.array(f['max_proj']))
        # frame_lists.append(np.array(f['frame_list']))

try:
    # run the matching software
    spatial_union, assignments, matchings = register_multisession(A=footprint_list, dims=size_list[0], templates=template_list, 
                                                                  align_flag=True, use_opt_flow=True, max_thr=0.1, thresh_cost=0.8, max_dist=8)
except Exception:
    # generate an empty array for saving
    assignments = np.zeros((1, len(date_list)))

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
