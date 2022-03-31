import os
import yaml
import processing_parameters
import paths
import datetime
import functions_bondjango as bd
import h5py
import functions_misc as fm
import numpy as np

try:

    # get the path to the file, parse and turn into a dictionary
    target_path = snakemake.input[0]
    video_data = yaml.load(snakemake.params.file_info, Loader=yaml.FullLoader)
    # video_data = yaml.load(snakemake.params.file_info
    # get the save paths
    out_path = snakemake.output[0]

    # get the parts to assemble the input file path
    # animal = video_data
    rig = video_data['rig']
    if rig == 'miniscope':
        animal = '_'.join(os.path.basename(video_data['bonsai_path']).split('_')[7:10])
    else:
        animal = '_'.join(os.path.basename(video_data['bonsai_path']).split('_')[6:9])
    trial_date = datetime.datetime.strptime(video_data['date'], '%Y-%m-%dT%H:%M:%SZ')
    day = trial_date.strftime('%m_%d_%Y')
    # time = trial_date.strftime('%H_%M_%S')

    # print('scatter calcium:'+calcium_path)
    # print('scatter out:'+out_path)
except NameError:

    # define the target file
    search_string = processing_parameters.search_string

    # query the database for data to plot
    video_data = bd.query_database('analyzed_data', search_string + ',analysis_type:preprocessing')[0]

    # get the parts to assemble the input file path
    rig = video_data['rig']
    if rig == 'miniscope':
        animal = '_'.join(os.path.basename(video_data['analysis_path']).split('_')[7:10])
    else:
        animal = '_'.join(os.path.basename(video_data['analysis_path']).split('_')[6:9])
    trial_date = datetime.datetime.strptime(video_data['date'], '%Y-%m-%dT%H:%M:%SZ')
    day = trial_date.strftime('%m_%d_%Y')
    # time = trial_date.strftime('%H_%M_%S')

    # assemble the output path
    out_path = video_data['analysis_path'].replace('_preproc.hdf5', '_combinedanalysis.hdf5')

    # get the input file path
    target_path = os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'regressionday.hdf5')))

# get the regressed variables and their names
variable_list = processing_parameters.variable_list

# open the regression file
with h5py.File(target_path, 'r') as input_file, h5py.File(out_path, 'w') as output_file:
    if ('no_ROIs' in input_file.keys()) or ('nomini' in out_path) or ('nofluo' in out_path):
        output_file.create_dataset('no_ROIs', data=[])
        skip_entry = 1
    else:
        skip_entry = 0
        # get the frame list
        frame_list = np.array(input_file['frame_list'])
        # get this file's timestamp
        timestamp = int(''.join(os.path.basename(out_path).split('_')[3:6]))

        # get the index of the corresponding file in the list
        file_start = frame_list[frame_list[:, 0] == timestamp, 1][0]
        file_end = frame_list[frame_list[:, 0] == timestamp, 2][0]

        # get the time shifts
        time_shifts = processing_parameters.time_shifts

        # for all the time shifts
        for time_shift in time_shifts:
            # for shuffle and non shuffle
            for shuffler in np.arange(processing_parameters.regression_shuffles+1):
                if shuffler > 0:
                    suffix = 'shuffle'+str(shuffler)
                else:
                    suffix = 'real'
                # add the shift to the suffix
                suffix += '_shift'+str(time_shift)
                # for all the variables
                for idx in variable_list:
                    # generate the names of the fields
                    coeff_name = '_'.join(['coefficients', idx, suffix])
                    prediction_name = '_'.join(['prediction', idx, suffix])
                    cc_name = '_'.join(['cc', idx, suffix])
                    # check if the key is there, if not, skip
                    if coeff_name in input_file.keys():
                        # get the data
                        coefficients = np.array(input_file[coeff_name])
                        prediction = np.array(input_file[prediction_name])
                        cc = np.array(input_file[cc_name])
                        # get the relevant portion of prediction
                        prediction = prediction[file_start:file_end]
                        # save in the output file
                        output_file.create_dataset('regression/' + coeff_name, data=coefficients)
                        output_file.create_dataset('regression/' + prediction_name, data=prediction)
                        output_file.create_dataset('regression/' + cc_name, data=cc)

# # save the file
# with h5py.File(out_path, 'w') as f:
#     # save an empty
#     f.create_dataset('regression_data', data=results_dict)

if skip_entry == 0:
    # save as a new entry to the data base
    # assemble the entry data
    entry_data = {
        'analysis_type': 'combined_analysis',
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
