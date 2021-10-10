# imports
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import functions_bondjango as bd
import processing_parameters
import paths
import yaml
import h5py
import pandas as pd
import numpy as np
import vame
from sklearn import cluster

# check if launched from snakemake, otherwise, prompt user
try:
    # get the path to the file, parse and turn into a dictionary
    input_path = snakemake.input[0]

    files = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
    # get the save paths
    save_path = snakemake.output[0]
except NameError:
    # USE FOR DEBUGGING ONLY (need to edit the search query and the object selection)
    # define the search string
    search_string = processing_parameters.search_string

    # define the target model
    if 'miniscope' in search_string:
        target_model = 'video_experiment'
    else:
        target_model = 'vr_experiment'

    # get the queryset
    files = bd.query_database(target_model, search_string)[0]
    input_path = files['bonsai_path'].replace('.csv', '_rawcoord.hdf5').replace('VideoExperiment', 'AnalyzedData')
    # assemble the save paths
    save_path = os.path.join(paths.analysis_path,
                             os.path.basename(files['bonsai_path'][:-4]))+'_motifs.hdf5'

# load the data
with h5py.File(input_path, 'r') as f:
    try:
        values = np.array(f['matched_calcium/block0_values'])
        labels = np.array(f['matched_calcium/block0_items']).astype(str)
    except KeyError:
        values = np.array(f['full_traces/block0_values'])
        labels = np.array(f['full_traces/block0_items']).astype(str)
    data = pd.DataFrame(values, columns=labels)

# get the list of columns
column_list_all = data.columns
column_list = [el for el in column_list_all if (('x' in el) or ('y' in el))]

# get the config info
config_file = os.path.join(paths.vame_path, paths.vame_current_model_name, 'config.yaml')
config = vame.read_config(config_file)

# get the egocentrically aligned coordinates
aligned_traj, frames = vame.egocentric_alignment(config, pose_ref_index=[0, 7], crop_size=(200, 200), 
                                                 use_video=False, video_format='.mp4', check_video=False, 
                                                 save_flag=False, filename=[data], column_list=column_list)

# get the motifs and latents
# assemble the template path
first_file = os.listdir(os.path.join(paths.vame_results, 'results'))[0]
template_path = os.path.join(paths.vame_results, 'results', first_file, 'VAME', 'kmeans-15', 'cluster_center_'+first_file+'.npy')
# load the cluster centers
cluster_centers = np.load(template_path)

# create a new kmeans object
random_state = config['random_state_kmeans']
n_init = config['n_init_kmeans']
n_cluster = config['n_cluster']
kmeans_object = cluster.KMeans(init='k-means++', n_clusters=n_cluster, random_state=random_state, n_init=n_init)
# set the cluster centers from the template file to apply to this file
kmeans_object.cluster_centers_ = cluster_centers
# set the number fo threads, required when building the kmeans object without fitting
kmeans_object._n_threads = 1
# run the pose segmentation
latents, clusters = vame.batch_pose_segmentation(config, [0], aligned_traj, kmeans_obj=kmeans_object)

# save the file and create the bondjango entry
with h5py.File(save_path, 'w') as f:
    f.create_dataset('egocentric_coord', data=aligned_traj)
    f.create_dataset('latents', data=np.squeeze(latents))
    f.create_dataset('motifs', data=np.squeeze(clusters))
    f.create_dataset('columns', data=column_list)
# assemble the entry data
entry_data = {
    'analysis_type': 'motifs',
    'analysis_path': save_path,
    'date': files['date'],
    'result': files['result'],
    'rig': files['rig'],
    'lighting': files['lighting'],
    'imaging': files['imaging'],
    'slug': files['slug'] + '_motifs',
    'video_analysis': [files['url']] if files['rig'] == 'miniscope' else [],
    'vr_analysis': [] if files['rig'] == 'miniscope' else [files['url']],
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