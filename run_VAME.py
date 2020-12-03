import functions_bondjango as bd
import os
import yaml
import paths
import numpy as np
import functions_plotting as fp
# # check if launched from snakemake, otherwise, prompt user
# try:
#     # get the path to the file, parse and turn into a dictionary
#     raw_path = snakemake.input[0]
#     calcium_path = snakemake.input[1]
#     files = yaml.load(snakemake.params.info, Loader=yaml.FullLoader)
#     # get the save paths
#     save_path = snakemake.output[0]
#     pic_path = snakemake.output[1]
# except NameError:
#     # USE FOR DEBUGGING ONLY (need to edit the search query and the object selection)
#     # define the search string
#
#     search_string = 'slug:03_12_2020_16_52_33_miniscope_MM_200129_b_succ'
#
#     # define the target model
#     target_model = 'video_experiment'
#     # get the queryset
#     files = bd.query_database(target_model, search_string)[0]
#     raw_path = files['bonsai_path']
#     calcium_path = files['bonsai_path'][:-4] + '_calcium.hdf5'
#     # assemble the save paths
#     save_path = os.path.join(paths.analysis_path,
#                              os.path.basename(files['bonsai_path'][:-4]))+'_preproc.hdf5'
#     pic_path = os.path.join(save_path[:-13] + '.png')


folder_path = r'F:\VAME_projects\VAME_prey-Dec1-2020\results\11_12_2019_16_50_34_miniscope_DG_190806_a_succ_nofluo\VAME_prey_model\kmeans-30\behavior_quantification'

# load the 3 matrices
adjacency = np.load(os.path.join(folder_path, 'adjacency_matrix.npy'))
motif_usage = np.load(os.path.join(folder_path, 'motif_usage.npy'))
transition = np.load(os.path.join(folder_path, 'transition_matrix.npy'))

fp.plot_image([np.array(adjacency)])
fp.plot_2d([[motif_usage]])
fp.plot_image([transition])

fp.show()


