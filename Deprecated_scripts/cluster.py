# imports

import paths
import functions_bondjango as bd
import pandas as pd
import numpy as np
import sklearn.mixture as mix
import sklearn.decomposition as decomp
import functions_plotting as fp
import functions_data_handling as fd
import umap

# get the data paths
try:
    data_path = snakemake.input[0]
except NameError:
    # define the search string
    search_string = 'result:succ, lighting:normal, rig:miniscope, =analysis_type:aggEnc'
    # query the database for data to plot
    data_all = bd.query_database('analyzed_data', search_string)
    data_path = data_all[0]['analysis_path']
print(data_path)

# load the data
data = fd.aggregate_loader(data_path)

# assemble the array with the parameters of choice
target_data = data.loc[:, ['mouse_cricket_distance'] + ['encounter_id', 'trial_id']].groupby(
    ['trial_id', 'encounter_id']).agg(list).to_numpy()
target_data = np.array([el for sublist in target_data for el in sublist])

# PCA the data before clustering
pca = decomp.PCA()
transformed_data = pca.fit_transform(target_data)
fp.plot_2d([[pca.explained_variance_ratio_]])

# cluster the data

# define the vector of components
component_vector = [2, 3, 4, 5, 10, 20, 30]
# allocate memory for the results
gmms = []
# for all the component numbers
for comp in component_vector:
    # # define the number of components
    # n_components = 10
    gmm = mix.GaussianMixture(n_components=comp, covariance_type='diag', n_init=10)
    gmm.fit(transformed_data[:, :7])
    gmms.append(gmm.bic(transformed_data[:, :7]))

# plot the BIC
fp.plot_2d([[np.array([component_vector, gmms]).T]])

# select the minimum bic number of components
n_components = np.array(component_vector)[np.argmin(gmms)]
# predict the cluster indexes
gmm = mix.GaussianMixture(n_components=n_components, covariance_type='diag', n_init=10)
cluster_idx = gmm.fit_predict(transformed_data[:, :7])
# add the cluster indexes to the dataframe
cluster_data = np.array([np.mean(target_data[cluster_idx == el, :], axis=0) for el in np.arange(n_components)])
# plot the results
fp.plot_2d([cluster_data])

# embed the data via UMAP
reducer = umap.UMAP(min_dist=0.5, n_neighbors=10)
embedded_data = reducer.fit_transform(transformed_data)

# plot the embedding
fp.plot_scatter([[embedded_data]])


fp.show()
