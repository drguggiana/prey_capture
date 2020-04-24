import functions_bondjango as bd
import pandas as pd
import numpy as np
import sklearn.mixture as mix
import sklearn.decomposition as decomp
import functions_plotting as fp
import functions_data_handling as fd

# get the data paths
try:
    data_path = snakemake.input[0]
except NameError:
    # define the search string
    search_string = 'result:succ, lighting:normal, rig:miniscope, =analysis_type:aggEncCA'
    # query the database for data to plot
    data_all = bd.query_database('analyzed_data', search_string)
    data_path = data_all[0]['analysis_path']
print(data_path)

# load the data
data = fd.aggregate_loader(data_path)

#

print('yay')
