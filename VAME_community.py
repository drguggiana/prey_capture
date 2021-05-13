# Initialize a new VAME project
import vame
import functions_bondjango as bd
import functions_vame as fv
import paths
import random
import os
import shutil
import h5py
import numpy as np


# Manually define the config path if project has already been created
config = r"D:\VAME_projects\VAME_prey_6-Apr28-2021\config.yaml"

# Visualize embedding
# vame.visualization(config, label=None)

# Community analysis

vame.community(config, show_umap=False, cut_tree=None)
