import paths
import os
import yaml

# Main script
try:
    # get the raw output_path
    raw_path = snakemake.output[0]
    # get the parsed path
    dict_path = yaml.load(snakemake.params.output_info, Loader=yaml.FullLoader)
    # get the input paths
    paths_all = snakemake.input

except NameError:
    # define the output path
    raw_path = os.path.join(paths.analysis_path, 'full_run.txt')
    # define the string
    paths_all = ['test']


# save a text file
with open(raw_path, 'w') as f:
    f.writelines(paths_all)
