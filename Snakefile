configfile: "snakemake_scripts/config_snake.yaml"
import os
import paths
import yaml
import json


def yaml_to_json(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
    # escape the double quotes inside the json
    json_dict = json.dumps(python_dict).replace('"', '\\"')
    return json_dict

rule dlc_extraction:
    input:
          lambda wildcards: os.path.join(config["target_path"], config["files"][wildcards.file] + '.avi'),
    output:
          os.path.join(config["target_path"], "{file}_dlc.h5"),
    params:
            info=yaml_to_json,
            dlc_path=config["dlc_path"],
          # info=lambda wildcards: config["file_info"][wildcards.file]
    # notebook:
    #     "snakemake_scripts/notebooks/Process_videoexperiment.ipynb"
    shell:
        r'conda activate DLC-GPU & python "{params.dlc_path}" "{input}" "{output}" "{params.info}"'
        # r'conda activate DLC-GPU & D:/ProgramData/Miniconda3/envs/DLC-GPU/python.exe "D:/Code Repos/prey_capture/snakemake_scripts/run_dlc.py" '
        # + r'"{input}" "{output}" "{params.info}"'

def dlc_input_selector(wildcards):
    if config["dlc_flag"][wildcards.file]:
        return rules.dlc_extraction.output
    else:
        return os.path.join(config["target_path"], config["files"][wildcards.file] + '.csv')

rule calcium_extraction:
    input:
          lambda wildcards: os.path.join(config["target_path"], config["files"][wildcards.file] + '.tif'),
    output:
          os.path.join(config["target_path"], "{file}_calcium.hdf5"),
    params:
          info=yaml_to_json,
          cnmfe_path=config["cnmfe_path"],
    shell:
          r'conda activate caiman & python "{params.cnmfe_path}" "{input}" "{output}" "{params.info}"'

def calcium_input_selector(wildcards):
    if config["calcium_flag"][wildcards.file]:
        return rules.calcium_extraction.output
    else:
        return os.path.join(config["target_path"], config["files"][wildcards.file] + '.avi')

rule preprocess:
    input:
          dlc_input_selector,
          calcium_input_selector,
    output:
          os.path.join(paths.analysis_path, "{file}_preproc.hdf5"),
          os.path.join(paths.analysis_path, "{file}.png")
    params:
          info=lambda wildcards: config["file_info"][wildcards.file]
    script:
          "snakemake_scripts/preprocess_all.py"


rule just_preprocess:
    input:
          expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files'])
    output:
          os.path.join(paths.analysis_path, "just_preprocess.txt")
    wildcard_constraints:
          query=".*_agg.*"
    params:
          file_info=expand("{info}", info=config["file_info"].values()),
          output_info=config["output_info"]
    script:
          "snakemake_scripts/just_preprocess.py"


rule aggregate_preprocessed:
    input:
          expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files'])
    output:
          os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
    wildcard_constraints:
          query=".*_agg.*"
    params:
          file_info=expand("{info}", info=config["file_info"].values()),
          output_info=config["output_info"]
    script:
          "snakemake_scripts/aggregate.py"


rule triggered_averages:
    input:
          expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files'])
    output:
         os.path.join(paths.analysis_path, "preprocessing_{query}_trigAveCA.hdf5")
    params:
          file_info=expand("{info}", info=config["file_info"].values()),
          output_info=config["output_info"],
          interval=config['interval'],
    script:
          "snakemake_scripts/trigAve.py"


rule visualize_aggregates:
    input:
        os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
    output:
        os.path.join(paths.figures_path, "averages_{query}.html")
    notebook:
        "snakemake_scripts/notebooks/Vis_averages.ipynb"
