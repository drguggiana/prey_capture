configfile: "snakemake_scripts/config_snake.yaml"
import os
import paths
import yaml
import json
import datetime


def yaml_to_json(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
    # escape the double quotes inside the json
    json_dict = json.dumps(python_dict).replace('"', '\\"')
    return json_dict

def yaml_list_to_json(wildcards):
    day_paths = day_selector(wildcards)

    python_list = [yaml.load(config["file_info"][os.path.basename(el)[:-4]], Loader=yaml.FullLoader)
                   for el in day_paths]
    url_dict = {}
    for idx, el in enumerate(day_paths):
        url_dict[os.path.basename(el)[:-4]] = python_list[idx]['url']
    json_dict = json.dumps(url_dict).replace('"', '\\"')
    return json_dict

rule dlc_extraction:
    input:
          lambda wildcards: os.path.join(config["target_path"], config["files"][wildcards.file] + '.avi'),
    output:
          os.path.join(config["target_path"], "{file}_dlc.h5"),
    params:
            info=yaml_to_json,
            dlc_path=config["dlc_path"],
    shell:
        r'conda activate DLC-GPU & python "{params.dlc_path}" "{input}" "{output}" "{params.info}"'


def dlc_input_selector(wildcards):
    if config["dlc_flag"][wildcards.file]:
        return rules.dlc_extraction.output
    else:
        return os.path.join(config["target_path"], config["files"][wildcards.file] + '.csv')


def day_selector(wildcards):
    name_parts = wildcards.file.split('_')
    day = datetime.datetime.strptime('_'.join(name_parts[0:3]), '%m_%d_%Y').strftime('%Y-%m-%d')
    animal = '_'.join([name_parts[3].upper()] + name_parts[4:6])
    info_list = [yaml.load(config["file_info"][el], Loader=yaml.FullLoader) for el in config["file_info"]]

    day_paths = [el['tif_path'] for el in info_list if
                 (config['calcium_flag'][os.path.basename(el['bonsai_path'])[:-4]]
                  and el['mouse']==animal and el['date'][:10]==day)]
    wildcards.day_paths = day_paths
    return day_paths


def day_animal_calcium_file(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
    animal = python_dict['mouse']
    day = datetime.datetime.strptime(python_dict['date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%m_%d_%Y')
    rig = python_dict['rig']
    return os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'calciumday.hdf5')))


rule calcium_extract:
    input:
        day_selector,
    output:
        os.path.join(paths.analysis_path, '{file}_calciumday.hdf5'),
    params:
        info=yaml_list_to_json,
        cnmfe_path=config["cnmfe_path"],
    shell:
        r'conda activate caiman & python "{params.cnmfe_path}" "{input}" "{output}" "{params.info}"'

rule calcium_scatter:
    input:
        day_animal_calcium_file,
    output:
        os.path.join(config["target_path"], "{file}_calcium.hdf5"),
    params:
        info=lambda wildcards: config["file_info"][wildcards.file]
    script:
        "snakemake_scripts/scatter_calcium.py"


def calcium_input_selector(wildcards):
    if config["calcium_flag"][wildcards.file]:
        # return rules.calcium_extraction.output
        return rules.calcium_scatter.output
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


rule match_cells:
    input:
        expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files']),
    output:
        os.path.join(paths.analysis_path, 'cellMatch_{query}.hdf5'),
    params:
        cnmfe_path=config['cnmfe_path'],
        info=yaml_to_json,
    shell:
          r'conda activate caiman & python "{params.cnmfe_path}" "{input}" "{output}" "{params.info}"'



rule visualize_aggregates:
    input:
        os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
    output:
        os.path.join(paths.figures_path, "averages_{query}.html")
    notebook:
        "snakemake_scripts/notebooks/Vis_averages.ipynb"
