configfile: "snakemake_scripts/config_snake.yaml"
import os
import paths
import yaml
import json
import datetime
import numpy as np
import processing_parameters


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

# def yaml_animal(wildcards):
#     animal_paths = [os.path.basename(el) for el in matched_input(wildcards)]
#     # animal = '_'.join((wildcards.file.split('_')[:3]))
#     # python_list = [yaml.load(config["file_info"][el], Loader=yaml.FullLoader)
#     #                for el in config["file_info"].keys() if animal in el]
#     # python_list = [yaml.load(config["file_info"][os.path.basename(el)[:-4]], Loader=yaml.FullLoader)
#     #                for el in animal_paths]
#     # url_dict = {}
#     # for idx, el in enumerate(animal_paths):
#     #     url_dict[os.path.basename(el)[:-4]] = python_list[idx]['url']
#     json_dict = json.dumps(animal_paths).replace('"', '\\"')
#     return json_dict

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
                 (config['calcium_flag'][os.path.basename(el['avi_path'])[:-4]]
                  and el['mouse']==animal and el['date'][:10]==day)]
    wildcards.day_paths = day_paths
    return day_paths


def day_animal_calcium_file(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
    animal = python_dict['mouse']
    day = datetime.datetime.strptime(python_dict['date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%m_%d_%Y')
    # rig = python_dict['rig']
    return os.path.join(paths.analysis_path, '_'.join((day, animal, 'calciumday.hdf5')))


rule calcium_extract:
    input:
        day_selector,
    output:
        os.path.join(paths.analysis_path, '{file}_calciumday.hdf5'),
    params:
        info=yaml_list_to_json,
        cnmfe_path=config["cnmfe_path"],
    shell:
        # r'conda activate caiman & python "{params.cnmfe_path}" "{input}" "{output}" "{params.info}"'
        r'conda activate minian & python "{params.cnmfe_path}" "{input}" "{output}" "{params.info}"'

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
        return rules.calcium_scatter.output
    else:
        return os.path.join(config["target_path"], config["files"][wildcards.file] + '.avi')


def matched_input(wildcards):
    name_parts = wildcards.file.split('_')
    # day = datetime.datetime.strptime('_'.join(name_parts[0:3]), '%m_%d_%Y').strftime('%Y-%m-%d')
    animal = '_'.join([name_parts[0].upper()] + name_parts[1:3])
    # rig = name_parts[3]

    info_list = [yaml.load(config["file_info"][el], Loader=yaml.FullLoader) for el in config["file_info"]]
    # leave only the files with calcium data
    available_dates = np.unique([el['slug'][:10] for el in info_list
                                 if (config['calcium_flag'][os.path.basename(el['avi_path'])[:-4]] == True) &
                                 (animal in el['mouse'])])

    # assemble the paths to the calciumday files
    animal_routes = ['_'.join((el, animal, 'calciumday.hdf5')) for el in available_dates]
    animal_routes = [os.path.join(paths.analysis_path, el) for el in animal_routes]
    # wildcards.day_routes = day_routes
    return animal_routes


rule match_cells:
    input:
        # expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files']),
        matched_input,
    output:
        os.path.join(paths.analysis_path,'{file}_cellMatch.hdf5'),
    # params:
        # cnmfe_path=config['cnmfe_path'],
        # info=yaml_animal,
    shell:
        r'conda activate caiman & python "{paths.matching_script}" "{input}" "{output}"'


def match_selector(wildcards):
    if config["calcium_flag"][wildcards.file]:
        # return rules.calcium_extraction.output
        python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
        animal = python_dict['mouse']
        # rig = python_dict['rig']
        return os.path.join(paths.analysis_path,'_'.join((animal, 'cellMatch.hdf5')))
    else:
        return os.path.join(config["target_path"], config["files"][wildcards.file] + '.avi')


rule preprocess:
    input:
          dlc_input_selector,
          calcium_input_selector,
          match_selector,
    output:
          os.path.join(paths.analysis_path, "{file}_rawcoord.hdf5"),
          os.path.join(paths.analysis_path, "{file}.png")
    params:
          info=lambda wildcards: config["file_info"][wildcards.file]
    script:
          "snakemake_scripts/preprocess_all.py"


rule motifs:
    input: 
        lambda wildcards: os.path.join(paths.analysis_path, config["files"][wildcards.file] + '_rawcoord.hdf5'),
    output:
        os.path.join(paths.analysis_path, "{file}_motifs.hdf5"),
    params:
          info=yaml_to_json,
    shell:
        r'conda activate vame & python "{paths.vame_latents}" "{input}" "{output}" "{params.info}"'


def motif_selector(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)

    if python_dict['rig'] == 'miniscope':
        return os.path.join(paths.analysis_path, "{file}_motifs.hdf5"),
    else:
        return python_dict['avi_path']


rule preprocess_compile:
    input:
        lambda wildcards: os.path.join(paths.analysis_path, config["files"][wildcards.file] + '_rawcoord.hdf5'),
        motif_selector,
    output:
        os.path.join(paths.analysis_path, "{file}_preproc.hdf5"),
    params:
        info=lambda wildcards: config["file_info"][wildcards.file]
    script:
        "snakemake_scripts/combine_preprocessing.py"


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


def files_to_day(wildcards):
    name_parts = wildcards.file.split('_')
    day = datetime.datetime.strptime('_'.join(name_parts[0:3]), '%m_%d_%Y').strftime('%Y-%m-%d')
    animal = '_'.join([name_parts[3].upper()] + name_parts[4:6])
    info_list = [yaml.load(config["file_info"][el], Loader=yaml.FullLoader) for el in config["file_info"]]

    day_routes = [el['avi_path'].replace('.avi', '_preproc.hdf5').replace('VideoExperiment', 'AnalyzedData')
                  for el in info_list if (config['calcium_flag'][os.path.basename(el['avi_path'])[:-4]]
                  and el['mouse']==animal and el['date'][:10]==day)]

    wildcards.day_routes = day_routes
    return day_routes


def days_to_file(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
    animal = python_dict['mouse']
    day = datetime.datetime.strptime(python_dict['date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%m_%d_%Y')
    rig = python_dict['rig']
    return os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'regressionday.hdf5')))


def days_to_file_neuron_drop(wildcards):
    python_dict = yaml.load(config["file_info"][wildcards.file], Loader=yaml.FullLoader)
    animal = python_dict['mouse']
    day = datetime.datetime.strptime(python_dict['date'], '%Y-%m-%dT%H:%M:%SZ').strftime('%m_%d_%Y')
    rig = python_dict['rig']
    return os.path.join(paths.analysis_path, '_'.join((day, animal, rig, 'neurondropday.hdf5')))


def all_days(wildcards):
    # initialize the keyword list
    keyword_list = []

    # run through all files in the config
    for files in config["file_info"].keys():
        # get the day, rig and animal
        name_parts = files.split('_')
        day = '_'.join(name_parts[0:3])
        rig = name_parts[6]
        animal = '_'.join(name_parts[7:10])
        keyword_list.append([day, animal, rig])

    # ID the unique combinations
    unique_patterns = np.unique(np.array(keyword_list), axis=0)
    # generate the corresponding file names and output
    path_list = [os.path.join(paths.analysis_path, '_'.join((el[0], el[1], el[2], 'tcday.hdf5')))
                 for el in unique_patterns]

    return path_list


rule gather_regression:
    input:
        files_to_day,
    output:
        os.path.join(paths.analysis_path,'{file}_regressionday.hdf5'),
    params:
        file_info = config["file_info"],
    script:
        "snakemake_scripts/classify_batch.py"


rule gather_neuron_drop:
    input:
        files_to_day,
    output:
        os.path.join(paths.analysis_path,'{file}_neurondropday.hdf5'),
    params:
        file_info = config["file_info"],
    script:
        "snakemake_scripts/neuron_drop.py"


rule scatter_analysis:
    input:
        days_to_file,
        # days_to_file_neuron_drop,
        # animal_to_file,
    output:
        os.path.join(paths.analysis_path,"{file}_combinedanalysis.hdf5"),
    params:
        file_info = lambda wildcards: config["file_info"][wildcards.file],
    script:
        "snakemake_scripts/combine_analyses.py"

rule tc_day:
    input:
        files_to_day,
    output:
        os.path.join(paths.analysis_path,'{file}_tcday.hdf5'),
    params:
        file_info = config["file_info"],
    script:
        "snakemake_scripts/tc_calculate.py"

rule tc_consolidate:
    input:
        # days_to_file,
        # animal_to_file,
        # expand(os.path.join(paths.analysis_path,"{file}_tcday.hdf5"), file=config['files'])
        all_days,
    output:
        os.path.join(paths.analysis_path,"{file}_tcconsolidate.hdf5"),
    script:
        "snakemake_scripts/tc_consolidate.py"


rule beh_consolidate:
    input:
        # days_to_file,
        # animal_to_file,
        # expand(os.path.join(paths.analysis_path,"{file}_tcday.hdf5"), file=config['files'])
        expand(os.path.join(paths.analysis_path,"{file}_preproc.hdf5"),file=config['files'])
    output:
        os.path.join(paths.analysis_path,"{file}_behconsolidate.hdf5"),
    params:
        info=expand("{info}", info=config["file_info"].values()),
    script:
        "snakemake_scripts/beh_consolidate.py"

# rule visualize_aggregates:
#     input:
#         os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
#     output:
#         os.path.join(paths.figures_path, "averages_{query}.html")
#     notebook:
#         "snakemake_scripts/notebooks/Vis_averages.ipynb"


# def run_selector(wildcards):
#     """Define which processing stream goes"""
#     if config['analysis_type'] == 'combinedanalysis':
#         return expand(os.path.join(paths.analysis_path,"{file}_combinedanalysis.hdf5"),file=config['files'])
#     elif config['analysis_type'] == 'full_run':
#         return expand(os.path.join(paths.analysis_path,"{file}"+processing_parameters.full_run_file),file=config['files'])

rule combinedanalysis_run:
    input:
          expand(os.path.join(paths.analysis_path,"{file}"+'_combinedanalysis.hdf5'), file=config['files']),
    output:
          os.path.join(paths.analysis_path, "combinedanalysis_run.txt")
    params:
          file_info=expand("{info}", info=config["file_info"].values()),
          output_info=config["output_info"]
    script:
          "snakemake_scripts/full_run.py"


rule full_run:
    input:
          expand(os.path.join(paths.analysis_path,"{file}"+'_preproc.hdf5'), file=config['files']),
    output:
          os.path.join(paths.analysis_path, "preprocessing_run.txt")
    params:
          file_info=expand("{info}", info=config["file_info"].values()),
          output_info=config["output_info"]
    script:
          "snakemake_scripts/full_run.py"