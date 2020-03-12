configfile: "snakemake_scripts/config_snake.yaml"
import os
import paths


rule preprocess:
    input:
         lambda wildcards: os.path.join(config["target_path"], config["files"][wildcards.file] + '.csv')
    output:
          os.path.join(paths.analysis_path, "{file}_preproc.hdf5")
    script:
          "snakemake_scripts/preprocess_all.py"


rule aggregate_preprocessed:
    input:
          expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files'])
         # expand(lambda wildcards: config['files'][wildcards.file].replace('.csv','_preproc.hdf5'), file=config["files"])
         # expand( ,file=config["files"])
    output:
         os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
    script:
          "snakemake_scripts/aggregate.py"
