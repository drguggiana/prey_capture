configfile: "snakemake_scripts/config_snake.yaml"
import os
import paths


rule preprocess:
    input:
          lambda wildcards: os.path.join(config["target_path"], config["files"][wildcards.file] + '.csv'),
    output:
          os.path.join(paths.analysis_path, "{file}_preproc.hdf5"),
          os.path.join(paths.analysis_path, "{file}.png")
    params:
          info=lambda wildcards: config["file_info"][wildcards.file]
    script:
          "snakemake_scripts/preprocess_all.py"


rule aggregate_preprocessed:
    input:
          expand(os.path.join(paths.analysis_path, "{file}_preproc.hdf5"), file=config['files'])
    output:
         os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
    params:
          file_info=expand("{info}", info=config["file_info"].values()),
          output_info=config["output_info"]
    script:
          "snakemake_scripts/aggregate.py"

# rule cluster_encounters:
#     input:
#     output:
#     params:
#     script:


rule visualize_aggregates:
    input:
        os.path.join(paths.analysis_path, "preprocessing_{query}.hdf5")
    output:
        os.path.join(paths.figures_path, "averages_{query}.html")
    notebook:
        "snakemake_scripts/notebooks/Vis_averages.ipynb"
