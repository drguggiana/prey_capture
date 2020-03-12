import snakemake_scripts.sub_preprocess_S1 as sub
import snakemake_scripts.sub_preprocess_S2 as sub2


# run the first stage of preprocessing
out_path, filtered_traces = sub.run_preprocess(snakemake.input[0], snakemake.output[0],
                                                         ['cricket_x', 'cricket_y'])
# TODO: add corner detection to calibrate the coordinate to real size
# in the meantime, add a rough manual correction based on the size of the arena and the number of pixels

# run the preprocessing kinematic calculations
kinematics_data = sub2.kinematic_calculations(out_path, filtered_traces)
