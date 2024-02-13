import numpy as np
import os
import sys
import json
import h5py
import yaml
from skimage.io import imread, imsave
from dask.distributed import Client, LocalCluster
from pathlib import Path

# Insert the cwd for local imports
os.chdir(os.getcwd())
sys.path.insert(0, os.getcwd())

import paths
import processing_parameters
import functions_bondjango as bd
import functions_io as fi
import functions_denoise_calcium as fdn

if __name__ == "__main__":

    try:
        # get the target video path
        video_path = sys.argv[1]

        # read the output path and the input file urls
        out_path = sys.argv[2]
        data_all = sys.argv[3]

        # get the parts for the file naming
        name_parts = os.path.basename(video_path).split('_')
        day = '_'.join(name_parts[0:3])
        rig = name_parts[6]
        animal = '_'.join([name_parts[7].upper()] + name_parts[8:10])

        # get the ca raw file
        ca_raw_path = os.path.join(paths.analysis_path, '_'.join((Path(video_path).stem, 'calciumraw')) + '.hdf5')

    except IndexError:
        # get the search string
        # search_string = processing_parameters.search_string
        animal = processing_parameters.animal
        day = processing_parameters.day
        rig = processing_parameters.rig
        search_string = f'rig:{rig}, imaging:wirefree, mouse:{animal}, slug:{day}'

        # query the database for data to plot
        data_all = bd.query_database('vr_experiment', search_string)
        # video_data = data_all[0]
        video_path = data_all[0]['tif_path']
        # overwrite data_all with just the urls
        data_all = {os.path.basename(el['tif_path'])[:-4]: el['url'] for el in data_all}
        # get ca raw path
        search_string = f'rig:{rig}, imaging:wirefree, mouse:{animal}, slug:{day}'
        data_all = bd.query_database('analyzed_data', search_string)

        path_template = [data['analysis_path'] for data in data_all if ('_preproc' in data['analysis_path']) and
                         (animal in data['analysis_path'])][0]
        ca_raw_path = path_template.replace('_preproc', '_calciumraw')
        out_path = ca_raw_path.replace('calciumraw.hdf5', 'update_ca_reg.txt')

    # Here is some stupidity to deal with how MiniAn expects the directory to be formatted
    save_path = os.path.join(paths.temp_path, animal, rig)
    # Handle file renaming for denoised file and save the tif in the modified temp path
    out_path_tif = os.path.join(save_path, os.path.basename(video_path).replace('.tif', '_denoised.tif'))

    if (os.path.isdir(save_path)) and (os.path.isfile(out_path_tif)):
        print('Already denoised')
    else:
        # delete the folder contents
        fi.delete_contents(paths.temp_minian)
        fi.delete_contents(paths.temp_path)
        os.makedirs(save_path)
        # denoise the video
        stack = fdn.denoise_stack(video_path)
        # Save the denoised stack
        imsave(out_path_tif, stack, plugin="tifffile", bigtiff=True)
        del stack

    if (os.path.isdir(save_path)) and (os.path.isdir(os.path.join(save_path, 'minian\motion.zarr'))):
        print('Motion already calculated, skipping to updating the calciumraw file')
        from minian.utilities import open_minian

    else:
        # Run minian
        print("starting minian")
        # Set up Initial Basic Parameters
        minian_path = paths.minian_path

        # Here is some stupidity to deal with how Minian expects data to be formatted
        dpath = save_path
        minian_ds_path = os.path.join(dpath, "minian")
        intpath = paths.temp_minian
        n_workers = int(os.getenv("MINIAN_NWORKERS", 4))

        # Load the yaml file containing the relevant parameters
        with open(paths.minian_parameters, 'r') as f:
            # Load the contents of the file into a dictionary
            params = yaml.unsafe_load(f)
            # Get the parameters based on the animal and rig
            animal_rig_params = params.get(animal).get(rig)
            # Get the default param set
            default_params = params.get('default')

        param_save_minian = {
            "dpath": minian_ds_path,
            "meta_dict": dict(session=-1, animal=-2),
            "overwrite": True,
        }

        # Pre-processing Parameters#
        subset = None  # This is overwritten later
        param_load_videos = {
            "pattern": ".tif$",
            "dtype": np.uint8,
            "downsample": dict(frame=1, height=1, width=1),
            "downsample_strategy": "subset",
        }

        # Denoising and background removal #
        param_denoise = animal_rig_params.get('param_denoise', default_params['param_denoise'])
        param_background_removal = {"method": "tophat", "wnd": 15}

        # Motion Correction Parameters #
        subset_mc = None
        param_estimate_motion = {"dim": "frame", "npart": 5, "aggregation": "mean"}

        # Initialization Parameters #
        param_seeds_init = {
            "wnd_size": 800,
            "method": "rolling",
            "stp_size": 400,
            "max_wnd": 20,
            "diff_thres": 3,
        }

        # Load the noise_freq from the yaml, but have a default as well
        noise_freq = animal_rig_params.get('noise_freq', default_params['noise_freq'])

        param_pnr_refine = {"noise_freq": noise_freq, "thres": 1.05}
        param_ks_refine = {"sig": 0.05}
        param_seeds_merge = {"thres_dist": 3, "thres_corr": 0.85, "noise_freq": noise_freq}
        param_initialize = {"thres_corr": 0.85, "wnd": 15, "noise_freq": noise_freq}
        param_init_merge = {"thres_corr": 0.85}

        # CNMF Parameters #
        param_get_noise = {"noise_range": (noise_freq, 0.5)}

        param_first_spatial = animal_rig_params.get('param_first_spatial', default_params['param_first_spatial'])
        param_first_temporal = animal_rig_params.get('param_first_temporal', default_params['param_first_temporal'])
        param_first_merge = {"thres_corr": 0.85}

        param_second_spatial = animal_rig_params.get('param_second_spatial', default_params['param_second_spatial'])
        param_second_temporal = animal_rig_params.get('param_second_temporal', default_params['param_second_temporal'])

        # Environment variables #
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MINIAN_INTERMEDIATE"] = intpath

        # import the relevant minian functions
        sys.path.append(minian_path)

        from minian.motion_correction import apply_transform, estimate_motion
        from minian.preprocessing import denoise, remove_background
        from minian.utilities import (
            TaskAnnotation,
            get_optimal_chk,
            load_videos,
            save_minian,
            rechunk_like,
        )

        # get the path
        dpath = os.path.abspath(dpath)

        # start the cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            memory_limit="15GB",
            resources={"MEM": 1},
            threads_per_worker=2,
            dashboard_address=":8789",
        )

        annt_plugin = TaskAnnotation()
        cluster.scheduler.add_plugin(annt_plugin)
        client = Client(cluster)

        # load the videos and the chk parameter
        varr = load_videos(dpath, **param_load_videos)
        chk, _ = get_optimal_chk(varr, dtype=float)
        print(f'Current chunk size: {chk}')

        # intermediate save
        varr = save_minian(varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"), intpath,
                           overwrite=True)

        # subset so that we exclude the first n frames and the last frame
        last_idx = int(varr.frame[-2].values)
        varr_subset = varr.sel(frame=slice(400, last_idx))
        varr_ref = varr.sel(frame=slice(400, last_idx))

        # BACKGROUND CORRECTION
        print("background correction")

        # remove glow
        varr_min = varr_ref.chunk({"frame": -1, "height": -1, "width": -1}).quantile(0.0025,
                                                                                     dim="frame", skipna=True).compute()
        varr_min = varr_min - np.min(varr_min)
        varr_ref = varr_ref - varr_min
        varr_ref = varr_ref.where(varr_ref > 0, 0).astype(np.uint8)

        # Rechunk
        # chk, _ = get_optimal_chk(varr_ref, dtype=float)
        # varr_ref = varr_ref.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr_ref")
        varr_ref = rechunk_like(varr_ref, varr)

        # denoise
        varr_ref = denoise(varr_ref, **param_denoise)

        # remove background
        varr_ref = remove_background(varr_ref, **param_background_removal)

        # save background corrected
        varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

        # MOTION CORRECTION
        print("motion correction")
        # estimate motion
        motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)

        # save motion estimate
        motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian)

        # apply motion correction
        Y = apply_transform(varr_subset, motion, fill=0)

        # Close down MiniAn
        client.close()
        cluster.close()

        # save the denoised and registered stack
        reg_stack = Y.compute().astype(np.uint8)
        imsave(video_path.replace('.tif', '_registered.tif'), reg_stack, plugin="tifffile", bigtiff=True)

    # Update the calciumraw file
    df = open_minian(os.path.join(save_path, 'minian'))
    motion = df['motion']
    with h5py.File(ca_raw_path, 'a') as f:
        # save the motion data
        if 'motion' not in f.keys():
            f.create_dataset('motion', data=np.array(motion))

    # create the temp output file
    with open(out_path, 'w') as f:
        f.write('done')
