import os
import sys
import yaml
import numpy as np
from dask.distributed import Client, LocalCluster
import paths


def minian_main(rig, animal, override_dpath=None, subset_start_idx=400):

    # Set up Initial Basic Parameters#
    minian_path = paths.minian_path

    # Here is some stupidity to deal with how Minian expects data to be formatted
    if override_dpath is None:
        dpath = paths.temp_path
    else:
        dpath = override_dpath

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
    subset = None    # This is overwritten later
    param_load_videos = {
        "pattern": ".tif$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    }

    # Denoising and background removal #
    # param_denoise = {"method": "median", "ksize": 5}
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
    # Most of these are loaded from the minian_params_wirefree.yaml file, but have defaults set just in case
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
    from minian.cnmf import (
        compute_AtC,
        compute_trace,
        get_noise_fft,
        smooth_sig,
        unit_merge,
        update_spatial,
        update_temporal,
        update_background,
    )
    from minian.initialization import (
        gmm_refine,
        initA,
        initC,
        intensity_refine,
        ks_refine,
        pnr_refine,
        seeds_init,
        seeds_merge,
    )
    from minian.motion_correction import apply_transform, estimate_motion
    from minian.preprocessing import denoise, remove_background
    from minian.utilities import (
        TaskAnnotation,
        get_optimal_chk,
        load_videos,
        open_minian,
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
    varr_ref = varr.sel(frame=slice(subset_start_idx, last_idx))

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
    Y = apply_transform(varr_ref, motion, fill=0)

    # save motion corrected results
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True,
                           chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    # Added by MM, not part of minian pipeline
    # Get average frame fluorescence across the recording
    mean_frame_fluor = Y_hw_chk.mean(['height', 'width']).rename('mean_fluorescence')

    # CNMF Initialization
    print("CNMF Initialization")
    # max projection
    max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian).compute()

    # initialize seeds
    seeds = seeds_init(Y_fm_chk, **param_seeds_init)

    # pnr refinement
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)

    # Kolmogorov-Smirnoff refinement
    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

    # seed merging
    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)

    # initialize spatial matrix
    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

    # initialize temporal matrix
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})

    # merge units from initialization
    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})

    # initialize background terms
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    # CNMF

    # get noise for spatial update
    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    # spatial update 1
    print("spatial update 1")
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_first_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)

    # update background
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    # save spatial results
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # save raw signal for temporal update
    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath,
                      overwrite=True, chunks={"unit_id": 1, "frame": -1})

    # temporal update 1
    print("temporal update 1")
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_first_temporal)

    # save temporal results
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)

    A = A.sel(unit_id=C.coords["unit_id"].values)

    # merge units
    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)

    # save merged results
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # second spatial update
    print("spatial update 2")
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_second_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)

    # update background
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    # save second spatial results
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1})
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    # save second raw for temporal
    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath,
                      overwrite=True, chunks={"unit_id": 1, "frame": -1},)

    # second temporal update
    print("temporal update 2")
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)

    # save second temporal results
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # save final results for output
    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)

    # close cluster
    client.close()
    cluster.close()

    print("MiniAn processing finished!")

    return {'A': A, 'C': C, 'S': S, 'c0': c0, 'b0': b0, 'b': b, 'f': f,
            'max_proj': max_proj, 'mean_frame_fluor': mean_frame_fluor}
