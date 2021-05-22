
import itertools as itt
import os
import sys
import paths
import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
# from holoviews.operation.datashader import datashade, regrid
# from holoviews.util import Dynamic


def minian_main():

    # Set up Initial Basic Parameters#
    minian_path = paths.minian_path
    dpath = paths.temp_path
    minian_ds_path = os.path.join(dpath, "minian")
    intpath = paths.temp_minian
    subset = dict(frame=slice(0, None))
    subset_mc = None
    interactive = True
    output_size = 100
    n_workers = int(os.getenv("MINIAN_NWORKERS", 4))
    param_save_minian = {
        "dpath": minian_ds_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }

    # Pre-processing Parameters#
    param_load_videos = {
        "pattern": ".tif$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    }
    param_denoise = {"method": "median", "ksize": 9}
    param_background_removal = {"method": "tophat", "wnd": 15}

    # Motion Correction Parameters#
    subset_mc = None
    param_estimate_motion = {
        "dim": "frame",
        "npart": 2}

    # Initialization Parameters#
    param_seeds_init = {
        "wnd_size": 100,
        "method": "rolling",
        "stp_size": 50,
        "max_wnd": 25,
        "diff_thres": 4,
    }
    param_pnr_refine = {"noise_freq": 0.06, "thres": 1}
    param_ks_refine = {"sig": 0.05}
    param_seeds_merge = {"thres_dist": 5, "thres_corr": 0.8, "noise_freq": 0.06}
    param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
    param_init_merge = {"thres_corr": 0.8}

    # CNMF Parameters#
    param_get_noise = {"noise_range": (0.06, 0.5)}
    param_first_spatial = {
        "dl_wnd": 5,
        "sparse_penal": 0.01,
        "update_background": True,
        "size_thres": (25, None),
    }
    param_first_temporal = {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.2,
    }
    param_first_merge = {"thres_corr": 0.8}
    param_second_spatial = {
        "dl_wnd": 5,
        "sparse_penal": 0.01,
        "update_background": True,
        "size_thres": (25, None),
    }
    param_second_temporal = {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
    }

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
    )
    from minian.initialization import (
        gmm_refine,
        initA,
        initbf,
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
    )
    from minian.visualization import (
        # CNMFViewer,
        # VArrayViewer,
        # generate_videos,
        # visualize_gmm_fit,
        # visualize_motion,
        # visualize_preprocess,
        # visualize_seeds,
        # visualize_spatial_update,
        # visualize_temporal_update,
        write_video,
    )

    # get the path
    dpath = os.path.abspath(dpath)

    # start the cluster
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit="10GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=":8787",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # load the videos and the chk parameter
    varr = load_videos(dpath, **param_load_videos)
    chk, _ = get_optimal_chk(varr, dtype=float)
    print(chk)

    # intermediate save
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    # select a subset if defined
    varr_ref = varr.sel(subset)

    # BACKGROUND CORRECTION

    # remove glow
    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min

    # denoise
    varr_ref = denoise(varr_ref, **param_denoise)

    # remove background
    varr_ref = remove_background(varr_ref, **param_background_removal)

    # save background corrected
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

    # MOTION CORRECTION

    # estimate motion
    motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)

    # save motion estimate
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian
    )

    # apply motion correction
    Y = apply_transform(varr_ref, motion, fill=0)

    # save motion corrected results
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )

    # OPTIONAL generate motion corrected video
    vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
    write_video(vid_arr, "minian_mc.mp4", dpath)

    # CNMF Initialization

    # max projection
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
    ).compute()

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
    C_init = save_minian(
        C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1}
    )

    # merge units from initialization
    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )

    # initialize background terms
    b, f = initbf(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    # CNMF

    # get noise for spatial update
    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    # spatial update 1
    A_new, b_new, f_new, mask = update_spatial(
        Y_hw_chk, A, b, C, f, sn_spatial, **param_first_spatial
    )

    # save spatial results
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = C.sel(unit_id=A.coords["unit_id"].values)
    C_chk = C_chk.sel(unit_id=A.coords["unit_id"].values)

    # save raw signal for temporal update
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )

    # temporal update
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param_first_temporal
    )

    # save temporal results
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # merge units
    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)

    # save merged results
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # second spatial update
    A_new, b_new, f_new, mask = update_spatial(
        Y_hw_chk, A, b, sig, f, sn_spatial, **param_second_spatial
    )

    # save second spatial results
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = C.sel(unit_id=A.coords["unit_id"].values)
    C_chk = C_chk.sel(unit_id=A.coords["unit_id"].values)

    # save second raw for temporal
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )

    # second temporal update
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param_second_temporal
    )

    # save second temporal results
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # save final results
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

    return {'A': A, 'C': C, 'S': S, 'c0': c0, 'b0': b0, 'b': b, 'f': f}
