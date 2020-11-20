import numpy as np
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.motion_correction import MotionCorrect


def main():
    # fnames = [r"J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\12_07_2019_15_06_28_miniscope_MM_191108_a_succ.tif"]
    fnames = [r"C:\Users\drguggiana\caiman_data\example_movies\09_08_2020_15_26_21_miniscope_DG_200701_a_succ.tif"]
    # Batch (offline) approach

    # We start with motion correction and then proceed with the source extraction using the CNMF-E algorithm.
    # For a detailed 1p demo check `demo_pipeline_cnmfE.ipynb`.

    # # motion correction parameters
    # motion_correct = True            # flag for performing motion correction
    # pw_rigid = False                 # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    # gSig_filt = (7, 7)               # size of high pass spatial filtering, used in 1p data
    # max_shifts = (20, 20)            # maximum allowed rigid shift
    # border_nan = 'copy'              # replicate values along the boundaries
    #
    # mc_dict = {
    #     'pw_rigid': pw_rigid,
    #     'max_shifts': max_shifts,
    #     'gSig_filt': gSig_filt,
    #     'border_nan': border_nan
    # }
    #
    # online_opts = cnmf.params.CNMFParams(params_dict=mc_dict)
    #
    # start a cluster for parallel processing
    # (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    #
    # mc = MotionCorrect(fnames, dview=dview, **online_opts.get_group('motion'))
    # mc.motion_correct(save_movie=True)
    #
    # # We then proceed with memory mapping
    #
    # from time import time
    # fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
    #                            border_to_0=0, dview=dview)
    # Yr, dims, T = cm.load_memmap(fname_new)
    # images = Yr.T.reshape((T,) + dims, order='F')

    # Set parameters for source extraction

    # min_pnr = 6
    min_corr = 0.8
    # rf = 48                                        # half size of each patch
    # stride = 8                                     # amount of overlap between patches
    # ssub = 1                                       # spatial downsampling factor
    # decay_time = 1                               # length of typical transient (in seconds)
    # fr = 10                                        # imaging rate (Hz)
    # gSig = (10, 10)                                  # expected half size of neurons
    # gSiz = (30, 30)                                # half size for neuron bounding box
    # p = 0                                          # order of AR indicator dynamics
    # min_SNR = 1.5                                  # minimum SNR for accepting new components
    rval_thr = 0.85                                # correlation threshold for new component inclusion
    # merge_thr = 0.65                               # merging threshold
    # K = None                                       # initial number of components
    #
    # cnmfe_dict = {'fnames': fnames,
    #               'fr': fr,
    #               'decay_time': decay_time,
    #               'method_init': 'corr_pnr',
    #               'gSig': gSig,
    #               'gSiz': gSiz,
    #               'rf': rf,
    #               'stride': stride,
    #               'p': p,
    #               'nb': 0,
    #               'ssub': ssub,
    #               'min_SNR': min_SNR,
    #               'min_pnr': min_pnr,
    #               'min_corr': min_corr,
    #               'bas_nonneg': False,
    #               'center_psf': True,
    #               'rval_thr': rval_thr,
    #               'only_init': True,
    #               'merge_thr': merge_thr,
    #               'K': K}
    # # online_opts.change_params(cnmfe_dict)
    # online_opts = cnmf.params.CNMFParams(params_dict=cnmfe_dict)

    # print(online_opts)

    rf = 48  # half size of patch (used only during initialization)
    stride = 8  # overlap between patches (used only during initialization)
    ssub = 1  # spatial downsampling factor (during initialization)
    ds_factor = 2 * ssub  # spatial downsampling factor (during online processing)
    ssub_B = 4  # background downsampling factor (use that for faster processing)
    gSig = (10 // ds_factor, 10 // ds_factor)  # expected half size of neurons
    gSiz = (30 // ds_factor, 30 // ds_factor)
    sniper_mode = False  # flag using a CNN to detect new neurons (o/w space correlation is used)
    init_batch = 200  # number of frames for initialization (presumably from the first file)
    expected_comps = 500  # maximum number of expected components used for memory pre-allocation (exaggerate here)
    dist_shape_update = False  # flag for updating shapes in a distributed way
    min_num_trial = 5  # number of candidate components per frame
    K = None  # initial number of components
    epochs = 2  # number of passes over the data
    show_movie = False  # show the movie with the results as the data gets processed
    use_corr_img = True  # flag for using the corr*pnr image when searching for new neurons (otherwise residual)
    decay_time = 1

    online_dict = {'epochs': epochs,
                   'nb': 0,
                   'ssub': ssub,
                   'ssub_B': ssub_B,
                   'ds_factor': ds_factor,  # ds_factor >= ssub should hold
                   'gSig': gSig,
                   'gSiz': gSiz,
                   'gSig_filt': (3, 3),
                   'min_corr': min_corr,
                   'bas_nonneg': False,
                   'center_psf': True,
                   'max_shifts_online': 20,
                   'rval_thr': rval_thr,
                   'motion_correct': True,
                   'init_batch': init_batch,
                   'only_init': True,
                   # 'init_method': 'cnmf',
                   'method_init': 'corr_pnr',
                   'normalize_init': False,
                   'update_freq': 200,
                   'expected_comps': expected_comps,
                   'sniper_mode': sniper_mode,  # set to False for 1p data
                   'dist_shape_update': dist_shape_update,
                   'min_num_trial': min_num_trial,
                   'use_corr_img': use_corr_img,
                   'fnames': fnames,
                   'decay_time': decay_time,
                   'show_movie': show_movie}
    # online_opts.change_params(online_dict)
    online_opts = cnmf.params.CNMFParams(params_dict=online_dict)

    print(online_opts)

    cnm_online = cnmf.online_cnmf.OnACID(params=online_opts, dview=dview)
    cnm_online.fit_online()

    # if online_opts.online['motion_correct']:
    #     shifts = cnm_online.estimates.shifts[-cnm_online.estimates.C.shape[-1]:]
    #     if not online_opts.motion['pw_rigid']:
    #         memmap_file = cm.motion_correction.apply_shift_online(images, shifts,
    #                                                               save_base_name='MC')
    #     else:
    #         mc = MotionCorrect(fnames, dview=dview, **online_opts.get_group('motion'))
    #
    #         mc.y_shifts_els = [[sx[0] for sx in sh] for sh in shifts]
    #         mc.x_shifts_els = [[sx[1] for sx in sh] for sh in shifts]
    #         memmap_file = mc.apply_shifts_movie(fnames, rigid_shifts=False,
    #                                             save_memmap=True,
    #                                             save_base_name='MC')
    # else:  # To do: apply non-rigid shifts on the fly
    #     memmap_file = images.save(fnames[0][:-4] + 'mmap')
    # cnm_online.mmap_file = memmap_file
    # Yr_online, dims, T = cm.load_memmap(memmap_file)
    #
    # # cnm_online.estimates.dview=dview
    # # cnm_online.estimates.compute_residuals(Yr=Yr_online)
    # images_online = np.reshape(Yr_online.T, [T] + list(dims), order='F')
    # min_SNR = 2  # peak SNR for accepted components (if above this, acept)
    # rval_thr = 0.85  # space correlation threshold (if above this, accept)
    # use_cnn = False  # use the CNN classifier
    # cnm_online.params.change_params({'min_SNR': min_SNR,
    #                                  'rval_thr': rval_thr,
    #                                  'use_cnn': use_cnn})
    #
    # # cnm_online.estimates.evaluate_components(images_online, cnm_online.params, dview=dview)
    # # cnm_online.estimates.Cn = pnr

    print(cnm_online.estimates)
    print(cnm_online.estimates.shape)

    # get the spatial and temporal components


if __name__ == "__main__":
    main()
