import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore',
                        message="param.Dimension: Use method 'get_param_values' via param namespace")
import caiman as cm
from caiman.source_extraction import cnmf as cnmf


def cnmfe_function(fnames, save_path):

    # We start with motion correction and then proceed with the source extraction using the CNMF-E algorithm.
    # For a detailed 1p demo check `demo_pipeline_cnmfE.ipynb`.

    # start a cluster for parallel processing
    # (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # Set parameters for source extraction

    min_pnr = 6
    min_corr = 0.8

    min_SNR = 1.5                                  # minimum SNR for accepting new components
    rval_thr = 0.85                                # correlation threshold for new component inclusion
    merge_thr = 0.65                               # merging threshold

    rf = 48  # half size of patch (used only during initialization)
    stride = 8  # overlap between patches (used only during initialization)
    ssub = 1  # spatial downsampling factor (during initialization)
    ds_factor = 2 * ssub  # spatial downsampling factor (during online processing)
    ssub_B = 4  # background downsampling factor (use that for faster processing)
    gSig = (10 // ds_factor, 10 // ds_factor)  # expected half size of neurons
    gSiz = (30 // ds_factor, 30 // ds_factor)
    sniper_mode = False  # flag using a CNN to detect new neurons (o/w space correlation is used)
    init_batch = 150  # number of frames for initialization (presumably from the first file)
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
                   'ring_size_factor': 1.5,
                   'only_init_patch': True,
                   'gnb': 0,
                   'max_shifts_online': 20,
                   'rval_thr': rval_thr,
                   'merge_thr': merge_thr,
                   'min_SNR': min_SNR,
                   'min_pnr': min_pnr,
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
                   'n_pixels_per_process': 128,
                   'show_movie': show_movie}

    online_opts = cnmf.params.CNMFParams(params_dict=online_dict)

    # create the cnmf online element
    cnmf_online = cnmf.online_cnmf.OnACID(params=online_opts, dview=dview)

    # try, if no ROIs are found, skip
    try:
        # fit the data
        cnmf_online.fit_online()
    except ValueError:
        print(f'no ROIs found in file {fnames}')
    # kill the server
    if 'dview' in locals():
        cm.stop_server(dview=dview)

    # transfer only estimates and params to a dummy variable
    dummy = cnmf.online_cnmf.OnACID(params=online_opts)
    dummy.estimates = cnmf_online.estimates
    # save the output
    dummy.save(save_path)

    return
