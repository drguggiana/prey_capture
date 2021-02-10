# Set parameters for source extraction

min_pnr = 6
min_corr = 0.8

min_SNR = 1.5                                  # minimum SNR for accepting new components
rval_thr = 0.85                                # correlation threshold for new component inclusion
merge_thr = 0.65                               # merging threshold

rf = 48  # half size of patch (used only during initialization)
stride = 8  # overlap between patches (used only during initialization)
ssub = 1  # spatial downsampling factor (during initialization)
ds_factor = 4 * ssub  # spatial downsampling factor (during online processing)
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
fr = 10  # frame rate

online_dict = {'epochs': epochs,
               'nb': 0,
               'ssub': ssub,
               'ssub_B': ssub_B,
               'ds_factor': ds_factor,  # ds_factor >= ssub should hold
               'gSig': gSig,
               'gSiz': gSiz,
               'gSig_filt': (20, 20),
               'min_corr': min_corr,
               'bas_nonneg': False,
               'center_psf': True,
               'ring_size_factor': 1.5,
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
               # 'fnames': fnames,
               'decay_time': decay_time,
               'n_pixels_per_process': 128,
               'show_movie': show_movie,
               'fr': fr,
               }
