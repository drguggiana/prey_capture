# define the input dictionary for run_snake
input_dictionary = {
    'analysis_type': ['tuning_run'],
    # 'analysis_type': ['preprocessing_run'],
    # 'analysis_type': ['combinedanalysis'],
    # 'analysis_type': ['tcconsolidate'],
    # 'analysis_type': ['behconsolidate'],
    # 'analysis_type': ['aggFullCA'],
    # 'analysis_type': ['trigAveCA'],
    # 'result': ['habi'],
    # 'rig': ['VWheelWF'],   # 'VTuningWF', 'VWheelWF'
    # 'lighting': ['normal', ],
    'mouse': ['MM_230518_b'],
    # 'MM_221110_a', 'MM_221109_a', 'MM_220928_a', 'MM_220915_a',
    # 'MM_230518_b', 'MM_230705_b', 'MM_230706_a', 'MM_230706_b'
    # 'gtdate': ['2023-08-15T00-00-00'],
    # 'ltdate': ['2023-08-14T00-00-00'],
    'slug': ['07_26_2023', '07_27_2023', '07_28_2023', '08_01_2023'],
    # 'notes': ['vrcrickets_3'],
}

# error files

# define the file to call during full run
# full_run_file = '_combinedanalysis.hdf5'
full_run_file = '_preproc.hdf5'    # use for preprocessing

# define the search string (preprocess_all, aggregation, scatter_calcium, run_cnmfe, pose_repair)
# search_string = 'mouse:MM_230518_b'
search_string = 'mouse:MM_230706_b, slug:08_11_2023'

# define a search string for consolidated files
# search_consolidate = 'slug:preprocessing_ALL_miniscope_ALL_ALL_ALL_ALL_ALL_ALL_ALL_ALL_tcconsolidate'
# search_consolidate = 'slug:preprocessing_ALL_miniscope_ALL_ALL_ALL_ALL_ALL_ALL_ALL_DG_200701_a_tcconsolidate'
search_consolidate = 'slug:tcconsolidate'

# search list for classify2
search_list = [
               # 'mouse:MM_230706_b, slug:08_11_2023, rig:VTuningWF, analysis_type:preprocessing',
               # 'mouse:DG_210202_a, slug:03_29_2021, rig:miniscope',
               # 'slug:DG_200701_a, rig:miniscope',  #gtdate:2020-08-10T00-00-00, ltdate:2020-08-11T00-00-00'
               # 'slug:DG_200701_a',
               # 'mouse:DG_210202_a, slug:03_24_2021, rig:miniscope',
               # 'mouse:DG_210202_a, slug:03_23_2021, rig:miniscope',
               # 'mouse:DG_210202_a, slug:03_30_2021, rig:miniscope',
               # 'mouse:DG_210202_a, slug:03_31_2021, rig:miniscope',
               # 'mouse:DG_210202_a, gtdate:2021-03-24T00-00-00, ltdate:2021-04-01T00-00-00, rig:miniscope'
               # 'slug:DG_210202_a, rig:miniscope',
               # 'slug:08_10_2023',
               'slug:MM_200129_a', 'slug:MM_200129_b', 'slug:DG_210202_a', 'slug:MM_191108_a', 'slug:MM_191105_a',
               'slug:MM_191107_a', 'slug:MM_191106_a', 'slug:DG_190806_a', 'slug:DG_190810_a', 'slug:DG_200701_a',
               # 'slug:03_23_2021_07_15_34_miniscope_DG_210202_a_habi',
               # 'mouse:DG_210323_b, slug:06_14_2021, rig:miniscope',
               ]
meta_fields = ['lighting']

# for aggregation
# analysis_type = 'aggFullCA'
analysis_type = 'aggEnc'

# --- for the calcium analysis --- #
# define the target animal and date
animal = 'MM_230706_a'
day = '01_20_2023'
rig = 'VWheelWF'

# define the search string
roi_parameters = {'area_min': 20,
                  'area_max': 300,
                 }

# define the DLC search string
dlc_string = 'slug:09_08_2020_15_26_21_miniscope_DG_200701_a_succ'

# define the string for the caiman parameters notebook
caiman_string = r'04_02_2021_10_34_47_miniscope_DG_210202_a_succ'

# define the string for the visualization notebook
# vis_string = 'slug:09_08_2020_15_26_21_miniscope_DG_200701_a_succ'
# vis_string = search_string
# vis_string = 'rig:miniscope'1
vis_string = 'slug:04_01_2021_09_35_49_miniscope_DG_210202_a_succ'
# vis_string = 'result:fail'

# define the string for vame visualization
# vame_vis_string = 'slug:09_08_2020_15_26_21_miniscope_DG_200701_a_succ'
vame_vis_string = 'rig:miniscope, analysis_type:preprocessing'
motif_sort = [4, 1, 7, 2, 6, 8, 9, 14, 13, 12, 5, 10, 11, 0, 3]
motif_revsort = [13, 1, 3, 14, 0, 10, 4, 2, 5, 6, 11, 12, 9, 8, 7]

mouse_motif_sort = [8, 2, 14, 12, 10, 1, 3, 13, 6, 4, 9, 5, 0, 11, 7]
mouse_motif_revsort = [12, 5, 1, 6, 9, 11, 8, 14, 0, 10, 4, 13, 3, 7, 2]

# define the test videos for DLC
test_videos = [
    r'J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\03_15_2021_10_35_50_miniscope_DG_210202_a_habi_nomini.avi',
    r'J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\08_08_2020_16_00_22_miniscope_DG_200617_b_succ.avi',
    r'J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\08_12_2020_15_44_23_miniscope_DG_200701_a_succ_noncon.avi',
    r'J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\11_11_2019_01_21_58_miniscope_DG_190806_a_fail_nofluo.avi',
    r'J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\04_02_2021_10_34_47_miniscope_DG_210202_a_succ.avi',
    r"J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\06_17_2021_10_39_40_miniscope_DG_210323_b_succ_dark_head.avi",
    r"J:\Drago Guggiana Nilo\Prey_capture\test_files\06_30_2021_18_07_01_miniscope_object_small.avi",
    r"J:\Drago Guggiana Nilo\Prey_capture\test_files\06_30_2021_18_09_49_miniscope_object_large.avi",
    r"J:\Drago Guggiana Nilo\Prey_capture\VideoExperiment\06_30_2021_17_19_57_miniscope_DG_210323_b_objt_nonres.avi"]

# label dictionary
label_dictionary = {
    'cricket_0_mouse_distance': 'Distance to prey (cm)',
    'cricket_0_delta_heading': 'Angle to prey (deg)',
    'mouse_x': 'Mouse horizontal position (cm)',
    'mouse_y': 'Mouse vertical position (cm)',
    'cricket_0_x': 'Prey horizontal position (cm)',
    'cricket_0_y': 'Prey vertical position (cm)',
    'cricket_0_speed': 'Prey speed (cm/s)',
    'mouse_speed': 'Mouse speed (cm/s)',
    'cricket_0_visual_angle': 'Prey visual angle (deg)',
}

wf_label_dictionary = {
    'VWheelWF': 'Head Fixed',
    'VTuningWF': 'Freely Moving',
    'direction': 'Direction (deg)',
    'direction_wrapped': 'Direction (deg)',
    'direction_rel_ground': 'Direction rel. Ground(deg)',
    'orientation_rel_ground': 'Orientation rel. Ground(deg)',
    'orientation': 'Orientation (deg)',
    'pupil_diameter': 'Pupil Diameter (px)',
    'wheel_speed': 'Wheel Speed (cm/s)',
    'wheel_speed_abs': 'Wheel Speed (cm/s)',
    'mouse_x_m': 'Mouse Horizontal Pos. (cm)',
    'mouse_y_m': 'Mouse Vertical Pos. (cm)',
    'head_direction': 'Head Direction (deg)',
    'head_height': 'Head Height (cm)',
    'mouse_angular_speed': 'Mouse Ang. Speed (deg/s)',
    'mouse_speed': 'Mouse Speed (cm/s)',
    'mouse_heading': 'Mouse Heading (deg)',
    'head_pitch': 'Head Pitch (deg)',
    'head_yaw': 'Head Yaw (deg)',
    'head_roll': 'Head Roll (deg)',
}

wf_vis_cols = ['Direction', 'Orientation', 'Dir.(still)', 'Ori. (still)']
wf_free_kinem_cols = ['Head Height', 'Head Pitch', 'Head Roll', 'Head Yaw', 'Head ang. sp.', 'Mouse sp.',
                      'Mouse X pos.', 'Mouse Y pos.']
wf_fixed_kinem_cols = ['Wheel sp.', 'Pupil Diam.']

# define the dictionary for result decoding
interpret_result = {
    'fail': 0,
    'succ': 1,
    'habi': -1,
}

# define the speed threshold for stopping
speed_threshold = 3  # in cm/s
angle_threshold = 90  # in deg
range_threshold = - 0.02  # in cm/s
approach_break_threshold = 250  # in ms
encounter_threshold = 3  # in cm

# define the variables to regress
bin_number = 10
variable_list_visual = ['direction', 'direction_wrapped', 'direction_rel_ground', 'orientation', 'orientation_rel_ground']
variable_list_free = ['mouse_x_m', 'mouse_y_m', 'head_height',
                      'head_pitch', 'head_yaw', 'head_roll',
                      'mouse_speed', 'mouse_angular_speed',  'head_direction', 'mouse_heading']
variable_list_fixed = ['wheel_speed_abs', 'pupil_diameter']
variable_list = ['cricket_0_mouse_distance', 'cricket_0_delta_heading', 'mouse_speed', 'mouse_x', 'cricket_0_x',
                 'cricket_0_visual_angle', 'hunt_trace']
# define the time shifts to regress to
time_shifts = [-20, -10, -5, 0, 5, 10, 20]
# define the number of regression shuffles
regression_shuffles = 5


# define the bins and variables for TC calculation
# tc_params = {
#     'mouse_x': [0, 40],
#     'mouse_y': [0, 40],
#     'cricket_0_x': [0, 40],
#     'cricket_0_y': [0, 40],
#     'cricket_0_mouse_distance': [0, 55],
#     'cricket_0_delta_heading': [-180, 180],
#     'mouse_speed': [0, 80],
#     'cricket_0_speed': [0, 40],
#     'cricket_0_visual_angle': [0, 360],
#     'hunt_trace': [0, 2],
# }
tc_params = {
    'mouse_x_m': [-60, 60],
    'mouse_y_m': [-40, 40],
    'mouse_speed': [0, 80],
    'wheel_speed': [-30, 30],
    'wheel_speed_abs': [0, 50],
    'mouse_angular_speed': [-500, 500],
    'head_pitch': [-180, 180],
    'head_yaw': [-180, 180],
    'head_roll': [-180, 180],
    'head_height': [0, 10],
    'head_direction': [-180, 180],
    'mouse_heading': [-180, 180],
    'pupil_diameter': [0, 150],
}

# --- WF calcium analysis variables --- #
wf_exp_types = ['fixed', 'free']
wf_frame_rate = 20
activity_datasets = ['norm_spikes_viewed', 'norm_spikes_viewed_still']
gof_type = 'rmse'
head_pitch_cutoff = (-90, 20)
view_fraction = 0.7
bootstrap_repeats = 1000

# define paths to use as templates for data generation (picked mostly arbitrarily)
template_paths = {
    'miniscope': {
        'calcium_path': r'06_18_2021_13_35_52_miniscope_DG_210323_b_succ_head_calcium.hdf5',
        'dlc_path': r'06_18_2021_13_35_52_miniscope_DG_210323_b_succ_head_dlc.h5'
    },
    'vtuning': {
        'calcium_path': r'06_18_2021_13_35_52_miniscope_DG_210323_b_succ_head_calcium.hdf5',
        'dlc_path': r'06_18_2021_13_35_52_miniscope_DG_210323_b_succ_head_dlc.h5'
    }
}

# define the mouse parameters for calcium extraction

calcium_mice = {
    'DG_200701_a': {
        'min_pnr': 8,
    },
    'DG_200617_b': {},
    'DG_210202_a': {},
    'MM_191108_a': {},
    'MM_200129_a': {},
    'MM_200129_b': {},
}


# default
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
               'method_init': 'corr_pnr',
               'normalize_init': False,
               'update_freq': 200,
               'expected_comps': expected_comps,
               'sniper_mode': sniper_mode,  # set to False for 1p data
               'dist_shape_update': dist_shape_update,
               'min_num_trial': min_num_trial,
               'use_corr_img': use_corr_img,
               'decay_time': decay_time,
               'n_pixels_per_process': 128,
               'show_movie': show_movie,
               'fr': fr,
               }

# initialize the full parameters dictionary
mouse_parameters = {
    'default':      online_dict,
}

# for all the mice
for mouse_name, mouse_dict in calcium_mice.items():
    # initialize the entry in the dictionary with the default
    mouse_parameters[mouse_name] = online_dict.copy()

    # for all the parameters
    for key, value in online_dict.items():
        # replace the specifics
        if key in mouse_dict:
            mouse_parameters[mouse_name][key] = mouse_dict[key]
