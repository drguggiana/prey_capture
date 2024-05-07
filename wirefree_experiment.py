import numpy as np
import pandas as pd
import xarray as xr
from os.path import join
from functions_bondjango import query_database

from ast import literal_eval
from functions_kinematic import wrap, wrap_negative, smooth_trace
from functions_tuning import normalize
import paths
import processing_parameters


class DataContainer():
    def list_attributes(self):
        return list(self.__dict__.keys())
    
    def load_from_dict(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

class Metadata(DataContainer):
    def __init__(self, dictionary):
        for k, v in dictionary.items():           
            setattr(self, k, v)


class Cell(DataContainer):
    def __init__(self, name):
        self.id = name

    def _add_attribute_container(self, name):
        setattr(self, name, DataContainer())


class WirefreeExperiment(DataContainer):
    def __init__(self, exp_info, preproc_info=None, tc_info=None):

        # Experiment attributes
        self.metadata = Metadata(exp_info)
        if preproc_info is not None:
            self.preproc_info = Metadata(preproc_info)
        if tc_info is not None:
            self.tc_info = Metadata(tc_info)

        # Experimental metadata
        self.exp_params = None
        self.cell_matches = None
        self.arena_corners = None
        
        self.roi_info = None
        self.trial_set = None
        self.full_kinematics = None

        # Neural and kinematic data structures
        self.kinematics = None
        self.inferred_spikes = None
        self.deconv_fluor = None

        # Tuning curve data structures
        self.visual_tcs = None
        self.self_motion_tcs = None


    def save_hdf(self, file, attributes):
        for attribute in attributes:
            df = getattr(self, attribute)
            df.to_hdf(file, attribute, mode='a', format='fixed')

    # def save_cell_properties(self, file, attributes):
    #     for attribute in attributes:
    #         df = self._cellprops_to_dataframe(attribute)
    #         df.to_hdf(file, attribute, mode='a', format='fixed')


    def _load_tc(self):
        if self.tc_info is not None:
            tc_file = self.tc_info.analysis_path
        else:
            raise AttributeError('No tuning curve info provided. Cannot load data.')
        
        self.visual_tcs = DataContainer()
        self.self_motion_tcs = DataContainer()

        vis_tc_dict = {}
        self_motion_tc_dict = {}

        with pd.HDFStore(tc_file, 'r') as tcf:

            if self.cell_matches is not None:
                self.cell_matches = self._parse_cell_matches(tcf['cell_matches'], self.tc_info.result)

            keys_to_exclude = ['cell_matches', 'counts', 'edges']
            keys_to_keep = [key for key in tcf.keys() if not any(x in key for x in keys_to_exclude)]
            for key in keys_to_keep:
                if any([x in key for x in processing_parameters.activity_datasets]):
                    vis_tc_dict[key[1:]] = tcf[key]
                else:
                    if ('counts' in key) or ('edges' in key):
                        pass
                    else:
                        self_motion_tc_dict[key[1:]] = tcf[key]

        self.visual_tcs.load_from_dict(vis_tc_dict)
        self.self_motion_tcs.load_from_dict(self_motion_tc_dict)

    def _load_preprocessing(self, full_kinem=False):
        """
        Load the data from the HDF file

        """	

        if self.preproc_info is not None:
            preproc_file = self.preproc_info.analysis_path
        else:
            raise AttributeError('No preprocessing info provided. Cannot load data.')

        with pd.HDFStore(preproc_file, 'r') as preproc:

            self._parse_params(preproc['params'])
            self._add_orientation_to_params()

            self.cell_matches = self._parse_cell_matches(preproc['cell_matches'], self.preproc_info.result)
        
            self._parse_preprocessing(preproc['matched_calcium'])
            if full_kinem:
                self.full_kinematics = preproc['full_traces']

            self.roi_info = preproc['roi_info']
            self.trial_set = preproc['trial_set']
            self.arena_corners = preproc['arena_corners']

    def _parse_preprocessing(self, matched_calcium):

        if self.metadata.rig in ['VWheel', 'VWheelWF']:
            self.metadata.exp_type = 'fixed'
        else:
            self.metadata.exp_type = 'free'

         # Apply wrapping for directions to get range [0, 360]
        matched_calcium['direction_wrapped'] = matched_calcium['direction'].copy()
        mask = matched_calcium['direction_wrapped'] > -1000
        matched_calcium.loc[mask, 'direction_wrapped'] = matched_calcium.loc[mask, 'direction_wrapped'].apply(wrap)

        if self.metadata.exp_type == 'free':
            matched_calcium['direction_rel_ground'] = matched_calcium['direction_wrapped'].copy()
            matched_calcium.loc[mask, 'direction_rel_ground'] = \
            matched_calcium.loc[mask, 'direction_rel_ground'] + matched_calcium.loc[mask, 'head_roll']
        else:
            matched_calcium['direction_rel_ground'] = matched_calcium['direction_wrapped'].copy()

        # Calculate orientation explicitly
        if 'orientation' not in matched_calcium.columns:
            matched_calcium['orientation'] = matched_calcium['direction_wrapped'].copy()
            matched_calcium['orientation_rel_ground'] = matched_calcium['direction_rel_ground'].copy()
            mask = matched_calcium['orientation'] > -1000
            matched_calcium.loc[mask, 'orientation'] = matched_calcium.loc[mask, 'orientation'].apply(wrap, bound=180.1)
            matched_calcium.loc[mask, 'orientation_rel_ground'] = matched_calcium.loc[mask, 'orientation_rel_ground'].apply(wrap, bound=180.1)


        # Get the columns for spikes, fluorescence, and kinematics
        spikes_cols = [key for key in matched_calcium.keys() if 'spikes' in key]
        fluor_cols = [key for key in matched_calcium.keys() if 'fluor' in key]
        motive_tracking_cols = ['mouse_y_m', 'mouse_z_m', 'mouse_x_m', 'mouse_yrot_m', 'mouse_zrot_m', 'mouse_xrot_m']

        # If there is more than one spatial or temporal frequency, include it, othewise don't
        stimulus_cols = ['trial_num', 'time_vector', 'direction', 'direction_wrapped', 'orientation', 'grating_phase']
        if len(self.exp_params.temporal_freq) > 1:
            stimulus_cols.append('temporal_freq')
        if len(self.exp_params.spatial_freq) > 1:
            stimulus_cols.append('spatial_freq')
        
        # For headfixed data
        eye_cols = ['eye_horizontal_vector_x', 'eye_horizontal_vector_y', 'eye_midpoint_x', 'eye_midpoint_y',
                    'pupil_center_ref_x', 'pupil_center_ref_y', 'fit_pupil_center_x', 'fit_pupil_center_y',
                    'pupil_diameter', 'minor_axis', 'pupil_rotation', 'eyelid_angle']
        eye_dlc_cols = ['pupil_center_x','pupil_center_y','pupil_top_left_x','pupil_top_left_y','pupil_top_x',
                        'pupil_top_y','pupil_top_right_x','pupil_top_right_y' 'pupil_right_x' 'pupil_right_y',
                        'pupil_bottom_right_x','pupil_bottom_right_y','pupil_bottom_x','pupil_bottom_y',
                        'pupil_bottom_left_x','pupil_bottom_left_y','pupil_left_x','pupil_left_y','eye_corner_nasal_x',
                        'eye_corner_nasal_y','eye_corner_temporal_x','eye_corner_temporal_y','eyelid_top_x',
                        'eyelid_top_y','eyelid_bottom_x','eyelid_bottom_y','led_x','led_y']
        wheel_cols = ['wheel_speed', 'wheel_acceleration']
        
        # For free data
        mouse_kinem_cols = ['mouse_heading', 'mouse_angular_speed', 'mouse_speed', 'mouse_acceleration',
                            'head_direction', 'head_height', 'head_pitch', 'head_yaw', 'head_roll']
        mouse_dlc_cols = ['mouse_snout_x', 'mouse_snout_y', 'mouse_barl_x', 'mouse_barl_y', 'mouse_barr_x',
                          'mouse_barr_y', 'mouse_x', 'mouse_y', 'mouse_body2_x', 'mouse_body2_y', 'mouse_body3_x',
                          'mouse_body3_y', 'mouse_base_x', 'mouse_base_y', 'mouse_head_x', 'mouse_head_y',
                          'miniscope_top_x', 'miniscope_top_y']

        if self.metadata.exp_type == 'fixed':
            self.kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + eye_cols + wheel_cols]
        else:
            self.kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + mouse_kinem_cols]

        if 'head_pitch' not in self.kinematics.columns:
            pitch = -wrap_negative(self.kinematics.mouse_xrot_m.values)
            self.kinematics['head_pitch'] = smooth_trace(pitch, range=(-180, 180), kernel_size=10, discont=2 * np.pi)

            yaw = wrap_negative(self.kinematics.mouse_zrot_m.values)
            self.kinematics['head_yaw'] = smooth_trace(yaw, range=(-180, 180), kernel_size=10, discont=2 * np.pi)

            roll = wrap_negative(self.kinematics.mouse_yrot_m.values)
            self.kinematics['head_roll'] = smooth_trace(roll, range=(-180, 180), kernel_size=10, discont=2*np.pi)
        
        # Convert to cm, cm/s or cm/s^2
        for col in ['mouse_y_m', 'mouse_z_m', 'mouse_x_m', 'head_height', 'mouse_speed', 'mouse_acceleration']:
            if col in self.kinematics.columns:
                self.kinematics[col] = self.kinematics[col] * 100.
            else:
                pass

        if 'wheel_speed' in self.kinematics.columns:
            self.kinematics['wheel_speed_abs'] = np.abs(self.kinematics['wheel_speed'].copy())
            self.kinematics['wheel_acceleration_abs'] = np.abs(self.kinematics['wheel_acceleration'].copy())
            self.kinematics['norm_wheel_speed'] = normalize(self.kinematics['wheel_speed_abs'])

        self.inferred_spikes = matched_calcium.loc[:, stimulus_cols + spikes_cols]
        self.inferred_spikes.columns = [key.rsplit('_', 1)[0] if 'spikes' in key else key for key in self.inferred_spikes.columns]
        self.deconv_fluor = matched_calcium.loc[:, stimulus_cols + fluor_cols]
        self.deconv_fluor.columns = [key.rsplit('_', 1)[0] if 'fluor' in key else key for key in self.deconv_fluor.columns]

    def _parse_params(self, df):
        params = df.to_dict('list')
        for key in params.keys():
            try:
                params[key] = np.array(literal_eval(params[key][0]))
            except:
                params[key] = params[key][0]
        
        self.exp_params = Metadata(params)

    def _add_orientation_to_params(self):
        self.exp_params.direction.sort()
        self.exp_params.orientation = self.exp_params.direction.copy()
        self.exp_params.orientation[self.exp_params.orientation < 0] += 180
        self.exp_params.orientation.sort()

        # Add wrapped directions, useful for plotting
        directions_wrapped = np.sort(wrap(self.exp_params.direction))
        directions_wrapped = np.concatenate((directions_wrapped, [360.]))
        self.exp_params.direction_wrapped = directions_wrapped

    def _parse_cell_matches(self, df, result):
        df = df.dropna().reset_index(drop=True).astype(int)
        old_cols = list(df.columns)
        if result != 'repeat':
            new_cols = [col.split("_")[-2] for col in old_cols[:2]]
            new_cols += old_cols[2:]
        else:
            new_cols = [col.split("_")[-1] for col in old_cols[:2]]
        col_map = dict(zip(old_cols, new_cols))
        new_df = df.rename(columns=col_map)
        return new_df

    # def _create_cell_class(self):
    #     cell_props = {}
    #
    #     for cell in self.cells:
    #         cell_props[cell] = Cell(cell)
    #
    #     return cell_props

    # def _cellprops_to_dataframe(self, attribute):
    #     df_list = []
    #     for cell_id, cell in self.cell_props.items():
    #         df = getattr(cell, attribute).copy()
    #         df.insert(loc=0, column='cell_id', value=cell_id)
    #         df_list.append(df)
    #
    #     attr_df = pd.concat(df_list).reset_index(drop=True)
    #     return attr_df