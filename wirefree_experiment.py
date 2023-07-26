import numpy as np
import pandas as pd
import xarray as xr

from ast import literal_eval
from functions_kinematic import wrap
import functions_tuning as tuning


class DataContainer():
    pass


class Cell():
    def __init__(self, name):
        self.id = name
        self.spikes_props = DataContainer()
        self.fluor_props = DataContainer()

    # def plot_tuning_curve(self)


class Metadata():
    def __init__(self, dictionary):
        for k, v in dictionary.items():           
            setattr(self, k, v)



class WirefreeExperiment():
    def __init__(self, filepath, use_xarray=False):
        # Experiment attributes
        self.metadata = None

        # Experimental metadata
        self.cell_matches = None
        self.arena_corners = None
        
        self.roi_info = None
        self.trial_set = None
        self.full_kinematics = None

        # Neural and kinematic data structures
        self.kinematics = None
        self.raw_spikes = None
        self.raw_fluor = None

        # Load the data
        self._load_hdf(filepath, use_xarray)
        print(f'Loaded experiment from {filepath}')

        # Create cell objects
        self.cells = [el for el in self.raw_spikes.columns if "cell" in el]
        self.cell_props = self._create_cell_class()


    def _load_hdf(self, file, use_xarray):
        """
        Load the data from the HDF file
        
        Parameters
        ----------
        file : str
            Path to the HDF file
        use_xarray : bool
            Whether to use xarray for the data structures
        """	

        with pd.HDFStore(file) as h:

                self._parse_params(h['params'])
                self._add_orientation_to_params()
            
                self._parse_kinematic_data(h['matched_calcium'], use_xarray)

                if use_xarray:
                    self.full_kinematics = h['full_traces'].to_xarray()
                else:
                    self.full_kinematics = h['full_traces']

                self.roi_info = h['roi_info']
                self.trial_set = h['trial_set']
                self.cell_matches = h['cell_matches']
                self.arena_corners = h['arena_corners']

    def _parse_kinematic_data(self, matched_calcium, use_xarray):

        # Calculate orientation expicitly
        if 'orientation' not in matched_calcium.columns:
            matched_calcium['orientation'] = matched_calcium['direction']
            matched_calcium['orientation'][(matched_calcium['orientation'] > -180) & (matched_calcium['orientation'] < 0)] += 180

        # Apply wrapping for directions to get range [0, 360]
        matched_calcium['direction_wrapped'] = matched_calcium['direction']
        mask = matched_calcium['direction_wrapped'] > -1000
        matched_calcium.loc[mask, 'direction_wrapped'] = matched_calcium.loc[mask, 'direction_wrapped'].apply(wrap)
           
        # For all data
        self.metadata.rig = matched_calcium['mouse'].values[0].split('_', 1)[0]
        if self.metadata.rig in ['VWheel', 'VWheelWF']:
            self.metadata.exp_type = 'fixed'
        else:
            self.metadata.exp_type = 'free'
        
        self.metadata.animal = matched_calcium['mouse'].values[0].split('_', 1)[1:]
        self.metadata.exp_date = matched_calcium['datetime'][0]
        spikes_cols = [key for key in matched_calcium.keys() if 'spikes' in key]
        fluor_cols = [key for key in matched_calcium.keys() if 'fluor' in key]
        motive_tracking_cols = ['mouse_y_m','mouse_z_m','mouse_x_m','mouse_yrot_m','mouse_zrot_m','mouse_xrot_m']


        # If there is more than one spatial or temporal frequency, include it, othewise don't
        stimulus_cols = ['trial_num', 'time_vector', 'direction', 'direction_wrapped', 'orientation', 'grating_phase',]
        if len(self.metadata.temporal_freq) > 1:
            stimulus_cols.append('temporal_freq')
        if len(self.metadata.spatial_freq) > 1:
            stimulus_cols.append('spatial_freq')
        
        # For headfixed data
        eye_cols = ['eye_horizontal_vector_x', 'eye_horizontal_vector_y', 'eye_midpoint_x', 'eye_midpoint_y', 'pupil_center_ref_x', 'pupil_center_ref_y', 'fit_pupil_center_x', 'fit_pupil_center_y',
                       'pupil_diameter', 'minor_axis', 'pupil_rotation', 'eyelid_angle']
        eye_dlc_cols = ['pupil_center_x','pupil_center_y','pupil_top_left_x','pupil_top_left_y','pupil_top_x','pupil_top_y','pupil_top_right_x','pupil_top_right_y' 'pupil_right_x' 'pupil_right_y', 
                           'pupil_bottom_right_x','pupil_bottom_right_y','pupil_bottom_x','pupil_bottom_y','pupil_bottom_left_x','pupil_bottom_left_y','pupil_left_x','pupil_left_y','eye_corner_nasal_x',
                           'eye_corner_nasal_y','eye_corner_temporal_x','eye_corner_temporal_y','eyelid_top_x','eyelid_top_y','eyelid_bottom_x','eyelid_bottom_y','led_x','led_y']
        wheel_cols = ['wheel_speed', 'wheel_acceleration']
        
        # For free data
        mouse_kinem_cols = ['mouse_heading', 'mouse_angular_speed', 'mouse_speed', 'mouse_acceleration', 'head_direction', 'head_height']
        mouse_dlc_cols = ['mouse_snout_x', 'mouse_snout_y', 'mouse_barl_x', 'mouse_barl_y', 'mouse_barr_x', 'mouse_barr_y', 'mouse_x', 'mouse_y', 'mouse_body2_x', 'mouse_body2_y', 
                          'mouse_body3_x', 'mouse_body3_y', 'mouse_base_x', 'mouse_base_y', 'mouse_head_x', 'mouse_head_y', 'miniscope_top_x', 'miniscope_top_y']

        if self.metadata.exp_type == 'fixed':
            self.kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + eye_cols + wheel_cols]
        else:
            self.kinematics = matched_calcium.loc[:, stimulus_cols + motive_tracking_cols + mouse_kinem_cols]
            
        self.raw_spikes = matched_calcium.loc[:, stimulus_cols + spikes_cols]
        self.raw_spikes.columns = [key.rsplit('_', 1)[0] if 'spikes' in key else key for key in self.raw_spikes.columns]
        self.raw_fluor = matched_calcium.loc[:, stimulus_cols + fluor_cols]
        self.raw_fluor.columns = [key.rsplit('_', 1)[0] if 'fluor' in key else key for key in self.raw_fluor.columns]

        if use_xarray:
            self.kinematics = self.kinematics.to_xarray()
            self.raw_spikes = self.raw_spikes.to_xarray()
            self.raw_fluor = self.raw_fluor.to_xarray()

    def _parse_params(self, df):
        params = df.to_dict('list')
        for key in params.keys():
            try:
                params[key] = np.array(literal_eval(params[key][0]))
            except:
                params[key] = params[key][0]
        
        self.metadata = Metadata(params)

    def _add_orientation_to_params(self):
        self.metadata.direction.sort()
        self.metadata.orientation = self.metadata.direction.copy()
        self.metadata.orientation[self.metadata.orientation < 0] += 180
        self.metadata.orientation.sort()

        # Add wrapped directions, useful for plotting
        directions_wrapped = np.sort(wrap(self.metadata.direction))
        directions_wrapped = np.concatenate((directions_wrapped, [360.]))
        self.metadata.direction_wrapped = directions_wrapped

    def _create_cell_class(self):
        cell_props = {}
    
        for cell in self.cells:
            cell_props[cell] = Cell(cell)

        return cell_props