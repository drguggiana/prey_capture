import os
import pandas as pd 
import numpy as np
from skimage import io


def get_trial_duration_stats(df, trial_key, time_key):
    grouped_trials = df[df[trial_key] > 0].groupby(trial_key)
    trial_durations = grouped_trials.apply(lambda x: x[time_key].to_list()[-1] - x[time_key].to_list()[0])
    print(f"Min. trial. dur.: {trial_durations.min():.2f}, Max. trial. dur.: {trial_durations.max():.2f}"
          f" Mean. trial. dur.: {trial_durations.mean():.2f}")
    return np.array((trial_durations.min(), trial_durations.max(), trial_durations.mean()))


def extract_timestamp(frame):
    """Extract the timestamp from a wirefree miniscope frame, based on their example code"""

    footer = frame[-1, -8:]
    timestamp = footer[0] + (footer[1] << 8) + (footer[2] << 16) + (footer[3] << 24)
    return timestamp


def correct_timestamp_jumps(timestamps, jump_size=5):
    timestamps_corrected = timestamps.copy()
    jump_idxs = np.argwhere(np.abs(np.diff(timestamps)) > jump_size)

    if jump_idxs.size == 0:
        jump_idxs = []
        jump_vals = []

    else:
        if jump_idxs.size == 1:
            jump_idxs = jump_idxs[0]
            jump_vals = np.diff(timestamps)[jump_idxs]
            jump_idxs += 1     # Add one for correct index
            timestamps_corrected[jump_idxs[0]:] = timestamps[jump_idxs[0]:] - jump_vals[0]

        else:
            jump_idxs = jump_idxs.squeeze()
            jump_vals = np.diff(timestamps)[jump_idxs]
            jump_idxs += 1     # Add one for correct index

            # If an odd number of discontinuities, handle the last one first
            if len(jump_idxs) % 2 != 0:

                timestamps_corrected[jump_idxs[-1]:] = timestamps[jump_idxs[-1]:] - jump_vals[-1]

                # Remove the last jump_idx and jump_val to make the arrays even
                jump_idxs_even = jump_idxs[:-1]
                jump_vals_even = jump_vals[:-1]
                    
                # Reshape the arrays to be pairs of jump_idxs and jump_vals
                jump_idxs_even = jump_idxs_even.reshape(-1, 2)
                jump_vals_even = jump_vals_even.reshape(-1, 2)

            else:
                # Reshape the arrays to be pairs of jump_idxs and jump_vals
                jump_idxs_even = jump_idxs.reshape(-1, 2)
                jump_vals_even = jump_vals.reshape(-1, 2)

            for jump_idx, jump_val in zip(jump_idxs_even, jump_vals_even):
                timestamps_corrected[jump_idx[0]:jump_idx[1]] = timestamps[jump_idx[0]:jump_idx[1]] - jump_val[0]
    
    return timestamps_corrected, jump_idxs, jump_vals


def update_sync_file(timestamps_in, target_sync_in, miniscope_channel=4):
    """Insert the tif timestamps into the sync file corresponding to the target trial"""
    sync_info = pd.read_csv(target_sync_in, header=None)

    # # rename original sync file
    original_sync_filename = target_sync_in.replace('.csv', '_original.csv')
    os.rename(target_sync_in, original_sync_filename)

    # find where the miniscope recordings start
    sync_start = np.argwhere(sync_info.iloc[:, miniscope_channel].to_numpy() > 3)
    sync_start = sync_start[0][0]

    # get the time vector from the trial
    trial_time = sync_info.iloc[:, 0].to_numpy()

    # add the offset to the timestamps
    timestamps_in += trial_time[sync_start]

    # find the closest sync index for each timestamp
    best_idx = np.searchsorted(trial_time, timestamps_in)

    # get rid of the frames occurring after the end of the experiment
    best_idx = best_idx[best_idx < sync_info.shape[0]]

    # reset the timestamps before replacing them
    sync_info.iloc[sync_start + 1:, miniscope_channel] = 0

    # # modify the sync info to add the timestamps
    sync_info.iloc[best_idx, miniscope_channel] = 5

    # save the corrected file under the original filename (dropping index and header to match the original)
    sync_info.to_csv(target_sync_in, index=False, header=False)
    return 'Successful timestamp correction'


def fix_wirefree_timestamps(tif_path, sync_path):
    stack = io.imread(tif_path)
    timestamps = np.array([extract_timestamp(el)/1000 for el in stack])
    corrected_timestamps , idxs, _ = correct_timestamp_jumps(timestamps)

    # There are still these little mistakes throughout that cause timing errors down the line.
    # Let's try to fix them by fitting a line to the timestamps and then using that as the correction
    x = np.arange(len(corrected_timestamps))
    slope, intercept = np.polyfit(x, corrected_timestamps, deg=1)
    regressed_timestamps = x * slope + intercept
    message = update_sync_file(regressed_timestamps, sync_path)
    return message


if __name__ == '__main__':
    tif_path = file_info['tif_path']
    sync_path = file_info['sync_path']

    message = fix_wirefree_timestamps(tif_path, sync_path)
    print(message)
