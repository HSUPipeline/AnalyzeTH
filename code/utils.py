"""Utilities for working with Treasure Hunt data."""


import numpy as np

from sklearn.preprocessing import MinMaxScaler
from convnwb.io import load_nwbfile

from spiketools.utils.epoch import epoch_data_by_range, epoch_spikes_by_range
from spiketools.utils.base import count_elements
from spiketools.utils.extract import get_range, get_values_by_time_range, get_values_by_times

###################################################################################################
###################################################################################################

def select_navigation(data, navigation_starts, navigation_stops):
    """Helper function to select data from during navigation periods."""

    times_trials, values_trials = epoch_data_by_range(\
        data.timestamps[:], data.data[:].T, navigation_starts, navigation_stops)

    return times_trials, values_trials

def stack_trials(times_trials, values_trials):
    """Helper function to recombine data across trials."""

    times = np.hstack(times_trials)
    values = np.hstack(values_trials)

    return times, values

def normalize_data(data):
    """Helper function to normalize data into range of [0, 1]. """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def compute_distance_error(nwbfiles, data_folder):
    """Compute the normalized distance error across sessions."""
    
    error_avg = np.zeros(len(nwbfiles))
    for ind, nwbfile in enumerate(nwbfiles):
        nwbfile, io = load_nwbfile(nwbfile, data_folder, return_io=True)
        error = nwbfile.trials.error[:]
        error_avg[ind] = np.mean(error)
    error_normalized = normalize_data(error_avg)
    
    return error_normalized

def compute_recall_percent(nwbfiles, data_folder):
    """Compute the percentge of recall across sessions."""
    
    correct = np.zeros([len(nwbfiles)])
    for ind, nwbfile in enumerate(nwbfiles):
        nwbfile, io = load_nwbfile(nwbfile, data_folder, return_io=True)
        correct[ind] = np.mean(nwbfile.trials.correct[:])*100
        
    return correct

def get_confidence_response(nwbfiles, data_folder, labels):
    """Count the confidence response in each category across sessions."""
    
    for ind, nwbfile in enumerate(nwbfiles):
        nwbfile, io = load_nwbfile(nwbfile, data_folder, return_io=True)
        name = nwbfile.session_id
        conf_counts = count_elements(nwbfile.trials.confidence_response.data[:],
                                     labels=labels)
    
    return conf_counts

def reshape_bins(target_bins, bins):
    """Reshape chest bins from [3, 5] to [5, 7]"""
    
    add_row = np.zeros(len(target_bins[0]))
    add_col = np.zeros((bins[1],1))
    
    temp = np.vstack([add_row, target_bins])
    temp = np.vstack([temp, add_row])
    temp = np.hstack([temp, add_col])
    reshaped_target = np.hstack([add_col, temp])
    
    return reshaped_target

def get_pos_per_bin(intersect, chest_trial_number, ptimes, positions, spikes, nav_starts, ch_openings_all):
    """Get chest position, spikes, spike positions within a specific bin"""
    
    tpos_all, tspikes_x, tspikes_y = [], [], []
    for ind in intersect:
        if ind not in chest_trial_number[:,0]:
            t_time, t_pos = get_values_by_time_range(ptimes, positions, ch_openings_all[ind-1], ch_openings_all[ind])
            t_spikes = get_range(spikes, ch_openings_all[ind-1], ch_openings_all[ind])

        else:
            ch_trial = int(ind / 4)
            t_time, t_pos = get_values_by_time_range(ptimes, positions, nav_starts[ch_trial], ch_openings_all[ind])
            t_spikes = get_range(spikes, nav_starts[ch_trial], ch_openings_all[ind])

        tpos_all.append(t_pos)
        t_spike_positions = get_values_by_times(t_time, t_pos, t_spikes, threshold=0.25)
        tspikes_x.append(t_spike_positions[0])
        tspikes_y.append(t_spike_positions[1])

    tspikes_x = np.concatenate(tspikes_x).ravel()
    tspikes_y = np.concatenate(tspikes_y).ravel()
    
    tspikes_pos = np.array([tspikes_x, tspikes_y])
    
    return tpos_all, tspikes_pos


def normalize_segment_spikes(t_spikes, start, stop, feature_range):
    scaler = MinMaxScaler(feature_range=feature_range)
    seg_spikes = epoch_spikes_by_range(t_spikes, start, stop, reset=True)
    seg_spikes_norm = []
    for ind in range(len(seg_spikes)):
        spikes_norm = scaler.fit_transform(seg_spikes[ind].reshape(-1, 1)) if seg_spikes[ind].size != 0 else np.array([])
        seg_spikes_norm.append(spikes_norm.flatten())
    
    return seg_spikes_norm