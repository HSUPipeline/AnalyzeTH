"""Utilities for working with Treasure Hunt data."""

import numpy as np

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


def normalize_data(data, feature_range):
    """Helper function to normalize data into specific range. """

    data_std = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    data_scaled = data_std * (feature_range[1] - feature_range[0]) + feature_range[0]

    return data_scaled


def compute_recall_percent(nwbfiles, data_folder):
    """Compute the percentage of recall across sessions."""

    correct = np.zeros([len(nwbfiles)])
    for ind, nwbfile in enumerate(nwbfiles):
        nwbfile, io = load_nwbfile(nwbfile, data_folder, return_io=True)
        correct[ind] = np.mean(nwbfile.trials.correct[:]) * 100

    return correct


def get_confidence_response(nwbfiles, data_folder, labels):
    """Count the confidence response in each category across sessions."""

    conf_all = []
    for ind, nwbfile in enumerate(nwbfiles):
        nwbfile, io = load_nwbfile(nwbfile, data_folder, return_io=True)
        conf = nwbfile.trials.confidence_response.data[:]
        conf_all.append(conf)

    conf_all = np.concatenate(conf_all).ravel()
    conf_counts = count_elements(conf_all, labels=labels)

    return conf_counts


def reshape_bins(target_bins, bins):
    """Reshape chest bins from [5, 8] to [7, 10] for visualization purpose."""

    target_tp = np.transpose(target_bins)
    add_row = np.zeros(len(target_tp[0]))
    add_col = np.zeros((bins[0],1))

    temp = np.vstack([add_row, target_tp])
    temp = np.vstack([temp, add_row])
    temp = np.hstack([temp, add_col])
    reshaped_target_bins = np.hstack([add_col, temp])

    return reshaped_target_bins


def get_pos_per_bin(intersect, chest_trial_number, ptimes, positions,
                    spikes, nav_starts, ch_openings_all):
    """Get chest position, spikes, spike positions within a specific bin."""

    tpos_all, tspikes_x, tspikes_y = [], [], []
    for ind in intersect:
        if ind not in chest_trial_number[:,0]:
            t_time, t_pos = get_values_by_time_range(\
                ptimes, positions, ch_openings_all[ind-1], ch_openings_all[ind])
            t_spikes = get_range(spikes, ch_openings_all[ind-1], ch_openings_all[ind])

        else:
            ch_trial = int(ind / 4)
            t_time, t_pos = get_values_by_time_range(\
                ptimes, positions, nav_starts[ch_trial], ch_openings_all[ind])
            t_spikes = get_range(spikes, nav_starts[ch_trial], ch_openings_all[ind])

        tpos_all.append(t_pos)
        t_spike_positions = get_values_by_times(t_time, t_pos, t_spikes, threshold=0.25)
        tspikes_x.append(t_spike_positions[0])
        tspikes_y.append(t_spike_positions[1])

    tspikes_x = np.concatenate(tspikes_x).ravel()
    tspikes_y = np.concatenate(tspikes_y).ravel()

    tspikes_pos = np.array([tspikes_x, tspikes_y])

    return tpos_all, tspikes_pos


def normalize_spikes_by_segment(t_spikes, start, stop, feature_range):
    """Normalize spikes into specific ranges by segment."""

    seg_spikes = epoch_spikes_by_range(t_spikes, start, stop, reset=True)
    seg_spikes_norm = []

    for seg in seg_spikes:
        spikes_norm = normalize_data(seg, feature_range) if seg.size != 0 else np.array([])
        spikes_norm[np.isnan(spikes_norm)] = 0
        seg_spikes_norm.append(spikes_norm.flatten())

    return seg_spikes_norm
