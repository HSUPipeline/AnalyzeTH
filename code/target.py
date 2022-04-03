""""Functions for spatial target analyses."""

from collections import Counter

import numpy as np

from spiketools.utils.data import restrict_range, get_value_by_time, get_value_by_time_range

from analysis import get_spike_positions

###################################################################################################
###################################################################################################

def compute_serial_position_fr(spikes, trial_starts, chest_openings, chest_trials, ptimes, positions):
    """Collect firing rates per segment across all trials"""

    all_frs = np.zeros([len(trial_starts), 4])
    for t_ind in range(len(trial_starts)):

        t_st = trial_starts[t_ind]
        ch_openings = chest_openings[t_ind]
        #ch_openings = chest_openings[chest_trials == t_ind]
        t_en = ch_openings[-1]

        t_mask = chest_trials == t_ind

        t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        t_spikes = restrict_range(spikes, t_st, t_en)

        seg_times = np.diff(np.insert(ch_openings, 0, t_time[0]))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))
        inds = count.keys()
        frs = np.array(list(count.values())) / seg_times * 1000

        all_frs[t_ind, :] = frs

    return all_frs


def compute_spatial_target_bins(spikes, trial_starts, chest_openings, chest_trials,
                                ptimes, positions, chest_bins, ch_xbin, ch_ybin):
    """Compute the binned firing rate based on spatial target."""

    # Collect firing per chest location across all trials
    target_bins = np.zeros(chest_bins)
    for t_ind in range(len(trial_starts)):

        t_st = trial_starts[t_ind]
        ch_openings = chest_openings[t_ind]
        t_en = ch_openings[-1]

        t_mask = chest_trials == t_ind

        t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        t_spikes = restrict_range(spikes, t_st, t_en)
        t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)

        seg_times = np.diff(np.insert(ch_openings, 0, t_time[0]))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))

        frs = np.array(list(count.values())) / seg_times * 1000

        cur_ch_xbin = ch_xbin[t_mask]
        cur_ch_ybin = ch_ybin[t_mask]

        for fr, xbin, ybin in zip(frs, cur_ch_xbin, cur_ch_ybin):
            target_bins[xbin, ybin] = fr

    return target_bins
