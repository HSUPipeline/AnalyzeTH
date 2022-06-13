""""Functions for serial position analyses."""

from functools import partial
from collections import Counter

import numpy as np

from spiketools.stats.anova import create_dataframe, fit_anova
from spiketools.utils.data import restrict_range, get_value_by_time, get_value_by_time_range

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(segment)'
FEATURE = 'C(segment)'
COLUMNS = ['segment', 'fr']

# Create functions for serial position model
create_df_serial = partial(create_dataframe, columns=COLUMNS)
fit_anova_serial = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################

def compute_serial_position_fr(spikes, nav_starts, chest_openings, chest_trials):
    """Collect firing rates per segment across all trials"""

    all_frs = np.zeros([len(nav_starts), 4])
    for t_ind in range(len(nav_starts)):

        t_st = nav_starts[t_ind]
        ch_openings = chest_openings[t_ind]
        t_en = ch_openings[-1]

        chest_trials == t_ind

        t_spikes = restrict_range(spikes, t_st, t_en)

        seg_times = np.diff(np.insert(ch_openings, 0, t_st))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))
        inds = count.keys()
        frs = np.array(list(count.values())) / seg_times

        all_frs[t_ind, :] = frs

    return all_frs


def get_spike_position_trial(t_ind, nav_starts, ch_openings, chest_trials, ptimes, positions, spikes):
    """get spike and position data of one trial"""
    
    t_st = nav_starts[t_ind]
    ch_open = ch_openings[t_ind]
    t_en = ch_open[-1]
    
    # Get the chests for the current trial
    t_mask = chest_trials == t_ind

    # Select chest openings for the current trial
    t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
    ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_open]

    # Restrict spikes to the selected trial
    t_spikes = restrict_range(spikes, t_st, t_en)
    t_spike_xs, t_spike_ys = get_spike_positions(t_spikes, t_time, t_pos)
    spike_positions = np.array([t_spike_xs, t_spike_ys])
    
    return t_mask, t_spikes, t_spike_xs, t_spike_ys, t_pos, ch_times


def get_frs_trial(t_ind, nav_starts, ch_openings, spikes, all_frs, all_spikes):
    """compute firing rates in each segment of one trial"""
    
    t_st = nav_starts[t_ind]
    ch_open = ch_openings[t_ind]
    t_en = ch_open[-1]

    t_spikes = restrict_range(spikes, t_st, t_en)
    all_spikes.append(t_spikes)
    
    seg_times[t_ind,:] = np.diff(np.insert(ch_openings[t_ind], 0, t_st))
    
    count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
    count.update(np.digitize(t_spikes, ch_open))
    frs = np.array(list(count.values())) / seg_times
    all_frs.append(frs[t_ind])

    return count, frs, all_frs, all_spikes
