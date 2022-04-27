""""Functions for serial position analyses."""

from collections import Counter

import numpy as np

from spiketools.utils.data import restrict_range, get_value_by_time, get_value_by_time_range

###################################################################################################
###################################################################################################

def compute_serial_position_fr(spikes, nav_starts, chest_openings, chest_trials, ptimes, positions):
    """Collect firing rates per segment across all trials"""

    all_frs = np.zeros([len(nav_starts), 4])
    for t_ind in range(len(nav_starts)):

        t_st = nav_starts[t_ind]
        ch_openings = chest_openings[t_ind]
        t_en = ch_openings[-1]

        t_mask = chest_trials == t_ind

        t_time, t_pos = get_value_by_time_range(ptimes, positions, t_st, t_en)
        ch_times = [get_value_by_time(t_time, t_pos, ch_op) for ch_op in ch_openings]

        t_spikes = restrict_range(spikes, t_st, t_en)

        seg_times = np.diff(np.insert(ch_openings, 0, t_time[0]))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))
        inds = count.keys()
        frs = np.array(list(count.values())) / seg_times

        all_frs[t_ind, :] = frs

    return all_frs