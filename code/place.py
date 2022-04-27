""""Functions for place analyses."""

import numpy as np

from spiketools.spatial.occupancy import (compute_nbins, compute_spatial_bin_assignment,
                                          compute_occupancy)
from spiketools.utils.data import restrict_range, get_value_by_time_range

from analysis import get_spike_positions, compute_bin_firing

###################################################################################################
###################################################################################################

def get_trial_place(spikes, trials, bins, ptimes, positions, speed,
                    x_edges=None, y_edges=None, occ_kwargs=None):
    """Get the spatially binned firing, per trial."""

    n_trials = len(trials)
    n_bins = compute_nbins(bins)

    bin_firing_trial = np.zeros([n_trials, n_bins])
    bin_firing_trial_norm = np.zeros([n_trials, n_bins])

    for ind, (nav_start, nav_stop) in enumerate(zip(trials.navigation_start[:], trials.navigation_stop[:])):

        # Get data for selected trial: trial positions, spikes, and spike positions
        t_times, t_pos = get_value_by_time_range(ptimes, positions, nav_start, nav_stop)
        t_spikes = restrict_range(spikes, nav_start, nav_stop)
        t_speed = restrict_range(speed, nav_start, nav_stop)
        t_spike_pos_x, t_spike_pos_y = get_spike_positions(t_spikes, t_times, t_pos)
        t_spike_pos = np.array([t_spike_pos_x, t_spike_pos_y])

        # Compute spatial bin assignments for each spike & compute occupancy
        xt_bin, yt_bin = compute_spatial_bin_assignment(t_spike_pos, x_edges, y_edges)
        tocc = compute_occupancy(t_pos, t_times, bins, t_speed, **occ_kwargs)

        # Compute and collect binned firing per trial
        bin_firing_trial[ind, :] = (compute_bin_firing(xt_bin, yt_bin, bins)).flatten()
        bin_firing_trial_norm[ind, :] = bin_firing_trial[ind, :] / tocc.flatten()

    return bin_firing_trial_norm
