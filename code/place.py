""""Functions for place analyses."""

#import warnings
from functools import partial

#import numpy as np

from spiketools.stats.anova import create_dataframe_bins, fit_anova
#from spiketools.spatial.occupancy import (compute_nbins, compute_bin_assignment,
#                                          compute_bin_counts_assgn, compute_bin_counts_pos,
#                                          normalize_bin_counts, compute_occupancy)
#from spiketools.utils.extract import get_range, get_values_by_time_range, get_values_by_times

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(bin)'
FEATURE = 'C(bin)'
COLUMNS = ['bin', 'fr']

# Create functions for place model
create_df_place = partial(create_dataframe_bins, columns=COLUMNS)
fit_anova_place = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################

# # OLD VERSION
# def compute_place_bins(spikes, bins, ptimes, positions, x_edges=None, y_edges=None, occ=None):
#     """Compute the binned firing rate based on player position."""

#     spike_positions = get_values_by_times(ptimes, positions, spikes, threshold=0.25)
#     x_binl, y_binl = compute_bin_assignment(spike_positions, x_edges, y_edges)
#     bin_firing = compute_bin_counts_assgn(bins, x_binl, y_binl)

#     if occ is not None:
#         bin_firing = normalize_bin_counts(bin_firing, occ)

#     return bin_firing

# OLD VERSION
# def get_trial_place(spikes, bins, tstarts, tends, ptimes, positions, speed,
#                     x_edges=None, y_edges=None, occ_kwargs=None):
#     """Get the spatially binned firing, per trial."""

#     n_trials = len(tstarts)
#     n_bins = compute_nbins(bins)

#     bin_firing_trial = np.zeros([n_trials, n_bins])
#     bin_firing_trial_norm = np.zeros([n_trials, n_bins])

#     # TODO / check - this might be replaceable by epoch functions
#     for ind, (nav_start, nav_stop) in enumerate(zip(tstarts, tends)):

#         # Get data for selected trial: trial positions, spikes, and spike positions
#         t_times, t_pos = get_values_by_time_range(ptimes, positions, nav_start, nav_stop)
#         _, t_speed = get_values_by_time_range(ptimes, speed, nav_start, nav_stop)
#         t_spikes = get_range(spikes, nav_start, nav_stop)
#         t_spike_pos = get_values_by_times(t_times, t_pos, t_spikes, threshold=0.25)

#         # Compute trial level occupancy
#         tocc = compute_occupancy(t_pos, t_times, bins, t_speed, **occ_kwargs)

#         # Compute and spatial bin assignments collect binned firing per trial
#         xt_bin, yt_bin = compute_bin_assignment(t_spike_pos, x_edges, y_edges)
#         bin_firing_trial[ind, :] = (compute_bin_counts_assgn(bins, xt_bin, yt_bin)).flatten()
#         bin_firing_trial_norm[ind, :] = normalize_bin_counts(bin_firing_trial[ind, :], tocc.flatten())

#     return bin_firing_trial_norm
