"""Utilities for working with Treasure Hunt data."""

import numpy as np

from spiketools.utils.epoch import epoch_data_by_range

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
