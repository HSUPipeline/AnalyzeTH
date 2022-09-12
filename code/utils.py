"""Utilities for working with Treasure Hunt data."""


import numpy as np

from convnwb.io import load_nwbfile
from spiketools.utils.epoch import epoch_data_by_range
from spiketools.utils.base import count_elements

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
