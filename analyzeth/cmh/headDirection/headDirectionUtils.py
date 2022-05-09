import numpy as np
from analyzeth.cmh.utils import *
from analyzeth.analysis import get_spike_heading, bin_circular
import numpy as np
import math
#import matplotlib.pyplot as plt


# def bin_circular(degrees, binsize = 10):
#     """Bin circular data.

#     Parameters
#     ----------
#     degrees : 1d array
#         Data to bin.

#     Returns
#     -------
#     bin_edges : 1d array
#         Bin edge definitions.
#     counts : 1d array
#         Count values per bin.
#     """
#     print('inside bin circular')
#     print('binsize', binsize)
    

#     bin_edges = np.arange(0, 361, binsize)
    
#     print('bin_edges', bin_edges)

#     counts, _ = np.histogram(degrees, bins=bin_edges)

#     return bin_edges, counts

# Modified from MNL
def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def get_hd_histogram(degrees, binsize = 1, windowsize=23, smooth = True):
    """
    Get histogram from head_directions in degrees

    Smoothed or not smoothed
    """
    bin_edges = np.arange(0,361, binsize)
    binned_hd, _ = np.histogram(degrees, bins = bin_edges)
    if smooth:
        smooth_hd = get_rolling_sum(binned_hd, window=windowsize)
        return smooth_hd
    else:
        return binned_hd



def compute_hd_occupancy(nwbfile, binsize = 1, windowsize = 23, smooth = True, return_hds = False, return_counts = False):
    """
    Compute occupancy in seconds for each HD bin during navigation periods

    It is more efficient to run this before headDirection_cell analysis when
    multiple runs are done in sequence. Occupancy does not change per session
    and this calculation is the most time intestive.

    Parameters
    ----------
    nwbfile: .nwb
        NWB file for a single session of TH

    Returns
    -------
    occupancy: 1D arr
        ammount of time spent in each HD bin (currently 10 degrees) in seconds

    """
    # Head direction data
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    
    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start

    # Get navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]

    # Get hd at each ms timepoint 
    trial_ms = np.arange(np.ceil(session_len))
    navigation_ms = subset_period_event_time_data(trial_ms, navigation_start_times, navigation_end_times)
    hd_ms = get_spike_heading(navigation_ms, hd_times, hd_degrees)
    print('Occupancy determined...')

    occ_counts = get_hd_histogram(hd_ms, binsize, windowsize, smooth)
    # # Compute occupancy for each HD bin in seconds
    # if smooth:
    #     print('Smoothing...')
    #     occ_counts = get_hd_histogram(hd_ms, binsize, windowsize, smooth)
    # else:
    #     _, occ_counts = bin_circular(hd_ms, binsize = binsize)
    
    occupancy = occ_counts/1e3  # getting #ms time points in each HD, convert to s by /1e3
    
    if return_hds:
        return occupancy, hd_ms   # return occupancy in seconds, head direction in degrees each ms
    else:
        return occupancy


