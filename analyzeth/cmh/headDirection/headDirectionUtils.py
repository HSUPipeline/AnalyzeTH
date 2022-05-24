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



def get_hd_spike_headings (nwbfile, unit_ix):
    """
    Get spike headings in degrees during navigation periods
    """
    # Spike data - navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]
    spikes = subset_period_event_time_data(nwbfile.units.get_unit_spike_times(unit_ix), navigation_start_times, navigation_stop_times)

    # Head Direction data 
    head_direction = nwbfile.acquisition['heading']['direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]

    # Histogram
    hd_spikes = np.array(get_spike_heading(spikes, hd_times, hd_degrees))

    return hd_spikes


def nwb_hd_cell_hist(nwbfile, unit_ix = 0, binsize = 1, windowsize = 23,  smooth = True):
    """ 
    Get head direction histogram 
    """
    # Spike data - navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]
    spikes = subset_period_event_time_data(nwbfile.units.get_unit_spike_times(unit_ix), navigation_start_times, navigation_stop_times)

    # Head Direction data 
    head_direction = nwbfile.acquisition['heading']['direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    hd_spikes = get_spike_heading(spikes, hd_times, hd_degrees)

    # Histogram
    hd_histogram = get_hd_histogram(hd_spikes, binsize, windowsize, smooth)

    return hd_histogram

def normalize_hd_hist_to_occupancy(hd_hist, occupancy=[], nwbfile = None, binsize = 1, smooth= True, windowsize = 23):
    """
    Normalize histogram or set of histograms to occpancy. Occupancy is seconds spent in each bin, thus
    this generates a psuedo firing rate in Hz per bin
    """
    # Occupancy - seconds per bin
    if occupancy == []:
        occupancy = compute_hd_occupancy(nwbfile, binsize, smooth, windowsize)
    hd_firingRate =  hd_hist/ occupancy
    return hd_firingRate

def compute_hd_score(hd_hist):
    """
    Compute head direction score based on method from Gerlei 2020

    HD score should be >= 0.5 to be hd cell

    """
    n_bins = len(hd_hist)
    binsize = 360/n_bins
    degs = np.arange(0, 360, binsize)
    rads = np.radians(degs)
    dy = np.sin(rads)
    dx = np.cos(rads)
    xtotal = sum(dx * hd_hist)/sum(hd_hist)
    ytotal = sum(dy * hd_hist)/sum(hd_hist)
    hd_score = np.sqrt(xtotal**2 + ytotal**2)

    return hd_score


#########################################
# Smooth HD Histograms
#########################################

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


#########################################
# Occupancy
#########################################

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
    head_direction = nwbfile.acquisition['heading']['direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    
    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start

    # Get navigation periods 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]

    # Get hd at each ms timepoint 
    trial_ms = np.arange(np.ceil(session_len))
    navigation_ms = subset_period_event_time_data(trial_ms, navigation_start_times, navigation_stop_times)
    hd_ms = get_spike_heading(navigation_ms, hd_times, hd_degrees)

    occ_counts = get_hd_histogram(hd_ms, binsize, windowsize, smooth)
    # # Compute occupancy for each HD bin in seconds
    # if smooth:
    #     print('Smoothing...')
    #     occ_counts = get_hd_histogram(hd_ms, binsize, windowsize, smooth)
    # else:
    #     _, occ_counts = bin_circular(hd_ms, binsize = binsize)
    
    occupancy = occ_counts #/1e3  # getting #ms time points in each HD, convert to s by /1e3   
    print('Occupancy determined...')

    if return_hds:
        return occupancy, hd_ms   # return occupancy in seconds, head direction in degrees each ms
    else:
        return occupancy


