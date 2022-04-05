""""Analysis functions for TH analysis."""

import numpy as np

###################################################################################################
###################################################################################################

def get_spike_positions(spikes, times, positions):
    """Get xy-positions for spike times."""

    #tspike, tpos = [], []
    #inds = []

    spike_xs = []
    spike_ys = []

    for spike in spikes:

        idx = (np.abs(times - spike)).argmin()
        diff = np.abs(times[idx] - spike)

        if diff < 100:

            spike_xs.append(positions[0, idx])
            spike_ys.append(positions[1, idx])

            # inds.append(idx)
            # tspike.append(spike)
            # tpos.append(ptimes[idx])

    return spike_xs, spike_ys


def get_spike_heading(spike_times, hd_times, hd_degrees):
    """Get head direciton for spike times.
    
    For each spike, search for the last recorded HD. HD is recorded only
    when subject changes direction.

    Parameters
    ----------
    spike times: 1D arr
        1D array of spike times in ms

    hd_times: 1D arr
        1D array of times when hd was recorded
        ix matches hd_degrees

    hd_degrees: 1D arr
        1D array of head direction in degrees (0-360)
        ix matches hd_times
    
    match_freedom_ms: int
        default = 10 (?)
        leniency in finding matches. this should not be needed with the new method
    
    Returns
    -------
    spike_hds: 1D arr
        1D array of head directions for each spike
        len(spike_hds) should match len(spikes)
    """

    spike_hds = []

    for spike_time in spike_times:
        
        hd_idx = np.abs(hd_times[hd_times <= spike_time] - spike_time).argmin() 
        spike_hds.append(hd_degrees[hd_idx])

    return spike_hds


def get_spike_heading_old(spikes, times, head_dirs, match_freedom_ms = 100):
    """Get head direciton for spike times.
    
    For each spike, search for the closest recorded HD within theshold set
    with match_freedom_ms

    Deprecated - see above

    Parameters
    ----------
    spikes: 1D arr
        1D array of spike times in ms

    times: 1D arr
        1D array of times when hd was recorded
        ix matches head_dirs

    head_dirs: 1D arr
        1D array of head direction in degrees (0-360)
        ix matches times
    
    match_freedom_ms: int
        default = 100 
    
    Returns
    -------
    spike_hds: 1D arr
        1D array of head directions for each spike
        len(spike_hds) should match len(spikes)
    """
    spike_hds = []

    for spike in spikes:

        idx = (np.abs(times - spike)).argmin()
        diff = np.abs(times[idx] - spike)

        if diff < match_freedom_ms:

            spike_hds.append(head_dirs[idx])

    return spike_hds


def calc_trial_frs(trials, tlen=1., average=True):
    """Calculate firing rates per trial."""

    fr_pre = [sum(trial < 0) / tlen for trial in trials]
    fr_post = [sum(trial > 0) / tlen for trial in trials]

    if average:
        fr_pre = np.mean(fr_pre)
        fr_post = np.mean(fr_post)

    return fr_pre, fr_post


def compute_bin_firing(x_binl, y_binl, bins):
    """Compute firing per bin, givin bin assignment of each spike."""

    bin_firing = np.zeros(bins)
    for x_bl, y_bl in zip(x_binl, y_binl):
        bin_firing[x_bl - 1, y_bl - 1] += 1

    return bin_firing


def bin_circular(degrees):
    """Bin circular data.

    Parameters
    ----------
    degrees : 1d array
        Data to bin.

    Returns
    -------
    bin_edges : 1d array
        Bin edge definitions.
    counts : 1d array
        Count values per bin.
    """

    bin_edges = np.arange(0, 370, 10)
    counts, _ = np.histogram(degrees, bins=bin_edges)

    return bin_edges, counts
