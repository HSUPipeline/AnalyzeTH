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


def get_spike_heading(spikes, times, head_dirs):
    """Get head direciton for spike times."""

    spike_hds = []

    for spike in spikes:

        idx = (np.abs(times - spike)).argmin()
        diff = np.abs(times[idx] - spike)

        if diff < 100:

            spike_hds.append(head_dirs[idx])

    return spike_hds


def calc_trial_frs(trials, tlen=1.):
    """Calculate firing rates per trial."""

    fr_pre = np.mean([sum(trial < 0) for trial in trials]) / tlen
    fr_post = np.mean([sum(trial > 0) for trial in trials]) / tlen

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
