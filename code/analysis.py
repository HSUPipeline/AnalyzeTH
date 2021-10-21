""""Analysis functions for TH analysis."""

import numpy as np

###################################################################################################
###################################################################################################

def get_spike_positions(spikes, ptimes, positions):
    """Get xy-positions for spike times."""

    tspike, tpos = [], []
    spike_xs = []
    spike_ys = []
    inds = []

    for spike in spikes:

        idx = (np.abs(ptimes - spike)).argmin()
        diff = np.abs(ptimes[idx] - spike)

        if diff < 100:

            inds.append(idx)

            spike_xs.append(positions[0, idx])
            spike_ys.append(positions[1, idx])

            tspike.append(spike)
            tpos.append(ptimes[idx])

    return spike_xs, spike_ys
