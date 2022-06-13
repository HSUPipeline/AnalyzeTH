""""Analysis functions for TH analysis."""

import numpy as np

###################################################################################################
###################################################################################################

def get_spike_positions(spikes, times, positions, threshold=0.25):
    """Get xy-positions for spike times."""

    spike_xs = []
    spike_ys = []

    for spike in spikes:

        idx = (np.abs(times - spike)).argmin()
        diff = np.abs(times[idx] - spike)

        if diff < threshold:

            spike_xs.append(positions[0, idx])
            spike_ys.append(positions[1, idx])

    return spike_xs, spike_ys


def get_spike_heading(spikes, times, head_dirs, threshold=0.25):
    """Get head direciton for spike times."""

    spike_hds = []

    for spike in spikes:

        idx = (np.abs(times - spike)).argmin()
        diff = np.abs(times[idx] - spike)

        if diff < threshold:

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
