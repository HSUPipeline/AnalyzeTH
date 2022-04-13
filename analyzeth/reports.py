"""Helper functions for creating reports."""

import numpy as np

from spiketools.measures import compute_spike_rate

###################################################################################################
###################################################################################################

def create_group_info(summary):
    """Create a dictionary of group information."""

    group_info = {}

    group_info['n_subjs'] = len(set([el.split('-')[0] for el in summary['ids']]))
    group_info['n_sessions'] = len(summary['ids'])

    return group_info


def create_group_str(group_info):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'Number of subjects:  {}'.format(group_info['n_subjs']),
        'Number of sessions:  {}'.format(group_info['n_sessions']),
    ])

    return string


def create_group_sessions_str(summary):
    """Create strings of detailed session information."""

    out = []
    strtemp = "{}: {:3d} units, {:3d} trials ({:5.2f}% correct, average error of {:5.2f})"
    for ind in range(len(summary['ids'])):
        out.append(strtemp.format(summary['ids'][ind], summary['n_units'][ind],
                                  summary['n_trials'][ind], summary['correct'][ind],
                                  summary['error'][ind]))

    return out


def create_subject_info(nwbfile):
    """Create a dictionary of subject information."""

    subject_info = {}

    subject_info['n_units'] = len(nwbfile.units)
    subject_info['subject_id'] = nwbfile.subject.subject_id
    subject_info['session_id'] = nwbfile.session_id
    subject_info['trials_start'] = nwbfile.intervals['trials'][0]['start_time'].values[0]
    subject_info['trials_end'] = nwbfile.intervals['trials'][-1]['stop_time'].values[0]

    return subject_info


def create_subject_str(subject_info):
    """Create a string representation of the subject / session information."""

    string = '\n'.join([
        'Recording:  {:5s}'.format(subject_info['session_id']),
        'Number of units:    {:10d}'.format(subject_info['n_units']),
        'Recording time range:',
        '{:5.4f} -\n {:5.4f}'.format(subject_info['trials_start'],
                                   subject_info['trials_end'])
    ])

    return string


def create_position_str(bins, occ):
    """Create a string representation of position information."""

    string = '\n'.join([
        'Position bins: {:2d}, {:2d}'.format(*bins),
        'Median occupancy: {:2.4f}'.format(np.nanmedian(occ)),
        'Min / Max occupancy:  {:2.4f}, {:2.4f}'.format(np.nanmin(occ), np.nanmax(occ))
    ])

    return string


def create_behav_info(nwbfile):
    """Create a dictionary of session behaviour information."""

    behav_info = {}

    behav_info['n_trials'] = len(nwbfile.trials)
    behav_info['n_chests'] = sum(nwbfile.trials.num_chests.data[:])
    behav_info['n_items'] = sum(nwbfile.trials.num_treasures.data[:])
    behav_info['error'] = np.mean(nwbfile.trials.error.data[:])

    return behav_info


def create_behav_str(behav_info):
    """Create a string representation of behavioural performance."""

    string = '\n'.join([
        'Number of trials: {}'.format(str(behav_info['n_trials'])),
        'Number of chests: {}'.format(str(behav_info['n_chests'])),
        'Number of items: {}'.format(str(behav_info['n_items'])),
        'Retrieval error : {:4.2f}'.format(behav_info['error']),
    ])

    return string


def create_unit_info(unit):
    """Create a dictionary of unit information."""

    spikes = unit['spike_times'].values[0] / 1000

    unit_info = {}
    unit_info['n_spikes'] = len(spikes)
    unit_info['spike_rate'] = compute_spike_rate(spikes) * 1000
    unit_info['first_spike'] = spikes[0]
    unit_info['last_spike'] = spikes[-1]
    unit_info['location'] = unit['location'].values[0]
    unit_info['channel'] = unit['channel'].values[0]
    unit_info['cluster'] = unit['cluster'].values[0]

    return unit_info


def create_unit_str(unit_info):
    """Create a string representation of the unit info."""

    string = '\n'.join([
        '\n',
        'Number of spikes:    {:10d}'.format(unit_info['n_spikes']),
        'Average spike rate:  {:10.4f}'.format(unit_info['spike_rate']),
        'Recording time range:',
        '{:5.4f} -\n {:5.4f}'.format(unit_info['first_spike'], unit_info['last_spike']),
        'Unit location: {}'.format(unit_info['location']),
        'Channel & Cluster: {} - {}'.format(unit_info['channel'], unit_info['cluster'])
    ])

    return string