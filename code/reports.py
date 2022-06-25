"""Helper functions for creating reports."""

import numpy as np

from spiketools.measures import compute_firing_rate
from spiketools.utils.timestamps import convert_sec_to_min

###################################################################################################
###################################################################################################

## GROUP REPORTS

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
    strtemp = "{} ({:3d} trials): {:3d} keep units ({:3d} total), ({:5.2f}% correct, avg error: {:5.2f})"
    for ind in range(len(summary['ids'])):
        out.append(strtemp.format(summary['ids'][ind], summary['n_trials'][ind],
                                  summary['n_keep'][ind], summary['n_units'][ind],
                                  summary['correct'][ind], summary['error'][ind]))

    return out

## SESSION REPORTS

def create_subject_info(nwbfile):
    """Create a dictionary of subject information."""

    subject_info = {}

    st = nwbfile.intervals['trials'][0]['start_time'].values[0]
    en = nwbfile.intervals['trials'][-1]['stop_time'].values[0]

    subject_info['n_units'] = len(nwbfile.units)
    subject_info['n_keep'] = sum(nwbfile.units.keep[:])
    subject_info['subject_id'] = nwbfile.subject.subject_id
    subject_info['session_id'] = nwbfile.session_id
    subject_info['trials_start'] = st
    subject_info['trials_end'] = en
    subject_info['session_length'] = float(convert_sec_to_min(en))

    return subject_info


def create_subject_str(subject_info):
    """Create a string representation of the subject / session information."""

    string = '\n'.join([
        'Recording:  {:5s}'.format(subject_info['session_id']),
        'Total # units:   {:10d}'.format(subject_info['n_units']),
        'Keep # units:    {:10d}'.format(subject_info['n_keep']),
        'Session length:     {:.2f}'.format(subject_info['length'])
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
    behav_info['n_chests'] = int(sum(nwbfile.trials.n_chests.data[:]))
    behav_info['n_items'] = int(sum(nwbfile.trials.n_treasures.data[:]))
    behav_info['avg_error'] = np.mean(nwbfile.trials.error.data[:])

    return behav_info


def create_behav_str(behav_info):
    """Create a string representation of behavioural performance."""

    string = '\n'.join([
        'Number of trials: {}'.format(str(behav_info['n_trials'])),
        'Number of chests: {}'.format(str(behav_info['n_chests'])),
        'Number of items: {}'.format(str(behav_info['n_items'])),
        'Retrieval error : {:4.2f}'.format(behav_info['avg_error']),
    ])

    return string


## UNIT REPORTS

def create_unit_info(unit):
    """Create a dictionary of unit information."""

    spikes = unit['spike_times'].values[0]

    unit_info = {}

    unit_info['wvID'] = int(unit['wvID'].values[0])
    unit_info['n_spikes'] = len(spikes)
    unit_info['firing_rate'] = float(compute_firing_rate(spikes))
    unit_info['first_spike'] = spikes[0]
    unit_info['last_spike'] = spikes[-1]
    unit_info['location'] = unit['location'].values[0]
    unit_info['channel'] = unit['channel'].values[0]
    unit_info['cluster'] = int(unit['cluster'].values[0])
    unit_info['keep'] = bool(unit['keep'].values[0])

    return unit_info


def create_unit_str(unit_info):
    """Create a string representation of the unit info."""

    string = '\n'.join([
        '\n',
        'WVID:    {}'.format(unit_info['wvID']),
        'Keep:    {}'.format(unit_info['keep']),
        '# spikes:   {:5d}'.format(unit_info['n_spikes']),
        'firing rate:  {:5.4f}'.format(unit_info['firing_rate']),
        'Location:   {} ({})'.format(unit_info['location'],
                                   unit_info['channel']),
        'Cluster:   {}'.format(unit_info['cluster'])
    ])

    return string
