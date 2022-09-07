"""Helper functions for creating reports."""

import numpy as np

from spiketools.measures import compute_firing_rate
from spiketools.utils.base import count_elements
from spiketools.utils.timestamps import convert_sec_to_min

###################################################################################################
###################################################################################################

## GROUP REPORTS

def create_group_info(summary):
    """Create a dictionary of group information."""

    group_info = {}

    group_info['n_subjs'] = len(set([el.split('_')[1] for el in summary['ids']]))
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

    strtemp = "{}: {:2d} trials ({:5.2f}% correct, {:5.2f} avg error), " \
                  "with {:3d}/{:3d} units (keep/total)"

    out = []
    for ind in range(len(summary['ids'])):
        out.append(strtemp.format(summary['ids'][ind],
                                  summary['n_trials'][ind],
                                  summary['correct'][ind],
                                  summary['error'][ind],
                                  summary['n_keep'][ind],
                                  summary['n_units'][ind]))

    return out

## SESSION REPORTS

def create_units_info(units):
    """Create a dictionary of units related information."""

    units_info = {}

    # Get units dataframe & select only the keep units
    units_df = units.to_dataframe()
    units_df = units_df[units_df.keep == True]

    units_info['n_units'] = len(units)
    units_info['n_keep'] = int(sum(units.keep[:]))
    units_info['n_unit_channels'] = len(set(units_df.channel))
    units_info['location_counts'] = count_elements(units_df.location)
    units_info['frs'] = [compute_firing_rate(spikes) for spikes in units_df.spike_times]

    return units_info


def create_units_str(subject_info):
    """Create a string representation of the subject / session information."""

    string = '\n'.join([
        'UNITS INFO',
        'Total # units:   {:10d}'.format(subject_info['n_units']),
        'Keep # units:    {:9d}'.format(subject_info['n_keep']),
        '# unit channels: {:5d}'.format(subject_info['n_unit_channels']),
    ])

    return string


def create_position_info(acquisition, bins):
    """Create a dictionary of position related information."""

    position_info = {}

    position_info['bins'] = bins
    position_info['area_range'] = \
        [acquisition['boundaries']['x_range'].data[:],
         acquisition['boundaries']['z_range'].data[:] + np.array([-10, 10])]
    position_info['chests'] = acquisition['stimuli']['chest_positions'].data[:].T

    return position_info


def create_position_str(position_info):
    """Create a string representation of position information."""

    string = '\n'.join([
        'POSITION INFO',
        'Position bins: {:2d}, {:2d}'.format(*position_info['bins']),
        'Median occupancy: {:2.4f}'.format(np.nanmedian(position_info['occupancy'])),
        'Min occupancy:  {:2.4f}'.format(np.nanmin(position_info['occupancy'])),
        'Max occupancy:  {:2.4f}'.format(np.nanmax(position_info['occupancy'])),
    ])

    return string


def create_behav_info(trials):
    """Create a dictionary of session behaviour information."""

    behav_info = {}

    behav_info['trials_start'] = trials.start_time[0]
    behav_info['trials_end'] = trials.stop_time[-1]
    behav_info['session_length'] = float(convert_sec_to_min(behav_info['trials_end']))
    behav_info['n_trials'] = len(trials)
    behav_info['n_chests'] = int(sum(trials.n_chests.data[:]))
    behav_info['n_items'] = int(sum(trials.n_treasures.data[:]))
    behav_info['%_correct'] = float(np.mean(trials.correct[:]) * 100)
    behav_info['avg_error'] = float(np.mean(trials.error.data[:]))
    behav_info['confidence_counts'] = \
        count_elements(trials.confidence_response.data[:], labels=['yes', 'maybe', 'no'])

    return behav_info


def create_behav_str(behav_info):
    """Create a string representation of behavioural performance."""

    string = '\n'.join([
        'BEHAVIOR INFO',
        'Number of trials: {}'.format(str(behav_info['n_trials'])),
        'Session length: {:.2f}'.format(behav_info['session_length']),
        'Number of chests: {}'.format(str(behav_info['n_chests'])),
        'Number of items: {}'.format(str(behav_info['n_items'])),
        'Correct recall : {:5.2f}%'.format(behav_info['%_correct']),
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
