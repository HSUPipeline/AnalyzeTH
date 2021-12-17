"""Helper functions for creating session reports."""

import numpy as np

###################################################################################################
###################################################################################################

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
