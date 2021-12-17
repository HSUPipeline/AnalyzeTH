"""Helper functions for creating reports."""

from spiketools.measures import compute_spike_rate

###################################################################################################
###################################################################################################

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
