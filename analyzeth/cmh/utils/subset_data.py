import numpy as np


def subset_period_event_time_data(event_times, period_start_times, period_end_times):
    """ Given 1D array of event times, subset only those times in periods of interest

    ex. all spikes within trial periods, all position times in navigation period
    """
    PETD = np.array([])
    for ix in range(len(period_start_times)):
        PETD = np.append(
                    PETD,
                    event_times[(event_times > period_start_times[ix]) \
                                & (event_times < period_end_times[ix])],   
                                axis = 0)
    return PETD

def subset_period_data(data, event_times, period_start_times, period_end_times):
    """ 
    Given 1D or 2D array of data (ex 1D head direction degrees, 2D position times) 
    and 1D array of matched event_times (ex times at which HD is recorded) with 
    matching indicies/length:

    Pull out data that falls in periods of interes (ex navigation periods)
    """
    period_ixs = np.array([], dtype=int)
    for ix in range(len(period_start_times)):
        period_ixs_ix = np.where((period_start_times[ix] < event_times) \
                                        & (event_times < period_end_times[ix]))
        period_ixs = np.append(period_ixs, period_ixs_ix[0])
    
    if data.ndim == 1:
        period_data = data[period_ixs]
    elif data.ndim == 2:
        period_data = data[:, period_ixs]
    else:
        raise ValueError('data must be 1D or 2D')
    
    return period_data