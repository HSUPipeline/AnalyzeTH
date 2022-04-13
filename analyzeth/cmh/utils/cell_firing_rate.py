import numpy as np

def cell_firing_rate(
        spikes,         #ms - may be s now 
        window = 100,   #ms 
        step = 10       #ms 
        ):

    """ Calculate firing rate across epoch 
    
    This function will take a 1D spike time array and calculate
    firing rate at each point in time (governed by step size) 
    with a rolling window (governed by window size)

    Edges are refelcted based on window size

    Parameters
    ----------
    spikes: 1D array
        time stamps of spikes in ms

    window: int 
        legth of window over which to calculate FR in ms

    step: int
        step size for moving window in ms

    Returns
    -------
    firing_rates: 1D array
        Hz (spikes per second)
        firing rates at each point in time
        length of array depends on step size

    """

    # Normalize spike time to start of spike triain
    spikes = spikes - spikes[0]
    
    # Length of epoch 
    len_epoch = spikes[-1] 

    # Reflect edges by window size
    window = int(window) if ((int(window) % 2) == 0) else int(window) + 1     # make window even
    start_reflection = np.flipud(spikes[spikes < window])
    end_reflection   = np.flipud(spikes[spikes > len_epoch - window])
    spikes = np.concatenate([start_reflection, spikes, end_reflection])

    # Iter 
    num_bins = int(np.ceil(len_epoch/step)) + 1 
    times = []
    FRs = []                                       
    for ix_win in range(num_bins):
        center = step * ix_win
        left = center - window/2
        right = center + window/2
        spikes_bin = spikes[(left<spikes) & (spikes < right)]
        fr_bin = len(spikes_bin)/(window/1e3) # get FR in Hz 
        times.append(center)
        FRs.append(fr_bin)

    # returning times in epoch time ms
    return FRs, times