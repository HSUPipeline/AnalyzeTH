
# Helper functions to make plots for time cell analysis. These
# should be called by using the analysis functions in timeCellAnalysis.py 


# Imports and format settings
import matplotlib.pyplot as plt
import seaborn as sns
import analyzeth.timeCells.settings_plots as PLOTSETTINGS 
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# -- HISTOGRAM PLOTS (Counts per bin) ---
def _plot_time_cell (bin_len, 
                     spikes_in_trial_time,
                     bins_in_trial_time,
                     spike_bin_counts,
                     title = '',
                     date='',
                     SAVEFIG = False
                     ):
    
    """Helper funciton for plotting time cell counts

    Can be used for single trial or sum of counts across all trials

    Note: this should only really be used by calling functions in timeCellAnalysis

    PARAMETERS
    ----------
    bin_len: int
        length of time bin (default 1000 ms)

    spikes_in_trial_time: array
        array of spike times normalized to start of trial (ms)

    bins_in_trial_time: array
        edges of bins normalized to start of trial (ms)

    spike_bin_counts: array
        counts of spikes within each bin

    title: str
        title for the plot
        used for saving and for plot title

    date: str
        date for saving, appended to start of figure name
        prefered format: 'yymmdd'
    
    SAVEFIG: bool
        whether or not to save the figure

    RETURNS
    -------
    fig, ax 

    """    
    # -- PLOT --
    if title == '':
        title = 'Time Response Singe Unit'
    
    # Plot spikes per bin 
    fig, ax = plt.subplots(1, 1, figsize = [10,5])
                            
    # -- CATCH NO SPIKES --
    if sum(spike_bin_counts) == 0:
        print ('\n -- NO SPIKES FOUND --')
        print (title)
        return fig, ax
    
    # Title
    plt.suptitle(title, x = 0.01, y = 1, ha = 'left', fontsize = 12)
    
    # Histogram
    ax.hist(spikes_in_trial_time, bins_in_trial_time) #, bins_in_trial_time);
    ax.set_ylim(0, max(spike_bin_counts)+1)
    ax.set_xlim(0, bins_in_trial_time[-1] + bin_len)
    ax.set_ylabel('Unit Spike Count (n)')
    ax.set_xlabel('Trial Time (ms)')
    ax.set_title('Spike Counts')

    # Add right label for spacing
    ax2 = ax.twinx()
    ax2.set_ylabel(' ')
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # Show
    plt.show()
    
    # Save
    if SAVEFIG:
        plt.savefig(date + '_' + title + '.pdf')
    
    return fig, ax

def _plot_trial_time_and_movement(trial_ix,
                                  bin_len, 
                                  spikes_in_trial_time,
                                  bins_in_trial_time,
                                  spike_bin_counts,
                                  t_pos, 
                                  t_spike_xs,
                                  t_spike_ys,
                                  ch_times,
                                  chest_xs,
                                  chest_ys,
                                  t_mask,
                                  title = '',
                                  date='',
                                  SAVEFIG = False
                                 ):
    
    """Helper funciton for plotting time cell counts with movement plot

    Used only for single trial (dont want to plot movement on all trials...)

    There are a lot of parameters and it might make sense to make this simpler, but 
    I don't want to recalclate everything from the anlysis functions to make the plot

    Note: this should only really be used by calling functions in timeCellAnalysis

    PARAMETERS
    ----------

    trial_ix: int
        trial number of interst

    bin_len: int
        length of time bin (default 1000 ms)

    spikes_in_trial_time: array
        array of spike times normalized to start of trial (ms)

    bins_in_trial_time: array
        edges of bins normalized to start of trial (ms)

    spike_bin_counts: array
        counts of spikes within each bin

    ... all the chest position stuff

    title: str
        title for the plot
        used for saving and for plot title

    date: str
        date for saving, appended to start of figure name
        prefered format: 'yymmdd'
    
    SAVEFIG: bool
        whether or not to save the figure

    RETURNS
    -------
    fig, ax 

    """ 
    
    
    # -- PLOT --
    if title == '':
        title = 'Time Bin & Movement Plot'
    

    
    # Plot spikes per bin (single trial)
    fig, axs = plt.subplots(1, 2, 
                            figsize = [10,5], 
                            gridspec_kw={'width_ratios': [3, 1]},
                            constrained_layout = True)
    
    # -- CATCH NO SPIKES --
    if sum(spike_bin_counts) == 0:
        print ('\n -- NO SPIKES FOUND --')
        print (title)
        return fig, axs
    
    # Title
    plt.suptitle(title, x = 0, y = 1.1, ha = 'left', fontsize = 12)
    
    # Left, histogram
    ax = axs[0]
    ax.hist(spikes_in_trial_time, bins_in_trial_time) #, bins_in_trial_time);
    ax.set_ylim(0, max(spike_bin_counts)+1)
    ax.set_xlim(0, bins_in_trial_time[-1] + bin_len)
    ax.set_ylabel('Unit Spike Count (n)')
    ax.set_xlabel('Trial Time (ms)')
    ax.set_title('Spike Counts | Trial {}'.format(trial_ix))


    # Right, movement
    ax = axs[1]
    ax.plot(*t_pos)
    ax.plot(*t_pos[:, 0], 'b.', ms=15)
    [ax.plot(*cht, 'y.', ms=20) for cht in ch_times];
    ax.plot(t_spike_xs, t_spike_ys, 'r.', ms=10)
    ax.plot(chest_xs[t_mask], chest_ys[t_mask], '.g', ms=50, alpha=0.25)
    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(' ')
    ax.set_title ('Movement')
    
    plt.show()
    
    if SAVEFIG:
        plt.savefig(date + '_' + title + '.pdf')
    
    return fig, axs



# ----------------------- FIRING RATE PLOTS ---------------------

def _plot_time_cell_FR (
                     bin_len, 
                     spikes_in_trial_time,
                     bins_in_trial_time,
                     spike_bin_counts,
                     firing_rates_per_bin,
                     title = '',
                     date='',
                     SAVEFIG = False
                     ):
    
    """Helper funciton for plotting time cell FIRING RATE

    Can be used for single trial or MEAN FR across all trials

    Note: this should only really be used by calling functions in timeCellAnalysis

    PARAMETERS
    ----------
    bin_len: int
        length of time bin (default 1000 ms)

    spikes_in_trial_time: array
        array of spike times normalized to start of trial (ms)

    bins_in_trial_time: array
        edges of bins normalized to start of trial (ms)

    spike_bin_counts: array
        counts of spikes within each bin

    title: str
        title for the plot
        used for saving and for plot title

    date: str
        date for saving, appended to start of figure name
        prefered format: 'yymmdd'
    
    SAVEFIG: bool
        whether or not to save the figure

    RETURNS
    -------
    fig, ax 

    """  
    # -- PLOT --
    if title == '':
        title = 'Time Response Singe Unit'
    
    # Plot spikes per bin 
    fig, ax = plt.subplots(1, 1, figsize = [10,5])
                            
    # -- CATCH NO SPIKES --
    if sum(firing_rates_per_bin) == 0:
        print ('\n -- NO SPIKES FOUND --')
        print (title)
        return fig, ax
    
    # Title
    plt.suptitle(title, x = 0.01, y = 1, ha = 'left', fontsize = 12)
    
    # FR Line Plot
    ax.plot(firing_rates_per_bin) 
    ax.set_ylim(0, max(firing_rates_per_bin)+1)
    # ax.set_xlim(0, bins_in_trial_time[-1] + bin_len)
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlabel('Trial Time (s)')
    ax.set_title('Firing Rate')

    # Add right label for spacing
    ax2 = ax.twinx()
    ax2.set_ylabel(' ')
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # Show
    plt.show()
    
    # Save
    if SAVEFIG:
        plt.savefig(date + '_' + title + '.pdf')
    
    return fig, ax

def _plot_trial_time_and_movement_FR (
                                  trial_ix,
                                  bin_len, 
                                  spikes_in_trial_time,
                                  bins_in_trial_time,
                                  spike_bin_counts,
                                  firing_rates_per_bin,
                                  t_pos, 
                                  t_spike_xs,
                                  t_spike_ys,
                                  ch_times,
                                  chest_xs,
                                  chest_ys,
                                  t_mask,
                                  title = '',
                                  date='',
                                  SAVEFIG = False
                                 ):

    """Helper funciton for plotting time cell FIRING RATE with movement plot

    Used only for single trial (dont want to plot movement on all trials...)

    There are a lot of parameters and it might make sense to make this simpler, but 
    I don't want to recalclate everything from the anlysis functions to make the plot

    Note: this should only really be used by calling functions in timeCellAnalysis

    PARAMETERS
    ----------

    trial_ix: int
        trial number of interst

    bin_len: int
        length of time bin (default 1000 ms)

    spikes_in_trial_time: array
        array of spike times normalized to start of trial (ms)

    bins_in_trial_time: array
        edges of bins normalized to start of trial (ms)

    spike_bin_counts: array
        counts of spikes within each bin

    ... all the chest position stuff

    title: str
        title for the plot
        used for saving and for plot title

    date: str
        date for saving, appended to start of figure name
        prefered format: 'yymmdd'
    
    SAVEFIG: bool
        whether or not to save the figure

    RETURNS
    -------
    fig, ax 
    """

    # -- PLOT --
    if title == '':
        title = 'Time Bin FR & Movement Plot'
    

    
    # Plot spikes per bin (single trial)
    fig, axs = plt.subplots(1, 2, 
                            figsize = [10,5], 
                            gridspec_kw={'width_ratios': [3, 1]},
                            constrained_layout = True)
    
    # -- CATCH NO SPIKES --
    if sum(spike_bin_counts) == 0:
        print ('\n -- NO SPIKES FOUND --')
        print (title)
        return fig, axs
    
    # Title
    plt.suptitle(title, x = 0, y = 1.1, ha = 'left', fontsize = 12)
    
    # FR Line Plot
    ax = axs[0]
    ax.plot(firing_rates_per_bin) 
    ax.set_ylim(0, max(firing_rates_per_bin)+1)
    # ax.set_xlim(0, bins_in_trial_time[-1] + bin_len)
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlabel('Trial Time (s)')
    ax.set_title('Firing Rate')


    # Right, movement
    ax = axs[1]
    ax.plot(*t_pos)
    ax.plot(*t_pos[:, 0], 'b.', ms=15)
    [ax.plot(*cht, 'y.', ms=20) for cht in ch_times];
    ax.plot(t_spike_xs, t_spike_ys, 'r.', ms=10)
    ax.plot(chest_xs[t_mask], chest_ys[t_mask], '.g', ms=50, alpha=0.25)
    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.yaxis.set_label_position("right")
    ax.set_ylabel(' ')
    ax.set_title ('Movement')
    
    plt.show()
    
    if SAVEFIG:
        plt.savefig(date + '_' + title + '.pdf')
    
    return fig, axs