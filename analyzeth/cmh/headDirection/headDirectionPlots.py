
from matplotlib import style
import numpy as np
import pandas as pd
from pingouin import convert_angles, circ_rayleigh

from analyzeth.cmh.utils.subset_data import subset_period_event_time_data, subset_period_data
from analyzeth.cmh.headDirection.headDirectionUtils import *
from analyzeth.analysis import get_spike_heading, bin_circular
from analyzeth.plts import plot_polar_hist

from spiketools.utils import restrict_range


# Plots
import matplotlib.pyplot as plt 
import seaborn as sns 
import analyzeth.cmh.settings_plots as PLOTSETTINGS
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# Analysis settings
import analyzeth.cmh.settings_analysis as SETTINGS

def plot_hd(data, bin_edges = [], ax=None):
    if not ax:
        ax = plt.subplot(111, polar=True)
    
    #if len(bin_edges) == 0:
    #    bin_edges = np.radians(np.arange(0,370,10))
    binsize = 360/len(data)
    bin_edges = np.radians(np.arange(0,361,binsize))

    max_datum = max(data)
    width = 2 * np.pi / (len(bin_edges)-1)

    ax.bar(bin_edges[:-1], data, alpha = 0.5, width=width, align = 'edge', edgecolor = 'none' )
    ax.set_rorigin(-1 * max_datum/5)

    #rad_edges = np.radians(bin_edges)
    #for xcoord in bin_edges:
    #    plt.axvline(x=xcoord, linestyle= '--')
    return ax


def plot_surrogates_95ci(surrogate_histograms, ax = None):
    """
    Plot polar line, surrogates 95ci 
    """
    if not ax:
        ax = plt.subplot(111, polar = True)
    
    binsize = surrogate_histograms.shape[1] / 360
    print(f'binsize = {binsize}')
    
    df = pd.DataFrame(surrogate_histograms).melt()
    df['variable'] = np.radians(df['variable']*binsize)
    sns.lineplot(data = df, x='variable', y = 'value', estimator=np.mean, ci=95, linewidth=0, color = 'r', alpha = 0.9)
    return ax

def plot_significant_bin_asterisks(significant_bins, asterisk_y = 5, ax = None):
    """
    add asterisks to plot for bins deemed significant (above ci95 for 10 bin stretch)
    """
    if ax == None:
        ax = plt.subplot(111, polar = True)

    sig_asterisks = significant_bins * asterisk_y
    sig_asterisks[sig_asterisks == 0] = np.nan
    x = np.radians(np.arange(0,360,1))
    sns.scatterplot(x=x, y = sig_asterisks, color = 'k')
    return ax

def plot_hd_full(hd_histogram, surrogates_ci95, significant_bins, surrogates = None, ax = None, title = ''):
    """
    Plot complete hd polar diagram with 95% confidence interval and significance
    asterisks
    """
    if not ax:
        fig = plt.figure(figsize = [10,10])
        ax = plt.subplot(111,polar=True)
    
    # Plot hd hist and ci95
    plot_hd(hd_histogram, ax = ax)
    
    # Plot ci95 - using mine rather than sns default for looks
        #plot_surrogates_95ci(surrogates, ax = ax)
    num_bins = len(hd_histogram)
    binsize = 360 / num_bins
    print('numbins', num_bins)
    x = np.radians(np.arange(0,num_bins, binsize))
    
    print('lenx', len(x))
    print('ci95', surrogates_ci95.shape) 



    sns.lineplot(x = x, y = surrogates_ci95[:,0], color='y')
    sns.lineplot(x=x, y = surrogates_ci95[:,1],color='g')
    
    # Plot significance markers
    max_fr = max(hd_histogram)
    asterisk_y = max_fr + 0.25*max_fr
    plot_significant_bin_asterisks(significant_bins, asterisk_y, ax = ax)

    # Formatting
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

    return ax

def plot_polar_hist_overlay_surrs(hd_degrees, mean_shuffle_hd_counts, ax=None):
    """Plot a polar histogram.

    Parameters
    ----------
    degrees : 1d array
        Data to plot in a circular histogram.
    """

    if not ax:
        ax = plt.subplot(111, polar=True)

    bin_edges, counts = bin_circular(hd_degrees)
    ax.bar(bin_edges[:-1], counts)
    
    #surr_edges, surr_counts = bin_circular(mean_surrogate_hd_degrees)
    #ax.bar(surr_edges[:-1], surr_counts, color = 'k', alpha = 0.4)
    ax.bar(bin_edges[:-1], mean_shuffle_hd_counts)

    return ax


def plot_hd_raster(
        nwbfile = None, 
        unit_ix = SETTINGS.UNIT_IX,
        ax = None,
        highlight = 'navigation'
        ):
    
    # -- LOAD -- 
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder) 

    # -- SESSION DATA -- 
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    

    # -- TRIAL DATA -- 
    trial_starts = (nwbfile.trials['start_time'].data[:])   #/1e3       #convert to trial time in s
    trial_ends = (nwbfile.trials['stop_time'].data[:])  #/1e3          #convert to trial time in s

    # -- NAVIGATION DATA --
    navigation_start_times = nwbfile.trials['navigation_start'][:]  #/1e3
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]    #/1e3
    
    # -- SPIKE DATA --
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)                            #get spikes in ms
    spikes = restrict_range(spikes, session_start, session_end)
    spikes = (spikes)   #/1e3                                          #convert to trial time in s  

    # -- HEAD DIRECTION DATA --
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = (head_direction.timestamps[:])   #/1e3                   #convert to trial time in s


    # -- PLOT --
    # if not ax:
    #     ax = plt.subplot(111)
    fig, ax = plt.subplots(figsize = [14, 5])

    # Add trial colors
    colors = ['r','y','b'] * 10
    if highlight == 'trial':
        for ix in range(len(trial_starts)):
            ax.axvspan(trial_starts[ix], trial_ends[ix], alpha=0.2, facecolor=colors[ix])

    else:
        for ix in range(len(navigation_start_times)):
            ax.axvspan(navigation_start_times[ix], navigation_stop_times[ix], alpha=0.2, facecolor=colors[ix])

    # Add events
    ax.eventplot([spikes, hd_times], linelengths = [0.9, 0.9], colors = ['g', 'b'])
    
    # Format
    ax.set_yticks([0,1])
    ax.set_yticklabels(['Spike Times', 'HD Times'])
    ax.set_xlabel('Time (s)')
    ax.set_title('Unit {} Raster Plot'.format(unit_ix))

    # Show plot
    plt.show()

    return fig, ax

def plot_line_hd_navigation(nwbfile):
    
    # Navigation period data
    navigation_start_times = nwbfile.trials['navigation_start'][:]  #/1e3
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]    #/1e3

    # Head direction data
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]     #/ 1e3    
    hd_degrees = head_direction.data[:]
    
    # Get nav data
    hd_times_nav = subset_period_event_time_data(hd_times, navigation_start_times, navigation_stop_times)
    hd_degrees_nav = subset_period_data(hd_degrees, hd_times, navigation_start_times, navigation_stop_times)

    # Plot
    fig, ax = plt.subplots(figsize = [14, 5])
    ax.plot(hd_degrees_nav)

    return fig, ax

def plot_line_hd(nwbfile, unit_ix = None):
    
    # Navigation period data
    navigation_start_times = nwbfile.trials['navigation_start'][:]  #/1e3
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]    #/1e3

    # Head direction data
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]     #/ 1e3    
    hd_degrees = head_direction.data[:]

    # -- PLOT --
    fig, ax = plt.subplots(figsize = [25, 5])

    # plot navigation periods
    colors = ['r','y','b'] * 10
    for ix in range(len(navigation_start_times)):
            ax.axvspan(navigation_start_times[ix], navigation_stop_times[ix], alpha=0.2, facecolor=colors[ix])

    # plot hd
    ax.plot(hd_times, hd_degrees, marker ='o', markerfacecolor='g', markeredgecolor='g')
    #ax.scatter(hd_times, hd_degrees)

    if unit_ix != None:
        spikes = nwbfile.units.get_unit_spike_times(unit_ix)    #/ 1e3
        ax.plot(spikes, len(spikes) * [365], 'rv')

    return fig, ax


def plot_hd_occupancy_vs_spike_probability_overlay(hd_hist, occupancy_hist, ax = None, figsize = [10,10]):
    """ Plot overlay of two polar histograms"""
    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar = True)
    
    
    hist1 = hist1/sum(hist1)
    hist2 = hist2/sum(hist2)
    plot_hd(hist1, ax=ax)
    plot_hd(hist2, ax=ax)
    plt.title('Overlay')
    plt.show()
    return ax


def plot_box_hd_vs_shuffle(hd_histogram, surrogate_hd_histograms, ax=None):
    """
    Plot box plot of surrogates vs sample HD with 95ci superimposed
    """
    if not ax:
        fig, ax = plt.subplots(figsize=[15,8])

    sns.boxplot(data = np.array(surrogate_hd_histograms), color = 'lightgray', notch= True, saturation=0.5)
    sns.pointplot(data = np.array(surrogate_hd_histograms), color = 'r', join = False, ci=95)
    sns.scatterplot(data = hd_histogram, color = 'b', s = 100)
    xlabels = np.arange(0,360,10)
    ax.set_xticklabels(xlabels, rotation = 90)
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlabel('Degrees')
    plt.show()
    return ax


def plot_bootstrap_replicates_PDF_hist(bootstrap_replicates, bins=30, density=1, ax = None):
    """
    Plot probability density function for bootstraps showing 95% confidence interval
    """
    # if ax == None:
    #     ax = plt.subplots(111)

    # Plot the PDF for bootstrap replicates as histogram
    plt.hist(bootstrap_replicates,bins=30,density=1, stacked=True)
    
    # Showing the related percentiles
    plt.axvline(x=np.percentile(bootstrap_replicates,[2.5]), ymin=0, ymax=1,label='2.5th percentile',c='y')
    plt.axvline(x=np.percentile(bootstrap_replicates,[97.5]), ymin=0, ymax=1,label='97.5th percentile',c='r')
    
    # Formatting
    plt.xlabel("Hz")
    plt.ylabel("PDF")
    plt.title("Probability Density Function")
    plt.legend()
    plt.show()

    return ax