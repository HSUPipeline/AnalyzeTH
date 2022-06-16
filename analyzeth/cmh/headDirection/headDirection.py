
#from random import shuffle


#from analyzeth.cmh.headDirection.headDirectionStats import *
#from analyzeth.cmh.headDirection.headDirectionUtils import *
#from analyzeth.cmh.headDirection.headDirectionPlots import *
#from analyzeth.analysis import get_spike_heading, bin_circular


# Local
from analyzeth.cmh.headDirection.headDirectionStats import nwb_shuffle_spikes_bincirc_navigation
from analyzeth.cmh.utils import *
from analyzeth.cmh.headDirection.headDirectionPlots import * 
from analyzeth.cmh.headDirection.headDirectionUtils import * 
from analyzeth.cmh.headDirection.headDirectionStats import * 
from analyzeth.cmh.utils.cell_firing_rate import *

# spiketools
from spiketools.stats.shuffle import shuffle_spikes

# General 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import os
from datetime import datetime
import glob

# Analysis Settings
import analyzeth.cmh.headDirection.settings_HD as SETTINGS

def nwb_headDirection_group(data_folder, settings = SETTINGS, plot=False, save_report = False, save_pickle = False, ix_break = np.inf):
    """
    Grep all .nwbs in dir and NOT subdirs and run analysis on all
    """

    files = glob.glob(data_folder +'*.nwb')
    
    df = None

    for ix, file_name in enumerate(files):
        print ('------ Starting Session ' + str(file_name) + ' -------')

        nwbfile = load_nwb(file_name)
        df_current = nwb_headDirection_session(nwbfile, settings, plot, save_report, save_pickle, ix_break)
        if ix == 0:
            df = df_current
        else:
            df.append(df_current)

    return df



def nwb_headDirection_session(nwbfile, settings = SETTINGS, plot=False, save_report = False, save_pickle = False, ix_break = np.inf):
    """"
    run and plot for each unit in session
    """

    ## OCCUPANCY
    # Compute     
    occupancy = compute_hd_occupancy(
        nwbfile, 
        return_hds = False, 
        smooth = True, 
        windowsize = SETTINGS.WINDOWSIZE, 
        binsize = SETTINGS.BINSIZE
    )

    # Verbose plot
    if plot:
        fig = plt.figure(figsize=[10,10])
        plot_hd(occupancy)
        plt.title('Occupancy (sec)')
        plt.xlabel('')
        plt.ylabel('')
        plt.show()

    ## RUN 
    # init dataframe & error list
    df = None
    error_list = []

    # Run across each unit
    n_units = len(nwbfile.units)
    for ix in range(n_units):

        # break for testing
        if ix > ix_break:
            break

        print (f'Working on unit {ix}...')
        try:
            # Run 
            res = nwb_headDirection_cell(
                nwbfile, 
                unit_ix = ix, 
                occupancy = occupancy,
                n_surrogates= SETTINGS.N_SURROGATES

            )

            # Plot & Save
            if save_report or save_pickle:
                save_folder = SETTINGS.SAVE_FOLDER
                save_subfolder = datetime.today().strftime('%Y%m%d_') + res['metadata']['session_id']
                save_dir = os.path.abspath(os.path.join(save_folder, save_subfolder))
                os.makedirs(save_dir, exist_ok=True)
        
            if save_report:
                fig = plot_headDirection_summary_PDF(nwbfile, res, occupancy)
                fig_name = res['metadata']['session_id'] + '_unit' + str(ix) + '.pdf'
                plt.savefig(os.path.join(save_dir, fig_name))
                plt.close()

            if save_pickle:
                pkl_name = res['metadata']['session_id'] + '_unit' + str(ix) + '.pickle'    
                fpath = os.path.join(save_dir, pkl_name)
                with open(fpath, 'wb') as handle:
                    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Build Dataframe
            if ix == 0:
                df = pd.json_normalize(res, sep='.')
            else:
                df_current = pd.json_normalize(res, sep = '.')
                df = df.append(df_current)
            

            error_list.append(False)

        except Exception as e:
            print('\t Error')
            print('\t', e)

            error_list.append(True)
    
    # for unit_ix in range(n_units):
    #     print(f'Working on unit {unit_ix}...')
    #     res = nwb_headDirection_cell(nwbfile, unit_ix, occupancy, n_surrogates = SETTINGS.N_SURROGATES, plot = plot)
    
    return df 


def nwb_headDirection_cell(nwbfile, unit_ix, 
        occupancy = [], 
        n_surrogates = 1000, 
        binsize = 1, 
        windowsize = 23, 
        smooth = True, 
        plot = False,
        ):
    """
    Run head direction analysis for unit and compare to surrogates
    """
    # Metadata -- CLEAN
    subject = nwbfile.subject.subject_id
    session_id = nwbfile.session_id
    n_trials = len(nwbfile.trials)
    n_units = len(nwbfile.units)
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    n_spikes_tot = len(spikes)
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_stop_times = nwbfile.trials['navigation_stop'][:]
    len_epochs = navigation_stop_times - navigation_start_times
    total_time = sum(len_epochs)
    spikes_navigation = subset_period_event_time_data(spikes, navigation_start_times, navigation_stop_times)
    n_spikes_navigation = len(spikes_navigation)
    
    # Firing rates
    firing_rates_over_time, mean_firing_rates_over_time = nwb_compute_navigation_firing_rates_over_time(nwbfile, unit_ix, return_means=True)
    mean_firing_rate = nwb_compute_navigation_mean_firing_rate(nwbfile, unit_ix)

    # Occupancy
    if occupancy == []:
        occupancy = compute_hd_occupancy(nwbfile, binsize, windowsize, smooth)
    
    # Head direction
    hd_hist = nwb_hd_cell_hist(nwbfile, unit_ix, binsize, windowsize, smooth)
    hd_score = compute_hd_score(hd_hist)
    hd_hist_norm = normalize_hd_hist_to_occupancy(hd_hist, occupancy)
    hd_norm_score = compute_hd_score(hd_hist)

    # Shuffle
    shuffled_spikes = nwb_shuffle_spikes_bincirc_navigation(nwbfile, unit_ix, n_surrogates, shift_range = [5, 20], verbose=False)
    head_direction = nwbfile.acquisition['heading']['direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]
    surrogate_hds = get_hd_shuffled_head_directions(shuffled_spikes, hd_times, hd_degrees)
    surrogate_histograms = get_hd_surrogate_histograms(surrogate_hds, binsize, windowsize, smooth)
    surrogates_norm = normalize_hd_surrogate_histograms_to_occupancy(surrogate_histograms, occupancy)
    
    # ci95
    #surrogates_ci95 = compute_pdf_ci95_from_surrogates(surrogates_norm)
    #surrogates_ci95 = bootstrap_ci_from_surrogates(surrogates_norm)                                               
    surrogates_ci95 = compute_std_ci95_from_surrogates(surrogates_norm)
    significant_bins = compute_significant_bins(hd_hist_norm, surrogates_ci95)

    hd_ax=None
    if plot:
        print(hd_hist_norm.shape)
        print(surrogates_ci95.shape)
        print(significant_bins.shape)

        title = session_id + f' | Unit {unit_ix}'
        hd_ax = plot_hd_full(hd_hist_norm, surrogates_ci95, significant_bins)

    res = {
        'metadata' : {
            'subject'                       : subject,
            'session_id'                    : session_id,
            'unit_ix'                       : unit_ix, 
            'nspikes'                       : n_spikes_tot,
            'nspikes_nav'                   : n_spikes_navigation,
            'navtime'                       : total_time,
            'n_units'                       : n_units,
            'n_trials'                      : n_trials,
            'n_surrogates'                  : n_surrogates,
            'hd_score'                      : hd_score,
            'hd_score_norm'                 : hd_norm_score
        },

        'occupancy': {
            'occupancy_norm'                : occupancy,
        },

        'head_direction' : {
            'hd_score'                      : hd_score,
            'hd_score_norm'                 : hd_norm_score,
            'hd_histogram'                  : hd_hist,
            'hd_histogram_norm'             : hd_hist_norm,
        },
        
        'surrogates' : {
            'surrogate_hds'                 : surrogate_hds,
            'surrogate_histograms'          : surrogate_histograms,
            'surrogate_histograms_norm'     : surrogates_norm,
            'surrogates_ci95'               : surrogates_ci95,
            'significant_bins'              : significant_bins,
        },
        
        'firing_rates' : {
            'firing_rates_over_time'        : firing_rates_over_time,
            'mean_firing_rates_over_time'   : mean_firing_rates_over_time,
            'mean_firing_rate'              : mean_firing_rate

        }
    }
    return res
        

######################
# Reports
######################

def headDirection_report(nwbfile, unit_ix):
    """
    Generate single page PDF report for unit
    """

    # Get metadata


    print (f'Analyzing Subject')










######################
# WIP
####################

# def compute_bootstrap_ci95_from_surrogates(surrogates):
#     """
#     Compute ci95 from surrogates using seaborn bootstrapping method
#     """

#     num_bins = surrogates.shape[1]
#     bisize = 360 / num_bins
#     df = pd.DataFrame(surrogates).melt()


#     df['variable'] = np.radians(df['variable']*binsize)
#     #ax = sns.lineplot


# def compare_hd_histogram_to_surrogates(hd_histogram, surrogates):
#     """
#     For each bin, compare the actual firing rate to the firing rates determined 
#     from shuffling for each. 
    
#     Bins in which the firing rate of the real histogram are above the 95% confidence 
#     interval of the surrogates are considered significant
#     """
    

#     ci95 = compute_ci95_from_surrogates(surrogates)
    

    


    


