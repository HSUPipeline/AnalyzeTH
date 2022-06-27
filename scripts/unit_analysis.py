"""Run TH analysis across all units."""

import warnings
import traceback

import numpy as np
from scipy.stats import sem, ttest_rel
import matplotlib.pyplot as plt
from matplotlib import gridspec

from convnwb.io import load_nwbfile

#from pingouin import convert_angles, circ_rayleigh

from convnwb.io import get_files, save_json, save_txt, file_in_list

from spiketools.measures import compute_isis
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.plts.spikes import plot_isis
from spiketools.plts.spatial import plot_positions, plot_heatmap
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_bar, plot_polar_hist, plot_text
from spiketools.plts.stats import plot_surrogates
from spiketools.plts.annotate import color_pval
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.spatial.occupancy import (compute_occupancy, compute_bin_edges,
                                          compute_bin_assignment, compute_bin_firing)
from spiketools.spatial.information import compute_spatial_information
from spiketools.utils.data import compute_range
from spiketools.utils.trials import (epoch_spikes_by_event, epoch_spikes_by_range,
                                     epoch_data_by_range)
from spiketools.utils.base import select_from_list

# Import settings from local file
from settings import (TASK, PATHS, IGNORE, UNIT_SETTINGS, METHOD_SETTINGS,
                      ANALYSIS_SETTINGS, SURROGATE_SETTINGS)

# Import local code
import sys
sys.path.append('../code')
from analysis import calc_trial_frs, get_spike_positions#, get_spike_heading
from place import get_trial_place, compute_place_bins, create_df_place, fit_anova_place
from target import compute_spatial_target_bins, get_trial_target, create_df_target, fit_anova_target
from serial import compute_serial_position_fr, create_df_serial, fit_anova_serial
from reports import create_unit_info, create_unit_str

###################################################################################################
###################################################################################################

def main():
    """Run analyses across all units."""

    print('\n\nANALYZING UNIT DATA - {}\n\n'.format(TASK))

    # Get the list of NWB files
    nwbfiles = get_files(PATHS['DATA'], select=TASK)

    # Get list of already generated and failed units, & drop file names
    output_files = get_files(PATHS['RESULTS'] / 'units' / TASK,
                             select='json', drop_extensions=True)
    failed_files = get_files(PATHS['RESULTS'] / 'units' / TASK / 'zFailed',
                             select='json', drop_extensions=True)

    # Define analyses to run (that will be surrogate tested)
    surr_analyses = ['place_info', 'place_anova',
                     'target_info', 'target_anova',
                     'sp_anova']

    for nwbfilename in nwbfiles:

        ## DATA LOADING

        # Check and ignore files set to ignore
        if file_in_list(nwbfilename, IGNORE):
            print('\nSkipping file (set to ignore): ', nwbfilename)
            continue

        # Print out status
        print('\nRunning unit analysis: ', nwbfilename)

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfile, PATHS['DATA'], return_io=True)

        ## GET DATA

        # Get start and stop time of trials
        trial_starts = nwbfile.trials['start_time'].data[:]
        trial_stops = nwbfile.trials['stop_time'].data[:]

        # Get trial indices of interest
        chest_trials = nwbfile.trials.chest_trials[:]
        chest_openings = nwbfile.trials['chest_opening_time'][:]

        # Get masks for full and empty chest trials
        full_mask = nwbfile.trials.full_chest.data[:]
        empty_mask = np.invert(full_mask)

        # Get the navigation time ranges
        nav_starts = nwbfile.trials.navigation_start[:]
        nav_stops = nwbfile.trials.navigation_stop[:]

        # Get area ranges, adding a buffer to the z-range (for tower transport)
        area_range = [nwbfile.acquisition['boundaries']['x_range'].data[:],
                      nwbfile.acquisition['boundaries']['z_range'].data[:] + np.array([-15, 15])]

        # Get the position data & speed data
        ptimes = nwbfile.acquisition['position']['player_position'].timestamps[:]
        positions = nwbfile.acquisition['position']['player_position'].data[:].T
        stimes = nwbfile.processing['position_measures']['speed'].timestamps[:]
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Get position data for navigation segments
        ptimes_trials, positions_trials = epoch_data_by_range(ptimes, positions, nav_starts, nav_stops)
        stimes_trials, speed_trials = epoch_data_by_range(stimes, speed, nav_starts, nav_stops)

        # Recombine position data across selected navigation trials
        ptimes = np.hstack(ptimes_trials)
        positions = np.hstack(positions_trials)
        stimes = np.hstack(stimes_trials)
        speed = np.hstack(speed_trials)

        # Extract head position data
        # hd_times = nwbfile.acquisition['heading']['direction'].timestamps[:]
        # hd_degrees = nwbfile.acquisition['heading']['direction'].data[:]

        # Get the chest positions
        chest_xs, chest_ys = nwbfile.acquisition['stimuli']['chest_positions'].data[:].T

        ## ANALYZE UNITS

        # Get unit information
        n_units = len(nwbfile.units)
        keep_inds = np.where(nwbfile.units.keep[:])[0]
        n_keep = len(keep_inds)

        # Loop across all units
        for unit_ind in keep_inds:

            # Initialize output unit file name & output dictionary
            name = nwbfile.session_id + '_U' + str(unit_ind).zfill(2)

            # Check if unit already run
            if UNIT_SETTINGS['SKIP_ALREADY_RUN'] and file_in_list(name, output_files):
                print('\tskipping unit (already run): \tU{:02d}'.format(unit_ind))
                continue

            if UNIT_SETTINGS['SKIP_FAILED'] and file_in_list(name, failed_files):
                print('\tskipping unit (failed): \tU{:02d}'.format(unit_ind))
                continue

            print('\trunning unit: \t\t\tU{:02d}'.format(unit_ind))

            try:

                # Collect information of interest
                unit_info = create_unit_info(nwbfile.units[unit_ind])

                # Extract spikes for a unit of interest
                spikes = nwbfile.units.get_unit_spike_times(unit_ind)

                # Initialize results and add unit metadata
                results = {}
                results['uid'] = int(unit_ind)
                results['session_id'] = nwbfile.session_id
                results['subject_id'] = nwbfile.subject.subject_id
                for field in ['wvID', 'keep', 'n_spikes', 'firing_rate'
                              'cluster', 'channel', 'location']:
                    results[field] = unit_info[field]

                ## Compute measures

                # Get the spiking data for each trial
                all_trials = epoch_spikes_by_range(spikes, trial_starts, trial_stops, reset=True)

                # Compute firing related to chest presentation
                all_chests = epoch_spikes_by_event(spikes, np.concatenate(chest_openings),
                                                   ANALYSIS_SETTINGS['TRIAL_RANGE'])
                empty_trials = select_from_list(all_chests, empty_mask)
                full_trials = select_from_list(all_chests, full_mask)

                # Calculate firing rate pre & post chest opening
                fr_pre_all, fr_post_all = calc_trial_frs(all_chests)
                fr_pre_empt, fr_post_empt = calc_trial_frs(empty_trials)
                fr_pre_full, fr_post_full = calc_trial_frs(full_trials)

                # Compute bin edges
                x_bin_edges, y_bin_edges = compute_bin_edges(\
                    positions, ANALYSIS_SETTINGS['PLACE_BINS'], area_range=area_range)

                # Get position values for each spike
                spike_xs, spike_ys = get_spike_positions(spikes, ptimes, positions)
                spike_positions = np.array([spike_xs, spike_ys])

                # Compute occupancy
                occ_kwargs = {'minimum' : ANALYSIS_SETTINGS['MIN_OCCUPANCY'],
                              'area_range' : area_range, 'set_nan' : True}
                occ = compute_occupancy(positions, ptimes,
                                        ANALYSIS_SETTINGS['PLACE_BINS'],
                                        speed, **occ_kwargs)

                # Compute spatial bin assignments & binned firing, and normalize by occupancy
                x_binl, y_binl = compute_bin_assignment(spike_positions, x_bin_edges, y_bin_edges)
                bin_firing = compute_bin_firing(ANALYSIS_SETTINGS['PLACE_BINS'], x_binl, y_binl)
                bin_firing = bin_firing / occ

                # Get head direction for each spike
                #spike_hds = get_spike_heading(spikes, hd_times, hd_degrees)

                # Compute edges for chest binning
                ch_x_edges, ch_y_edges = compute_bin_edges(\
                    positions, ANALYSIS_SETTINGS['CHEST_BINS'], area_range=area_range)

                # Assign each chest to a bin
                chest_pos = np.array([chest_xs, chest_ys])
                ch_xbin, ch_ybin = compute_bin_assignment(chest_pos, ch_x_edges, ch_y_edges)

                # Fix offset of chest binning
                ch_xbin = ch_xbin - 1
                ch_ybin = ch_ybin - 1

                # Compute chest occupancy
                chest_occupancy = compute_bin_firing(ANALYSIS_SETTINGS['CHEST_BINS'], ch_xbin, ch_ybin)

                ## STATISTICS

                # Compute t-tests for chest related firing
                results['fr_t_val_all'], results['fr_p_val_all'] = \
                    ttest_rel(*calc_trial_frs(all_chests, average=False))
                results['fr_t_val_empt'], results['fr_p_val_empt'] = \
                    ttest_rel(*calc_trial_frs(full_trials, average=False))
                results['fr_t_val_full'], results['fr_p_val_full'] = \
                    ttest_rel(*calc_trial_frs(empty_trials, average=False))

                # Place cell analysis
                if METHOD_SETTINGS['PLACE'] == 'INFO':
                    results['place_info'] = compute_spatial_information(bin_firing, occ)
                if METHOD_SETTINGS['PLACE'] == 'ANOVA':
                    place_trial = get_trial_place(spikes, nwbfile.trials, ANALYSIS_SETTINGS['PLACE_BINS'],
                                                  ptimes, positions, speed, x_bin_edges, y_bin_edges, occ_kwargs)
                    results['place_anova'] = fit_anova_place(create_df_place(place_trial, drop_na=True))

                # Target cell analysis
                target_bins = compute_spatial_target_bins(\
                    spikes, nav_starts, chest_openings, chest_trials, ptimes, positions,
                    ANALYSIS_SETTINGS['CHEST_BINS'], ch_xbin, ch_ybin)
                if METHOD_SETTINGS['TARGET'] == 'INFO':
                    results['target_info'] = compute_spatial_information(target_bins, chest_occupancy)

                if METHOD_SETTINGS['TARGET'] == 'ANOVA':
                    target_trial = get_trial_target(spikes, nav_starts, ANALYSIS_SETTINGS['CHEST_BINS'],
                                                    chest_openings, chest_trials, ch_xbin, ch_ybin, ptimes, positions)
                    results['target_anova'] = fit_anova_target(create_df_target(target_trial))

                # Serial position analysis
                sp_all_frs = compute_serial_position_fr(spikes, nav_starts, chest_openings, chest_trials)
                results['sp_anova'] = fit_anova_serial(create_df_serial(sp_all_frs))

                # Compute measures for head direction
                # results['hd_zstat'], results['hd_pstat']  = circ_rayleigh(convert_angles(spike_hds))

                ## SURROGATES

                # Create shuffled time series for comparison
                times_shuffle = shuffle_spikes(spikes,
                                               SURROGATE_SETTINGS['SHUFFLE_APPROACH'],
                                               SURROGATE_SETTINGS['N_SURROGATES'])

                # Initialize to store surrogate analysis outputs
                surrs = {analysis : np.zeros(SURROGATE_SETTINGS['N_SURROGATES']) for analysis in surr_analyses}

                for ind, shuffle in enumerate(times_shuffle):

                    # PLACE
                    if METHOD_SETTINGS['PLACE'] == 'INFO':
                        surr_bin_firing = compute_place_bins(stimes, bins, ptimes, positions, x_bin_edges, y_bin_edges)
                        surrs['place_info'][ind] = compute_spatial_information(bin_firing, occ, normalize=True)
                    if METHOD_SETTINGS['PLACE'] == 'ANOVA':
                        s_place_trial = get_trial_place(shuffle, nwbfile.trials, ANALYSIS_SETTINGS['PLACE_BINS'],
                                                        ptimes, positions, speed, x_bin_edges, y_bin_edges, occ_kwargs)
                        surrs['place_anova'][ind] = fit_anova_place(create_df_place(s_place_trial, drop_na=True))

                    # TARGET
                    if METHOD_SETTINGS['TARGET'] == 'INFO':
                        s_target_bins = compute_spatial_target_bins(shuffle, nav_starts, chest_openings, chest_trials,
                                                                    ptimes, positions, ANALYSIS_SETTINGS['CHEST_BINS'],
                                                                    ch_xbin, ch_ybin)
                        surrs['target_info'][ind] = compute_spatial_information(s_target_bins, chest_occupancy)
                    if METHOD_SETTINGS['TARGET'] == 'ANOVA':
                        s_target_trial = get_trial_target(shuffle, nav_starts, ANALYSIS_SETTINGS['CHEST_BINS'],
                                                          chest_openings, chest_trials, ch_xbin, ch_ybin, ptimes, positions)
                        surrs['target_anova'][ind] = fit_anova_target(create_df_target(s_target_trial))

                    # SERIAL POSITION
                    s_sp_all_frs = compute_serial_position_fr(shuffle, nav_starts, chest_openings, chest_trials)
                    surrs['sp_anova'][ind] = fit_anova_serial(create_df_serial(s_sp_all_frs))

                    # HEAD DIRECTION
                    #s_spike_hds = get_spike_heading(shuffle, hd_times, hd_degrees)
                    #surrs['hd_stat'][ind] = circ_rayleigh(convert_angles(s_spike_hds))[0]

                # Compute surrogate statistics
                for analysis in surr_analyses:
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                        compute_surrogate_stats(results[analysis], surrs[analysis])

                # Save out unit results
                save_json(results, name + '.json', folder=str(PATHS['RESULTS'] / 'units' / TASK))

                ### MAKE REPORT

                # Initialize figure
                _ = plt.figure(figsize=(15, 18))
                grid = gridspec.GridSpec(6, 3, wspace=0.4, hspace=1.)

                # 00: plot rasters across all trials
                ax00 = plt.subplot(grid[0, 0])
                plot_rasters(all_trials, ax=ax00, title='All Trials')
                # 01: unit information
                ax01 = plt.subplot(grid[0, 1])
                plot_text(create_unit_str(unit_info), ax=ax01)
                ax01.set_title("Unit Information", fontdict={'fontsize' : 16}, y=1.2)

                # 02: inter-spike intervals
                ax02 = plt.subplot(grid[0, 2])
                isis = compute_isis(spikes)
                plot_isis(isis, bins=100, range=(0, 2), ax=ax02)

                # 10: chest related firing
                title_str = '{} - Pre: {:1.2f} - Pos: {:1.2f}  (t:{:1.2f}, p:{:1.2f})'
                ax10 = plt.subplot(grid[1:3, 0:2])
                plot_rasters(all_chests, xlim=ANALYSIS_SETTINGS['TRIAL_RANGE'],
                             vline=0, figsize=(10, 7), ax=ax10)
                ax10.set_title(title_str.format('All Chests', fr_pre_all, fr_post_all, fr_t_val_all, results['fr_p_val_all']),
                               color=color_pval(results['fr_p_val_all']))

                # 12&22: Compare Empty & Full chest trials
                # Empty chest trials
                ax12 = plt.subplot(grid[1, 2])
                plot_rasters(empty_trials, xlim=ANALYSIS_SETTINGS['TRIAL_RANGE'], vline=0, ax=ax12)
                ax12.set_title(title_str.format('Empty', fr_pre_empt, fr_post_empt, fr_t_val_empt, results['fr_p_val_empt']),
                               color=color_pval(results['fr_p_val_empt']), fontdict={'fontsize' : 14})

                # Full chest trials
                ax22 = plt.subplot(grid[2, 2])
                plot_rasters(full_trials, xlim=ANALYSIS_SETTINGS['TRIAL_RANGE'], vline=0, ax=ax22)
                ax22.set_title(title_str.format('Full', fr_pre_full, fr_post_full, fr_t_val_full, results['fr_p_val_full']),
                               color=color_pval(results['fr_p_val_full']), fontdict={'fontsize' : 14})

                # ax30: positional firing
                ax30 = plt.subplot(grid[3:5, 0])
                plot_positions(positions, spike_positions,
                               x_bins=x_bin_edges, y_bins=y_bin_edges, ax=ax30,
                               title='Firing Across Positions')

                # ax31: positional heatmap
                ax31 = plt.subplot(grid[3:5, 1])
                plot_heatmap(bin_firing, transpose=True, ax=ax31,
                             title='Range: {:1.2f}-{:1.2f}'.format(*compute_range(bin_firing)))

                # # ax31: head direction of spike firing
                # ax32 = plt.subplot(grid[3, 2], polar=True)
                # plot_polar_hist(spike_hds, ax=ax32)
                # ax32.set(xticklabels=[], yticklabels=[])
                # ax32.set_title('Head Direction', color=color_pval(hd_p_val))

                # ax42: place surrogates
                ax42 = plt.subplot(grid[4, 2])
                #plot_surrogates(surrs['place_info'], results['place_info'],
                #                results['place_info_surr_p_val'], ax=ax42)
                plot_surrogates(surrs['place_anova'], results['place_anova'],
                                results['place_anova_surr_p_val'], ax=ax42)
                ax42.set_title('Place Surrogates', color=color_pval(place_p_val))

                # ax50: firing rates across trial segments
                ax50 = plt.subplot(grid[5, 0])
                plot_bar(sp_all_frs.mean(0), [0, 1, 2, 3], yerr=sem(sp_all_frs, 0), ax=ax50)
                ax50.set_title('Serial Position', color=color_pval(sp_p_val))

                # ax51: spatial target firing
                ax51 = plt.subplot(grid[5, 1])
                plot_heatmap(target_bins, transpose=True, ax=ax51,
                             title='Range: {:1.2f}-{:1.2f}'.format(*compute_range(target_bins)))

                # ax52: target surrogates
                ax52 = plt.subplot(grid[5, 2])
                #plot_surrogates(surrs['target_info'], results['target_info'],
                #                results['target_info_surr_p_val'], ax=ax52)
                plot_surrogates(surrs['target_anova'], results['target_anova'],
                                results['target_anova_surr_p_val'], ax=ax52)

                ax52.set_title('Target Surrogates', color=color_pval(target_p_val))

                # Add super title to the report
                suptitle = 'Unit Report: {}-U{}'.format(nwbfile.session_id, unit_ind)
                plt.suptitle(suptitle, fontsize=24, y=0.95);

                # Save out report
                report_name = 'unit_report_' + name + '.pdf'
                plt.savefig(PATHS['REPORTS'] / 'units' / TASK / report_name)
                plt.close()

            except Exception as excp:
                if not UNIT_SETTINGS['CONTINUE_ON_FAIL']:
                    raise
                print('\t\tissue running unit # {}'.format(unit_ind))
                save_txt(traceback.format_exc(), name, folder=str(PATHS['RESULTS'] / 'units' / TASK / 'zFailed'))

        # Close the nwbfile
        io.close()

    print('\n\nCOMPLETED UNIT ANALYSES\n\n')

if __name__ == '__main__':
    main()
