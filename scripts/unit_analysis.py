"""Run TH analysis across all units."""

import warnings
import traceback

import numpy as np
from scipy.stats import sem

from convnwb.io import load_nwbfile

#from pingouin import convert_angles, circ_rayleigh

from convnwb.io import get_files, save_json, save_txt, file_in_list

from spiketools.measures import compute_isis
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.stats.trials import compare_pre_post_activity
from spiketools.plts.spikes import plot_isis
from spiketools.plts.spatial import plot_positions, plot_heatmap
from spiketools.plts.trials import plot_rasters, create_trial_title
from spiketools.plts.data import plot_bar, plot_polar_hist, plot_text
from spiketools.plts.stats import plot_surrogates
from spiketools.plts.annotate import color_pval
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.spatial.occupancy import (compute_occupancy, compute_bin_edges,
                                          compute_bin_assignment, compute_bin_firing)
from spiketools.spatial.information import compute_spatial_information
from spiketools.utils.data import compute_range
from spiketools.utils.extract import get_values_by_times
from spiketools.utils.trials import (epoch_spikes_by_event, epoch_spikes_by_range,
                                     epoch_data_by_range)
from spiketools.utils.base import select_from_list

# Import settings from local file
from settings import TASK, PATHS, IGNORE, UNITS, METHODS, ANALYSES, SURROGATES

# Import local code
import sys
sys.path.append('../code')
from place import get_trial_place, compute_place_bins, create_df_place, fit_anova_place
from target import (compute_spatial_target_bins, get_trial_target,
                    create_df_target, fit_anova_target)
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
        ptimes_trials, positions_trials = epoch_data_by_range(\
            ptimes, positions, nav_starts, nav_stops)
        stimes_trials, speed_trials = epoch_data_by_range(\
            stimes, speed, nav_starts, nav_stops)

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
        for uid in keep_inds:

            # Initialize output unit file name & output dictionary
            name = nwbfile.session_id + '_U' + str(uid).zfill(2)

            # Check if unit already run
            if UNITS['SKIP_ALREADY_RUN'] and file_in_list(name, output_files):
                print('\tskipping unit (already run): \tU{:02d}'.format(uid))
                continue

            if UNITS['SKIP_FAILED'] and file_in_list(name, failed_files):
                print('\tskipping unit (failed): \tU{:02d}'.format(uid))
                continue

            print('\trunning unit: \t\t\tU{:02d}'.format(uid))

            try:

                # Collect information of interest
                unit_info = create_unit_info(nwbfile.units[uid])

                # Extract spikes for a unit of interest
                spikes = nwbfile.units.get_unit_spike_times(uid)

                # Initialize results and add unit metadata
                results = {}
                results['uid'] = int(uid)
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
                                                   ANALYSES['TRIAL_RANGE'])
                empty_trials = select_from_list(all_chests, empty_mask)
                full_trials = select_from_list(all_chests, full_mask)

                # Compute bin edges
                x_bin_edges, y_bin_edges = compute_bin_edges(\
                    positions, ANALYSES['PLACE_BINS'], area_range=area_range)

                # Get position values for each spike
                spike_positions = get_values_by_times(ptimes, positions, spikes, threshold=0.25)

                # Compute occupancy
                occ_kwargs = {'minimum' : ANALYSES['MIN_OCCUPANCY'],
                              'area_range' : area_range, 'set_nan' : True}
                occ = compute_occupancy(positions, ptimes,
                                        ANALYSES['PLACE_BINS'],
                                        speed, **occ_kwargs)

                # Compute spatial bin assignments & binned firing, and normalize by occupancy
                x_binl, y_binl = compute_bin_assignment(spike_positions, x_bin_edges, y_bin_edges)
                bin_firing = compute_bin_firing(ANALYSES['PLACE_BINS'], x_binl, y_binl)
                bin_firing = bin_firing / occ

                # Get head direction for each spike
                #spike_hds = get_values_by_times(hd_times, hd_degrees, spikes, threshold=0.25)

                # Compute edges for chest binning
                ch_x_edges, ch_y_edges = compute_bin_edges(\
                    positions, ANALYSES['CHEST_BINS'], area_range=area_range)

                # Assign each chest to a bin
                chest_pos = np.array([chest_xs, chest_ys])
                ch_xbin, ch_ybin = compute_bin_assignment(chest_pos, ch_x_edges, ch_y_edges)

                # Fix offset of chest binning
                ch_xbin = ch_xbin - 1
                ch_ybin = ch_ybin - 1

                # Compute chest occupancy
                chest_occupancy = compute_bin_firing(ANALYSES['CHEST_BINS'], ch_xbin, ch_ybin)

                ## STATISTICS

                # Compare pre/post chest events, computing firing rates & t-test comparison
                fr_pre_all, fr_post_all, results['fr_t_val_all'], results['fr_p_val_all'] = \
                    = compare_pre_post_activity(all_chests, ANALYSES['PRE'], ANALYSES['POST'])
                fr_pre_empt, fr_post_empt, results['fr_t_val_empt'], results['fr_p_val_empt'] = \
                    = compare_pre_post_activity(empty_trials, ANALYSES['PRE'], ANALYSES['POST'])
                fr_pre_full, fr_post_full, results['fr_t_val_full'], results['fr_p_val_full'] = \
                    = compare_pre_post_activity(full_trials, ANALYSES['PRE'], ANALYSES['POST'])

                # Place cell analysis
                if METHODS['PLACE'] == 'INFO':
                    results['place_info'] = compute_spatial_information(bin_firing, occ)
                if METHODS['PLACE'] == 'ANOVA':
                    place_trial = get_trial_place(spikes, nwbfile.trials, ANALYSES['PLACE_BINS'],
                                                  ptimes, positions, speed, x_bin_edges, y_bin_edges,
                                                  occ_kwargs)
                    results['place_anova'] = fit_anova_place(create_df_place(place_trial, drop_na=True))

                # Target cell analysis
                target_bins = compute_spatial_target_bins(\
                    spikes, nav_starts, chest_openings, chest_trials, ptimes, positions,
                    ANALYSES['CHEST_BINS'], ch_xbin, ch_ybin)
                if METHODS['TARGET'] == 'INFO':
                    results['target_info'] = compute_spatial_information(target_bins, chest_occupancy)

                if METHODS['TARGET'] == 'ANOVA':
                    target_trial = get_trial_target(spikes, nav_starts, ANALYSES['CHEST_BINS'],
                                                    chest_openings, chest_trials, ch_xbin, ch_ybin,
                                                    ptimes, positions)
                    results['target_anova'] = fit_anova_target(create_df_target(target_trial))

                # Serial position analysis
                sp_all_frs = compute_serial_position_fr(spikes, nav_starts, chest_openings, chest_trials)
                results['sp_anova'] = fit_anova_serial(create_df_serial(sp_all_frs))

                # Compute measures for head direction
                # results['hd_zstat'], results['hd_pstat']  = circ_rayleigh(convert_angles(spike_hds))

                ## SURROGATES

                # Create shuffled time series for comparison
                times_shuffle = shuffle_spikes(spikes,
                                               SURROGATES['SHUFFLE_APPROACH'],
                                               SURROGATES['N_SURROGATES'])

                # Initialize to store surrogate analysis outputs
                surrs = {analysis : np.zeros(SURROGATES['N_SURROGATES']) for analysis in surr_analyses}

                for ind, shuffle in enumerate(times_shuffle):

                    # PLACE
                    if METHODS['PLACE'] == 'INFO':
                        surr_bin_firing = compute_place_bins(stimes, bins, ptimes, positions,
                                                             x_bin_edges, y_bin_edges)
                        surrs['place_info'][ind] = compute_spatial_information(bin_firing, occ, normalize=True)
                    if METHODS['PLACE'] == 'ANOVA':
                        s_place_trial = get_trial_place(shuffle, nwbfile.trials, ANALYSES['PLACE_BINS'],
                                                        ptimes, positions, speed, x_bin_edges, y_bin_edges,
                                                        occ_kwargs)
                        surrs['place_anova'][ind] = fit_anova_place(create_df_place(s_place_trial, drop_na=True))

                    # TARGET
                    if METHODS['TARGET'] == 'INFO':
                        s_target_bins = compute_spatial_target_bins(\
                            shuffle, nav_starts, chest_openings, chest_trials, ptimes, positions,
                            ANALYSES['CHEST_BINS'], ch_xbin, ch_ybin)
                        surrs['target_info'][ind] = compute_spatial_information(s_target_bins, chest_occupancy)
                    if METHODS['TARGET'] == 'ANOVA':
                        s_target_trial = get_trial_target(\
                            shuffle, nav_starts, ANALYSES['CHEST_BINS'], chest_openings, chest_trials,
                            ch_xbin, ch_ybin, ptimes, positions)
                        surrs['target_anova'][ind] = fit_anova_target(create_df_target(s_target_trial))

                    # SERIAL POSITION
                    s_sp_all_frs = compute_serial_position_fr(\
                        shuffle, nav_starts, chest_openings, chest_trials)
                    surrs['sp_anova'][ind] = fit_anova_serial(create_df_serial(s_sp_all_frs))

                    # HEAD DIRECTION
                    #s_spike_hds = get_values_by_times(hd_times, hd_degrees, shuffle, threshold=0.25)
                    #surrs['hd_stat'][ind] = circ_rayleigh(convert_angles(s_spike_hds))[0]

                # Compute surrogate statistics
                for analysis in surr_analyses:
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                        compute_surrogate_stats(results[analysis], surrs[analysis])

                # Save out unit results
                save_json(results, name + '.json', folder=str(PATHS['RESULTS'] / 'units' / TASK))

                ### MAKE REPORT

                # Initialize figure
                grid = make_grid(6, 3, figsize=(15, 18), wspace=0.4, hspace=1.,
                                 title='Unit Report: {}-U{}'.format(nwbfile.session_id, uid))


                # 00: plot rasters across all trials
                plot_rasters(all_trials, title='All Trials', ax=get_grid_subplot(grid, 0, 0))

                # 01: unit information
                plot_text(create_unit_str(unit_info), title='Unit Information',
                          ax=get_grid_subplot(grid, 0, 1))

                # 02: inter-spike intervals
                plot_isis(compute_isis(spikes), bins=100, range=(0, 2), ax=get_grid_subplot(grid, 0, 2))

                # 10: chest related firing
                plot_rasters(all_chests, xlim=ANALYSES['TRIAL_RANGE'], vline=0,
                             title=create_trial_title('All Chests', \
                                fr_pre_all, fr_post_all, fr_t_val_all, results['fr_p_val_all']),
                             title_color=color_pval(results['fr_p_val_all']),
                             ax=get_grid_subplot(grid, slice(1, 3), slice(0, 2)))


                # 12&22: Compare Empty & Full chest trials
                # Empty chest trials
                plot_rasters(empty_trials, xlim=ANALYSES['TRIAL_RANGE'], vline=0,
                             title=create_trial_title('Empty', \
                                fr_pre_empt, fr_post_empt, fr_t_val_empt, results['fr_p_val_empt']),
                             title_color=color_pval(results['fr_p_val_empt']), title_fontsize=14,
                             ax=get_grid_subplot(grid, 1, 2))

                # Full chest trials
                plot_rasters(full_trials, xlim=ANALYSES['TRIAL_RANGE'], vline=0,
                             title=create_trial_title('Full', \
                                fr_pre_full, fr_post_full, fr_t_val_full, results['fr_p_val_full']),
                             title_color=color_pval(results['fr_p_val_full']), title_fontsize=14,
                             ax=get_grid_subplot(grid, 2, 2))

                # ax30: positional firing
                plot_positions(positions, spike_positions, x_bins=x_bin_edges, y_bins=y_bin_edges,
                               title='Firing Across Positions',
                               ax=get_grid_subplot(grid, slice(3, 5), 0))

                # ax31: positional heatmap
                plot_heatmap(bin_firing, transpose=True, ax=get_grid_subplot(grid, slice(3, 5), 1),
                             title='Range: {:1.2f}-{:1.2f}'.format(*compute_range(bin_firing)))

                # ax32: head direction of spike firing
                # plot_polar_hist(spike_hds, xticklabels=[], yticklabels=[],
                #                 title='Head Direction', title_color=color_pval(results['hd_surr_p_val']),
                #                 ax=get_grid_subplot(grid, 3, 2, polar=True))


                # ax42: place surrogates
                # plot_surrogates(surrs['place_info'], results['place_info'],
                #                 results['place_info_surr_p_val'], title='Place Surrogates',
                #                 title_color=color_pval(results['place_info_surr_p_val']),
                #                 ax=get_grid_subplot(grid, 4, 2))
                plot_surrogates(surrs['place_anova'], results['place_anova'],
                                results['place_anova_surr_p_val'], title='Place Surrogates',
                                title_color=color_pval(results['place_anova_surr_p_val']),
                                ax=get_grid_subplot(grid, 4, 2))

                # ax50: firing rates across trial segments
                plot_bar(sp_all_frs.mean(0), [0, 1, 2, 3], yerr=sem(sp_all_frs, 0),
                         title='Serial Position',
                         title_color=color_pval(results['sp_anova_surr_p_val']),
                         ax=get_grid_subplot(grid, 5, 0))

                # ax51: spatial target firing
                plot_heatmap(target_bins, transpose=True,
                             title='Range: {:1.2f}-{:1.2f}'.format(*compute_range(target_bins)),
                             ax=get_grid_subplot(grid, 5, 1))

                # ax52: target surrogates
                plot_surrogates(surrs['target_anova'], results['target_anova'],
                                results['target_anova_surr_p_val'], title='Target Surrogates',
                                title_color=color_pval(results['target_anova_surr_p_val']),
                                ax=get_grid_subplot(grid, 5, 2))

                # Save out report
                save_figure('unit_report_' + name + '.pdf', PATHS['REPORTS'] / 'units' / TASK, close=True)

            except Exception as excp:
                if not UNITS['CONTINUE_ON_FAIL']:
                    raise
                print('\t\tissue running unit # {}'.format(uid))
                save_txt(traceback.format_exc(), name,
                         folder=str(PATHS['RESULTS'] / 'units' / TASK / 'zFailed'))

        # Close the nwbfile
        io.close()

    print('\n\nCOMPLETED UNIT ANALYSES\n\n')

if __name__ == '__main__':
    main()
