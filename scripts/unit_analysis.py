"""Run TH analysis across all units."""

import traceback

import numpy as np
from scipy.stats import sem

from convnwb.io import load_nwbfile, get_files, save_json, save_txt
from convnwb.io.utils import file_in_list
from convnwb.utils.log import print_status

from spiketools.measures.spikes import compute_isis
from spiketools.measures.trials import compute_segment_frs
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.stats.trials import compare_pre_post_activity
from spiketools.plts.spikes import plot_isis
from spiketools.plts.spatial import plot_positions, plot_heatmap, create_heat_title
from spiketools.plts.trials import plot_rasters, create_raster_title
from spiketools.plts.data import plot_bar, plot_polar_hist, plot_text
from spiketools.plts.stats import plot_surrogates
from spiketools.plts.annotate import color_pvalue
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.spatial.occupancy import (compute_occupancy, compute_bin_edges, compute_bin_assignment,
                                          compute_bin_counts_pos, compute_bin_counts_assgn)
from spiketools.spatial.place import compute_place_bins, compute_trial_place_bins
from spiketools.spatial.target import compute_target_bins
from spiketools.spatial.information import compute_spatial_information
from spiketools.spatial.utils import convert_2dindices
from spiketools.utils.extract import get_values_by_times
from spiketools.utils.epoch import epoch_spikes_by_event, epoch_spikes_by_range
from spiketools.utils.base import select_from_list, add_key_prefix, combine_dicts
from spiketools.utils.run import create_methods_list

# Import settings from local file
from settings import RUN, PATHS, UNITS, METHODS, BINS, OCCUPANCY, WINDOWS, SURROGATES

# Import local code
import sys
sys.path.append('../code')
from utils import select_navigation, stack_trials
from models import (create_df_place, fit_anova_place,
                    create_df_target, fit_anova_target,
                    create_df_serial, fit_anova_serial)
from reports import create_unit_info, create_unit_str

###################################################################################################
###################################################################################################

def main():
    """Run analyses across all units."""

    print_status(RUN['VERBOSE'], '\n\nANALYZING UNIT DATA - {}\n\n'.format(RUN['TASK']), 0)

    # Define output folders
    results_folder = PATHS['RESULTS'] / 'units' / RUN['TASK']
    reports_folder = PATHS['REPORTS'] / 'units' / RUN['TASK']

    # Collect a copy of all settings with a prefixes
    all_settings = [
        add_key_prefix(BINS, 'bins'),
        add_key_prefix(OCCUPANCY, 'occupancy'),
        add_key_prefix(WINDOWS, 'windows'),
        add_key_prefix(SURROGATES, 'surrogates'),
    ]

    # Save out run settings
    save_json(METHODS, 'methods.json', folder=results_folder)
    save_json(combine_dicts([RUN, UNITS]), 'run.json', folder=results_folder)
    save_json(combine_dicts(all_settings), 'settings.json', folder=results_folder)

    # Get the list of NWB files
    nwbfiles = get_files(PATHS['DATA'], select=RUN['TASK'])

    # Get list of already generated and failed units, & drop file names
    output_files = get_files(results_folder, select='json', drop_extensions=True)
    failed_files = get_files(results_folder / 'zFailed', select='json', drop_extensions=True)

    for nwbfilename in nwbfiles:

        ## DATA LOADING

        # Check and ignore files set to ignore
        if file_in_list(nwbfilename, RUN['IGNORE']):
            print_status(RUN['VERBOSE'], '\nSkipping file (set to ignore): {}'.format(nwbfilename), 0)
            continue

        # Print out status
        print_status(RUN['VERBOSE'], '\nRunning unit analysis: {}'.format(nwbfilename), 0)

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfilename, PATHS['DATA'], return_io=True)

        ## GET DATA

        # Get start and stop time of trials
        trial_starts = nwbfile.trials['start_time'].data[:]
        trial_stops = nwbfile.trials['stop_time'].data[:]

        # Get the navigation time ranges
        nav_starts = nwbfile.trials.navigation_start[:]
        nav_stops = nwbfile.trials.navigation_stop[:]

        # Get trial indices of interest
        chest_trials = nwbfile.trials.chest_trials[:]
        chest_openings = nwbfile.trials['chest_opening_time'][:]

        # Get masks for full and empty chest trials
        full_mask = nwbfile.trials.full_chest.data[:]
        empty_mask = np.invert(full_mask)

        # Define the segment times of interest
        seg_times = np.insert(np.array(chest_openings), 0, nav_starts, axis=1)

        # Get area ranges, adding a buffer to the z-range (for tower transport)
        area_range = [nwbfile.acquisition['boundaries']['x_range'].data[:],
                      #nwbfile.acquisition['boundaries']['z_range'].data[:]]
                      nwbfile.acquisition['boundaries']['z_range'].data[:] + np.array([-15, 15])]

        # Get position data, selecting from navigation periods, and recombine across trials
        ptimes_trials, positions_trials = select_navigation(\
            nwbfile.acquisition['position']['player_position'], nav_starts, nav_stops)
        ptimes, positions = stack_trials(ptimes_trials, positions_trials)

        # Get speed data, selecting from navigation periods, and recombining across trials
        stimes, speed = stack_trials(*select_navigation(\
            nwbfile.processing['position_measures']['speed'], nav_starts, nav_stops))

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
                print_status(RUN['VERBOSE'], 'skipping unit (already run): \tU{:02d}'.format(uid), 1)
                continue

            if UNITS['SKIP_FAILED'] and file_in_list(name, failed_files):
                print_status(RUN['VERBOSE'], 'skipping unit (failed): \tU{:02d}'.format(uid), 1)
                continue

            print_status(RUN['VERBOSE'], 'running unit: \t\t\tU{:02d}'.format(uid), 1)

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
                for field in ['wvID', 'keep', 'n_spikes', 'firing_rate',
                              'cluster', 'channel', 'location']:
                    results[field] = unit_info[field]

                ## Compute measures

                # Get the spiking data for each trial
                all_trials = epoch_spikes_by_range(spikes, trial_starts, trial_stops, reset=True)

                # Compute firing related to chest presentation
                all_chests = epoch_spikes_by_event(spikes, np.concatenate(chest_openings),
                                                   WINDOWS['trial_range'])
                empty_trials = select_from_list(all_chests, empty_mask)
                full_trials = select_from_list(all_chests, full_mask)

                # Compute bin edges
                x_bin_edges, y_bin_edges = compute_bin_edges(\
                    positions, BINS['place'], area_range=area_range)

                # Get position values for each spike
                spike_positions = get_values_by_times(ptimes, positions, spikes,
                                                      threshold=OCCUPANCY['time_threshold'])

                # Compute occupancy
                occ = compute_occupancy(positions, ptimes, BINS['place'],
                                        area_range, speed, **OCCUPANCY)

                # Compute spatial bin assignments & binned firing, and normalize by occupancy
                place_bins = compute_bin_counts_pos(spike_positions, BINS['place'], area_range, occ)

                # Compute edges for chest binning
                ch_x_edges, ch_y_edges = compute_bin_edges(positions, BINS['chest'], area_range)

                # Assign each chest to a bin & compute equivalent 1d indices
                chest_pos = np.array([chest_xs, chest_ys])
                ch_xbin, ch_ybin = compute_bin_assignment(chest_pos, ch_x_edges, ch_y_edges)
                chbins = convert_2dindices(ch_xbin, ch_ybin, BINS['chest'])

                # Compute chest occupancy
                chest_occupancy = compute_bin_counts_assgn(BINS['chest'], ch_xbin, ch_ybin)

                ## STATISTICS

                # Compare pre/post chest events, computing firing rates & t-test comparison
                fr_pre_all, fr_post_all, results['fr_t_val_all'], results['fr_p_val_all'] = \
                    compare_pre_post_activity(all_chests, WINDOWS['pre'], WINDOWS['post'])
                fr_pre_empt, fr_post_empt, results['fr_t_val_empt'], results['fr_p_val_empt'] = \
                    compare_pre_post_activity(empty_trials, WINDOWS['pre'], WINDOWS['post'])
                fr_pre_full, fr_post_full, results['fr_t_val_full'], results['fr_p_val_full'] = \
                    compare_pre_post_activity(full_trials, WINDOWS['pre'], WINDOWS['post'])

                # Place cell analysis
                if 'INFO' in METHODS['PLACE']:
                    results['place_info'] = compute_spatial_information(place_bins, occ)
                if 'ANOVA' in METHODS['PLACE']:
                    place_trial = compute_trial_place_bins(spikes, positions, ptimes, BINS['place'],
                                                           nav_starts, nav_stops, area_range,
                                                           speed, flatten=True, **OCCUPANCY)
                    results['place_anova'] = fit_anova_place(create_df_place(place_trial))

                # Target cell analysis
                target_frs = compute_segment_frs(spikes, seg_times)
                target_bins = compute_target_bins(target_frs, BINS['chest'], ch_xbin, ch_ybin, chest_occupancy)

                if 'INFO' in METHODS['TARGET']:
                    results['target_info'] = compute_spatial_information(target_bins, chest_occupancy)

                if 'ANOVA' in METHODS['TARGET']:
                    target_df = create_df_target({'trial' : chest_trials, 'target_bin' : chbins,
                                                  'fr' : target_frs.flatten()})
                    results['target_anova'] = fit_anova_target(target_df)

                # Serial position analysis
                sp_all_frs = compute_segment_frs(spikes, seg_times)
                results['serial_anova'] = fit_anova_serial(create_df_serial(sp_all_frs))

                ## SURROGATES

                # Create shuffled time series for comparison
                times_shuffle = shuffle_spikes(spikes, SURROGATES['approach'], SURROGATES['n_shuffles'])

                # Collect list of which analyses are being run for surrogates, and initialize surrogate stores
                surr_analyses = create_methods_list(METHODS)
                surrs = {analysis : np.zeros(SURROGATES['n_shuffles']) for analysis in surr_analyses}

                for ind, shuffle in enumerate(times_shuffle):

                    # PLACE
                    if 'INFO' in METHODS['PLACE']:
                        surr_place_bins = compute_place_bins(shuffle, positions, ptimes, BINS['place'],
                                                             area_range, speed, OCCUPANCY['speed_threshold'],
                                                             OCCUPANCY['time_threshold'])
                        surrs['place_info'][ind] = compute_spatial_information(surr_place_bins, occ, normalize=True)

                    if 'ANOVA' in METHODS['PLACE']:
                        surr_place_trial = compute_trial_place_bins(shuffle, positions, ptimes, BINS['place'],
                                                                    nav_starts, nav_stops, area_range,
                                                                    speed, flatten=True, **OCCUPANCY)
                        surrs['place_anova'][ind] = fit_anova_place(create_df_place(surr_place_trial))

                    # TARGET
                    if 'INFO' in METHODS['TARGET']:
                        surr_target_frs = compute_segment_frs(shuffle, seg_times)
                        surr_target_bins = compute_target_bins(surr_target_frs, BINS['chest'],
                                                               ch_xbin, ch_ybin, chest_occupancy)
                        surrs['target_info'][ind] = compute_spatial_information(surr_target_bins, chest_occupancy)

                    if 'ANOVA' in METHODS['TARGET']:
                        surr_target_frs = compute_segment_frs(shuffle, seg_times).flatten()
                        surr_target_df = create_df_target({'trial' : chest_trials,
                                                           'target_bin' : chbins,
                                                           'fr' : surr_target_frs})
                        surrs['target_anova'][ind] = fit_anova_target(surr_target_df)

                    # SERIAL POSITION
                    if 'ANOVA' in METHODS['SERIAL']:
                        s_sp_all_frs = compute_segment_frs(shuffle, seg_times)
                        surrs['serial_anova'][ind] = fit_anova_serial(create_df_serial(s_sp_all_frs))

                # Compute surrogate statistics
                for analysis in surr_analyses:
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                        compute_surrogate_stats(results[analysis], surrs[analysis])

                # Save out unit results
                save_json(results, name + '.json', folder=results_folder)

                ### MAKE REPORT

                # Initialize figure
                grid = make_grid(7, 3, figsize=(15, 20), wspace=0.4, hspace=1.,
                                 title='Unit Report: {}-U{}'.format(nwbfile.session_id, uid))

                # 00: plot rasters across all trials
                plot_rasters(all_trials, title='All Trials', ax=get_grid_subplot(grid, 0, 0))

                # 01: unit information
                plot_text(create_unit_str(unit_info), title='Unit Information',
                          ax=get_grid_subplot(grid, 0, 1))

                # 02: inter-spike intervals
                plot_isis(compute_isis(spikes), bins=100, range=(0, 2), ax=get_grid_subplot(grid, 0, 2))

                # 10: chest related firing
                plot_rasters(all_chests, xlim=WINDOWS['trial_range'], vline=0,
                             title=create_raster_title('All Chests', fr_pre_all, fr_post_all,
                                 results['fr_t_val_all'], results['fr_p_val_all']),
                             title_color=color_pvalue(results['fr_p_val_all']),
                             ax=get_grid_subplot(grid, slice(1, 3), slice(0, 2)))


                # 12&22: Compare Empty & Full chest trials
                plot_rasters(empty_trials, xlim=WINDOWS['trial_range'], vline=0,
                             title=create_raster_title('Empty', fr_pre_empt, fr_post_empt,
                                 results['fr_t_val_empt'], results['fr_p_val_empt']),
                             title_color=color_pvalue(results['fr_p_val_empt']), title_fontsize=14,
                             ax=get_grid_subplot(grid, 1, 2))
                plot_rasters(full_trials, xlim=WINDOWS['trial_range'], vline=0,
                             title=create_raster_title('Full', fr_pre_full, fr_post_full,
                                results['fr_t_val_full'], results['fr_p_val_full']),
                             title_color=color_pvalue(results['fr_p_val_full']), title_fontsize=14,
                             ax=get_grid_subplot(grid, 2, 2))

                # ax30: positional firing
                plot_positions(positions_trials, spike_positions,
                               x_bins=x_bin_edges, y_bins=y_bin_edges,
                               title='Firing Across Positions',
                               ax=get_grid_subplot(grid, slice(3, 5), 0))

                # ax31: positional heatmap
                plot_heatmap(place_bins, ax=get_grid_subplot(grid, slice(3, 5), 1),
                             title=create_heat_title('Place Bins', place_bins))

                # ax32: place surrogates (info)
                if 'INFO' in METHODS['PLACE']:
                    plot_surrogates(surrs['place_info'], results['place_info'],
                                    results['place_info_surr_p_val'],
                                    title='Place Surrogates (INFO)',
                                    title_color=color_pvalue(results['place_info_surr_p_val']),
                                    ax=get_grid_subplot(grid, 3, 2))

                # ax42: place surrogates (anova)
                if 'ANOVA' in METHODS['PLACE']:
                    plot_surrogates(surrs['place_anova'], results['place_anova'],
                                    results['place_anova_surr_p_val'],
                                    title='Place Surrogates (ANOVA)',
                                    title_color=color_pvalue(results['place_anova_surr_p_val']),
                                    ax=get_grid_subplot(grid, 4, 2))

                # ax50: serial position
                plot_bar(sp_all_frs.mean(0), [0, 1, 2, 3], yerr=sem(sp_all_frs, 0),
                         title='Serial Position', ax=get_grid_subplot(grid, 5, 0))

                # ax51: spatial target firing
                plot_heatmap(target_bins, ax=get_grid_subplot(grid, slice(5, 7), 1),
                             title=create_heat_title('Target Bins', target_bins))

                # ax52: target surrogates (info)
                if 'INFO' in METHODS['TARGET']:
                    plot_surrogates(surrs['target_info'], results['target_info'],
                                    results['target_info_surr_p_val'],
                                    title='Target Surrogates (INFO)',
                                    title_color=color_pvalue(results['target_info_surr_p_val']),
                                    ax=get_grid_subplot(grid, 5, 2))
                    
                # ax60: serial position surrogates
                if 'ANOVA' in METHODS['SERIAL']:
                    plot_surrogates(surrs['serial_anova'], results['serial_anova'],
                                    results['serial_anova_surr_p_val'],
                                    title='Serial Surrogates (ANOVA)',
                                    title_color=color_pvalue(results['serial_anova_surr_p_val']),
                                    ax=get_grid_subplot(grid, 6, 0))

                # ax62: target surrogates (anova)
                if 'ANOVA' in METHODS['TARGET']:
                    plot_surrogates(surrs['target_anova'], results['target_anova'],
                                    results['target_anova_surr_p_val'],
                                    title='Target Surrogates (ANOVA)',
                                    title_color=color_pvalue(results['target_anova_surr_p_val']),
                                    ax=get_grid_subplot(grid, 6, 2))

                # Save out report
                save_figure('unit_report_' + name + '.pdf', reports_folder, close=True)

            except Exception as excp:
                if not UNITS['CONTINUE_ON_FAIL']:
                    raise
                print_status(RUN['VERBOSE'], 'issue running unit # {}'.format(uid), 2)
                save_txt(traceback.format_exc(), name, folder=results_folder / 'zFailed')

        # Close the nwbfile
        io.close()

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)

if __name__ == '__main__':
    main()
