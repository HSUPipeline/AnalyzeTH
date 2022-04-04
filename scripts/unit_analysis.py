"""Run TH analysis across all units."""

import warnings
from pathlib import Path

import numpy as np
from scipy.stats import sem, ttest_rel
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pynwb import NWBHDF5IO
from pingouin import convert_angles, circ_rayleigh

from convnwb.io import get_files, save_json

from spiketools.measures import compute_isis
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.plts.spikes import plot_waveform, plot_isis
from spiketools.plts.space import plot_positions, plot_heatmap
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_bar, plot_polar_hist
from spiketools.plts.stats import plot_surrogates
from spiketools.utils import restrict_range

from spiketools.stats.permutations import zscore_to_surrogates, compute_empirical_pvalue
from spiketools.spatial.occupancy import (compute_occupancy, compute_spatial_bin_edges,
                                          compute_spatial_bin_assignment)
from spiketools.spatial.information import (compute_spatial_information_2d,
                                            _compute_spatial_information)

from settings import (TASK, DATA_PATH, REPORTS_PATH, RESULTS_PATH, IGNORE,
                      SKIP_ALREADY_RUN, SKIP_FAILED)
from settings import TRIAL_RANGE, PLACE_BINS, CHEST_BINS, N_SURROGATES, SHUFFLE_APPROACH

# Import local code
import sys
sys.path.append('../code')
from utils import select_from_list
from analysis import calc_trial_frs, get_spike_positions, compute_bin_firing, get_spike_heading
from target import compute_serial_position_fr, compute_spatial_target_bins
from reports import *

###################################################################################################
###################################################################################################

def main():
    """Run unit analyses."""

    print('\n\nANALYZING UNIT DATA - {}\n\n'.format(TASK))

    # Get the list of NWB files
    nwbfiles = get_files(DATA_PATH, select=TASK)

    # Get list of already generated and failed units, & drop file names
    output_files = get_files(RESULTS_PATH / 'units' / TASK,
                             select='json', drop_extensions=True)
    failed_files = get_files(RESULTS_PATH / 'units' / TASK / 'zFailed',
                             select='json', drop_extensions=True)

    for nwbfilename in nwbfiles:

        ## DATA LOADING

        # Check and ignore files set to ignore
        if nwbfilename.split('.')[0] in IGNORE:
            print('\nSkipping file (set to ignore): ', nwbfilename)
            continue

        # Print out status
        print('\nRunning unit analysis: ', nwbfilename)

        # Get subject name & load NWB file
        nwbfile = NWBHDF5IO(str(DATA_PATH / nwbfilename), 'r').read()

        # Get the subject & session ID from file
        subj_id = nwbfile.subject.subject_id
        session_id = nwbfile.session_id

        ## GET DATA

        # Get position related data of interest
        pos = nwbfile.acquisition['position']['xy_position']
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Extract head position data
        hd_times = nwbfile.acquisition['position']['head_direction'].timestamps[:]
        hd_degrees = nwbfile.acquisition['position']['head_direction'].data[:]

        # Get the chest positions & trial indices
        chest_xs, chest_ys = nwbfile.acquisition['chest_positions']['chest_positions'].data[:]
        chest_trials = nwbfile.acquisition['chest_trials']['chest_trials'].data[:]
        chest_openings = nwbfile.trials.to_dataframe()['chest_opening_time'].values
        trial_starts = nwbfile.trials['start_time'].data[:]

        ## ANALYZE UNITS

        # Get unit information
        n_units = len(nwbfile.units)
        keep_inds = np.where(nwbfile.units.keep[:])[0]
        n_keep = len(keep_inds)

        # Loop across all units
        for unit_ind in keep_inds:

            # Initialize output unit file name
            name = session_id + '_U' + str(unit_ind).zfill(2)

            # Check if unit already run
            if SKIP_ALREADY_RUN and name in output_files:
                print('\tskipping unit (already run): \tU{:02d}'.format(unit_ind))
                continue

            if SKIP_FAILED and name in failed_files:
                print('\tskipping unit (failed): \tU{:02d}'.format(unit_ind))
                continue

            print('\trunning unit: \t\t\tU{:02d}'.format(unit_ind))

            # Extract spikes for a unit of interest
            spikes = nwbfile.units.get_unit_spike_times(unit_ind)

            # Drop spikes until task time
            st = nwbfile.trials['start_time'][0]
            en = nwbfile.trials['stop_time'][-1]
            spikes = restrict_range(spikes, st, en)

            results = {}

            try:

                ## Compute measures

                # Create shuffled time series for comparison
                times_shuffle = shuffle_spikes(spikes, SHUFFLE_APPROACH, N_SURROGATES)

                # Compute firing related to chest presentation
                all_chests = []
                for opening in nwbfile.trials.chest_opening_time[:]:
                    time_range = restrict_range(spikes, opening + TRIAL_RANGE[0], opening + TRIAL_RANGE[1])
                    all_chests.append(time_range - opening)
                fr_pre, fr_post = calc_trial_frs(all_chests)

                # Compute bin edges
                x_bin_edges, y_bin_edges = compute_spatial_bin_edges(pos.data[:], PLACE_BINS)

                # Get position values for each spike
                spike_xs, spike_ys = get_spike_positions(spikes, pos.timestamps[:], pos.data[:])
                spike_positions = np.array([spike_xs, spike_ys])

                # Compute occupancy
                occ = compute_occupancy(pos.data[:], pos.timestamps[:], PLACE_BINS, speed)

                # Compute spatial bin assignments & binned firing
                x_binl, y_binl = compute_spatial_bin_assignment(spike_positions, x_bin_edges, y_bin_edges)
                bin_firing = compute_bin_firing(x_binl, y_binl, PLACE_BINS)

                # Normalize bin firing by occupancy
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    norm_bin_firing = bin_firing / occ

                # Get head direction for each spike
                spike_hds = get_spike_heading(spikes, hd_times, hd_degrees)

                # Compute edges for chest binning
                area_range = [[360, 410], [320, 400]]
                x_edges, y_edges = compute_spatial_bin_edges(pos.data[:], CHEST_BINS,
                                                             area_range=area_range)

                # Assign each chest to a bin
                chest_pos = np.array([chest_xs, chest_ys])
                ch_xbin, ch_ybin = compute_spatial_bin_assignment(chest_pos, x_edges, y_edges)

                # Fix offset of chest binning
                ch_xbin = ch_xbin - 1
                ch_ybin = ch_ybin - 1

                # Compute chest occupancy
                chest_occupancy = compute_bin_firing(ch_xbin, ch_ybin, CHEST_BINS)


                # Compute firing rates per segment across all trials
                all_frs = compute_serial_position_fr(spikes, trial_starts, chest_openings,
                                                     chest_trials, pos.timestamps, pos.data[:])

                # Collect firing per chest location across all trials
                target_bins = compute_spatial_target_bins(spikes, trial_starts,
                                                          chest_openings, chest_trials,
                                                          pos.timestamps, pos.data[:],
                                                          CHEST_BINS, ch_xbin, ch_ybin)

                ## STATISTICS

                # Compute t-test for chest related firing
                fr_t_val, fr_p_val = ttest_rel(*calc_trial_frs(all_chests, average=False))

                # Compute the spatial information
                place_info = compute_spatial_information_2d(spike_xs, spike_ys,
                                                            [x_bin_edges, y_bin_edges], occ)

                # Compute spatial information for the target firing
                target_info = _compute_spatial_information(target_bins, chest_occupancy)

                # Compute measures for head direction
                hd_z_val, hd_p_val = circ_rayleigh(convert_angles(spike_hds))

                ## SURROGATES

                # Compute surrogate measures
                place_surrs = np.zeros(N_SURROGATES)
                target_surrs = np.zeros(N_SURROGATES)
                hd_surrs = np.zeros(N_SURROGATES)

                for ind, stimes in enumerate(times_shuffle):

                    # PLACE
                    s_spike_xs, s_spike_ys = get_spike_positions(stimes, pos.timestamps, pos.data[:])
                    place_surrs[ind] = compute_spatial_information_2d(s_spike_xs, s_spike_ys,
                                                                      [x_bin_edges, y_bin_edges], occ)

                    # TARGET
                    s_target_bins = compute_spatial_target_bins(stimes, trial_starts,
                                                                chest_openings, chest_trials,
                                                                pos.timestamps, pos.data[:],
                                                                CHEST_BINS, ch_xbin, ch_ybin)
                    target_surrs[ind] = _compute_spatial_information(s_target_bins, chest_occupancy)

                    # HEAD DIRECTION
                    s_spike_hds = get_spike_heading(stimes, hd_times, hd_degrees)
                    hd_surrs[ind] = circ_rayleigh(convert_angles(s_spike_hds))[0]


                # Place surrogate measures
                place_surr_p_val = compute_empirical_pvalue(place_info, place_surrs)
                place_z_score = zscore_to_surrogates(place_info, place_surrs)

                # Target surrogate measures
                target_surr_p_val = compute_empirical_pvalue(target_info, target_surrs)
                target_z_score = zscore_to_surrogates(target_info, target_surrs)

                # Head direction surrogate measures
                hd_surr_p_val = compute_empirical_pvalue(hd_z_val, hd_surrs)
                hd_z_score = zscore_to_surrogates(hd_z_val, hd_surrs)

                ## MAKE REPORT
                # Initialize figure
                _ = plt.figure(figsize=(15, 18))
                grid = gridspec.GridSpec(6, 3, wspace=0.4, hspace=1.)

                # 00: plot rasters across all trials
                ax00 = plt.subplot(grid[0, 0])
                all_trials = []
                for st, en in zip(nwbfile.trials.start_time[:], nwbfile.trials.stop_time[:]):
                    all_trials.append(restrict_range(spikes, st, en) - st)
                plot_rasters(all_trials, ax=ax00, title='All Trials')

                # 01: unit information
                ax01 = plt.subplot(grid[0, 1])
                text = create_unit_str(create_unit_info(nwbfile.units[unit_ind]))
                ax01.text(0.5, 0.7, text, fontdict={'fontsize' : 14}, ha='center', va='center');
                ax01.axis('off');
                ax01.set_title("Unit Information", fontdict={'fontsize' : 16}, y=1.2)

                # 02: inter-spike intervals
                ax02 = plt.subplot(grid[0, 2])
                isis = compute_isis(spikes)
                plot_isis(isis, bins=100, range=(0, 1000), ax=ax02)

                # 10: chest related firing
                ax10 = plt.subplot(grid[1:3, 0:2])
                plot_rasters(all_chests, xlim=TRIAL_RANGE, vline=0, figsize=(10, 7), ax=ax10)
                title_str = 'All Trials - Pre: {:1.2f} - Pos: {:1.2f}  (t:{:1.2f}, p:{:1.2f})'
                tcol10 = 'red' if fr_p_val < 0.05 else 'black'
                ax10.set_title(title_str.format(fr_pre, fr_post, fr_t_val, fr_p_val), color=tcol10)

                # 12&22: chest firing split by full / empty chests
                ax12 = plt.subplot(grid[1, 2])
                ax22 = plt.subplot(grid[2, 2])

                # Create masks to sub-select full & empty chests
                full_mask = nwbfile.trials.full_chest.data[:]
                empty_mask = np.invert(full_mask)

                # Sub-select full & empty chest trials
                empty_trials = select_from_list(all_chests, empty_mask)
                full_trials = select_from_list(all_chests, full_mask)

                # Calculate firing rate pre & post chest opening
                fr_pre_e, fr_post_e = calc_trial_frs(empty_trials)
                fr_pre_f, fr_post_f = calc_trial_frs(full_trials)

                # Plot a comparison of full and empty trials
                plot_rasters(empty_trials, xlim=TRIAL_RANGE, vline=0, ax=ax12,
                             title='Empty Chests - Pre: {:1.2f} - Pos: {:1.2f}'.format(fr_pre_e, fr_post_e))
                plot_rasters(full_trials, xlim=TRIAL_RANGE, vline=0, ax=ax22,
                             title='Full Chests - Pre: {:1.2f} - Pos: {:1.2f}'.format(fr_pre_f, fr_post_f))

                # ax30: positional firing
                ax30 = plt.subplot(grid[3:5, 0])
                plot_positions(pos.data[:], spike_positions,
                               x_bins=x_bin_edges, y_bins=y_bin_edges, ax=ax30,
                               title='Firing Across Positions')

                # ax31: positional heatmap
                ax31 = plt.subplot(grid[3:5, 1])
                plot_heatmap(norm_bin_firing, transpose=True, ax=ax31,
                             title='Range: {:1.2f}-{:1.2f}'.format(np.nanmin(norm_bin_firing), np.nanmax(norm_bin_firing)))

                # ax31: head direction of spike firing
                ax32 = plt.subplot(grid[3, 2], polar=True)
                plot_polar_hist(spike_hds, ax=ax32)
                ax32.set(xticklabels=[], yticklabels=[])
                tcol32 = 'red' if hd_surr_p_val < 0.05 else 'black'
                ax32.set_title('Head Direction', color=tcol32)

                # ax42: place surrogates
                ax42 = plt.subplot(grid[4, 2])
                plot_surrogates(place_surrs, place_info, place_surr_p_val, ax=ax42)
                tcol42 = 'red' if place_surr_p_val < 0.05 else 'black'
                ax42.set_title('Place Surrogates', color=tcol42)

                # ax50: firing rates across trial segments
                ax50 = plt.subplot(grid[5, 0])
                plot_bar(all_frs.mean(0), [0, 1, 2, 3], yerr=sem(all_frs, 0), ax=ax50,
                         title='Serial Position')

                # ax51: spatial target firing
                ax51 = plt.subplot(grid[5, 1])
                plot_heatmap(target_bins, transpose=True, ax=ax51,
                             title='Range: {:1.2f}-{:1.2f}'.format(np.nanmin(target_bins), np.nanmax(target_bins)))

                # ax52: target surrogates
                ax52 = plt.subplot(grid[5, 2])
                plot_surrogates(target_surrs, target_info, target_surr_p_val, ax=ax52)
                tcol52 = 'red' if target_surr_p_val < 0.05 else 'black'
                ax52.set_title('Target Surrogates', color=tcol52)

                # Add super title to the report
                suptitle = 'Unit Report: {}-U{}'.format(session_id, unit_ind)
                plt.suptitle(suptitle, fontsize=24, y=0.95);

                # Save out report
                report_name = 'unit_report_' + name + '.pdf'
                plt.savefig(REPORTS_PATH / 'units' / TASK / report_name)
                plt.close()

                ## COLLECT RESULTS

                results['fr_t_val'] = fr_t_val
                results['fr_p_val'] = fr_p_val

                results['place_info'] = place_info
                results['place_p_val'] = place_surr_p_val
                results['place_z_score'] = place_z_score

                results['target_info'] = target_info
                results['target_p_val'] = target_surr_p_val
                results['target_z_score'] = target_z_score

                results['hd_z_val'] = hd_z_val
                results['hd_p_val'] = hd_p_val

                results['hd_surr_p'] = hd_surr_p_val
                results['hd_surr_z'] = hd_z_score

                # Save out unit results
                save_json(results, name + '.json', folder=str(RESULTS_PATH / 'units' / TASK))

            except Exception as excp:
                print('\t\tissue running unit # {}'.format(unit_ind))
                save_json({}, name + '.json', folder=str(RESULTS_PATH / 'units' / TASK / 'zFailed'))
                continue

if __name__ == '__main__':
    main()
