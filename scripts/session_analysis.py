"""Run TH analysis across all sessions."""

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pynwb import NWBHDF5IO

from convnwb.io import get_files, save_json

from spiketools.measures import compute_spike_rate
from spiketools.spatial.occupancy import compute_occupancy
from spiketools.plts.data import plot_bar, plot_hist, plot_polar_hist
from spiketools.plts.space import plot_heatmap, plot_positions
from spiketools.plts.spikes import plot_unit_frs
from spiketools.utils.trials import epoch_data_by_range

# Import settings from local file
from settings import TASK, PATHS, IGNORE, ANALYSIS_SETTINGS

# Import local code
import sys
sys.path.append('../code')
from reports import *

###################################################################################################
###################################################################################################

def main():
    """Run session analyses."""

    print('\n\nRUNNING SESSION ANALYSES - {}\n\n'.format(TASK))

    nwbfiles = get_files(PATHS['DATA'], select=TASK)

    for nwbfile in nwbfiles:

        ## LOADING & DATA ACCESSING

        # Check and ignore files set to ignore
        if nwbfile.split('.')[0] in IGNORE:
            print('Skipping file: ', nwbfile)
            continue

        # Load file and prepare data
        print('Running session analysis: ', nwbfile)

        # Get subject name & load NWB file
        nwbfile = NWBHDF5IO(str(PATHS['DATA'] / nwbfile), 'r').read()

        # Get the subject & session ID from file
        subj_id = nwbfile.subject.subject_id
        session_id = nwbfile.session_id

        # Get epoch information
        nav_starts = nwbfile.trials.navigation_start[:]
        nav_stops = nwbfile.trials.navigation_stop[:]

        # Get area ranges, adding a buffer to the z-range (for tower transport)
        area_range = [nwbfile.acquisition['boundaries']['x_range'].data[:],
                      nwbfile.acquisition['boundaries']['z_range'].data[:] + np.array([-10, 10])]

        # Get position & speed information
        positions = nwbfile.acquisition['position']['player_position'].data[:].T
        ptimes = nwbfile.acquisition['position']['player_position'].timestamps[:]
        stimes = nwbfile.processing['position_measures']['speed'].timestamps[:]
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Get head directions
        hd_degrees = nwbfile.acquisition['position']['head_direction'].data[:]

        # Get chest positions
        chest_positions = nwbfile.acquisition['chest_positions']['chest_positions'].data[:].T

        # Get position data for navigation segments
        ptimes_trials, positions_trials = epoch_data_by_range(ptimes, positions, nav_starts, nav_stops)
        stimes_trials, speed_trials = epoch_data_by_range(stimes, speed, nav_starts, nav_stops)

        # Recombine position data across selected navigation trials
        ptimes = np.hstack(ptimes_trials)
        positions = np.hstack(positions_trials)
        stimes = np.hstack(stimes_trials)
        speed = np.hstack(speed_trials)

        ## ANALYZE SESSION DATA

        # Initialize output unit file name & output dictionary
        name = session_id
        results = {}

        # Get settings
        BINS = ANALYSIS_SETTINGS['PLACE_BINS']
        MIN_OCC = ANALYSIS_SETTINGS['MIN_OCCUPANCY']

        # Count confidence answers & fix empty values
        conf_counts = Counter(nwbfile.trials.confidence_response.data[:])
        for el in ['yes', 'maybe', 'no']:
            if el not in conf_counts:
                conf_counts[el] = 0

        # Get unit information
        n_units = len(nwbfile.units)
        keep_inds = np.where(nwbfile.units.keep[:])[0]
        n_keep = len(keep_inds)

        # Compute firing rates for all units marked to keep
        frs = [compute_spike_rate(nwbfile.units.get_unit_spike_times(uind)) \
            for uind in keep_inds]

        # Compute occupancy
        occ = compute_occupancy(positions, ptimes, bins=BINS, speed=speed,
                                minimum=MIN_OCC, area_range=area_range, set_nan=True)

        # Collect information of interest
        subject_info = create_subject_info(nwbfile)
        behav_info = create_behav_info(nwbfile)

        ## CREATE REPORT
        # Initialize figure
        _ = plt.figure(figsize=(15, 15))
        grid = gridspec.GridSpec(4, 3, wspace=0.4, hspace=1.0)
        plt.suptitle('Subject Report - {}'.format(session_id), fontsize=24, y=0.95);

        # 00: subject text
        ax00 = plt.subplot(grid[0, 0])
        plot_text(create_subject_str(subject_info), ax=ax00)

        # 01: neuron fig
        ax01 = plt.subplot(grid[0, 1:])
        plot_unit_frs(frs, ax=ax01)
        ax01.set(xticks=[])

        # 10: position text
        ax10 = plt.subplot(grid[1, 0])
        plot_text(create_position_str(BINS, occ), ax=ax10)

        # 11: occupancy map
        ax11 = plt.subplot(grid[1:3, 1])
        plot_heatmap(occ, transpose=True, title='Occupancy', ax=ax11)

        # 12: subject positions overlaid with chest positions
        ax12 = plt.subplot(grid[1:3, 2])
        plot_positions(positions_trials, ax=ax12)
        ax12.plot(*chest_positions, '.g');

        # 20: head direction
        ax20 = plt.subplot(grid[2, 0], polar=True)
        plot_polar_hist(hd_degrees, title='Head Direction', ax=ax20)

        # 30: behaviour text
        ax20 = plt.subplot(grid[3, 0])
        plot_text(create_behav_str(behav_info), ax=ax20)

        # 31: choice point plot
        ax21 = plt.subplot(grid[3, 1])
        plot_bar(conf_counts.values(), conf_counts.keys(), title='Confidence Reports', ax=ax21)

        # 32: errors plot
        ax22 = plt.subplot(grid[3, 2])
        plot_hist(nwbfile.trials.error.data[:], title='Response Error', ax=ax22)

        # Save out report
        report_name = 'session_report_' + session_id + '.pdf'
        plt.savefig(PATHS['REPORTS'] / 'sessions' / TASK / report_name)

        ## COLLECT RESULTS
        results['task'] = TASK
        results['subj_id'] = subj_id
        results['session_id'] = session_id
        results['session_length'] = subject_info['length']
        results['n_units'] = n_units
        results['n_keep'] = n_keep
        results['n_trials'] = behav_info['n_trials']
        results['n_chests'] = behav_info['n_chests']
        results['n_items'] = behav_info['n_items']
        results['avg_error'] = behav_info['avg_error']

        # Save out unit results
        save_json(results, name + '.json', folder=str(PATHS['RESULTS'] / 'sessions' / TASK))

    print('\n\nCOMPLETED SESSION ANALYSES\n\n')

if __name__ == '__main__':
    main()
