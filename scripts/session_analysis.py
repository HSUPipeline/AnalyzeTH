"""Run TH analysis across all sessions."""

import numpy as np

from convnwb.io import get_files, save_json, load_nwbfile
from convnwb.io.utils import file_in_list
from convnwb.utils.log import print_status

from spiketools.spatial.occupancy import compute_occupancy
from spiketools.plts.data import plot_bar, plot_hist, plot_text
from spiketools.plts.spatial import plot_heatmap, plot_positions
from spiketools.plts.spikes import plot_firing_rates
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure

# Import settings from local file
from settings import RUN, PATHS, BINS, OCCUPANCY

# Import local code
import sys
sys.path.append('../code')
from utils import select_navigation, stack_trials
from reports import (create_units_info, create_units_str,
                     create_position_info, create_position_str,
                     create_behav_info, create_behav_str)

###################################################################################################
###################################################################################################

def main():
    """Run session analyses."""

    print_status(RUN['VERBOSE'], '\n\nRUNNING SESSION ANALYSES - {}\n\n'.format(RUN['TASK']), 0)

    nwbfiles = get_files(PATHS['DATA'], select=RUN['TASK'])

    for nwbfilename in nwbfiles:

        ## LOADING & DATA ACCESSING

        # Check and ignore files set to ignore
        if file_in_list(nwbfilename, RUN['IGNORE']):
            print_status(RUN['VERBOSE'], '\nSkipping file (set to ignore): {}'.format(nwbfilename), 0)
            continue

        # Load file and prepare data
        print_status(RUN['VERBOSE'], 'Running session analysis: {}'.format(nwbfilename), 0)

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfilename, PATHS['DATA'], return_io=True)

        ## EXTRACT DATA OF INTEREST

        # Get epoch information
        nav_starts = nwbfile.trials.navigation_start[:]
        nav_stops = nwbfile.trials.navigation_stop[:]

        # Get position data, selecting from navigation periods, and recombine across trials
        ptimes_trials, positions_trials = select_navigation(\
            nwbfile.acquisition['position']['player_position'], nav_starts, nav_stops)
        ptimes, positions = stack_trials(ptimes_trials, positions_trials)

        # Get speed data, selecting from navigation periods, and recombining across trials
        stimes, speed = stack_trials(*select_navigation(\
            nwbfile.processing['position_measures']['speed'], nav_starts, nav_stops))

        ## PRECOMPUTE MEASURES OF INTEREST

        # Collect information
        units_info = create_units_info(nwbfile.units)
        behav_info = create_behav_info(nwbfile.trials)
        pos_info = create_position_info(nwbfile.acquisition, BINS['place'])

        # Compute occupancy
        pos_info['occupancy'] = compute_occupancy(\
            positions, ptimes, BINS['place'], pos_info['area_range'], speed=speed, **OCCUPANCY)

        ## CREATE SESSION REPORT

        # Collect output information to save out
        outputs = {}
        outputs['task'] = RUN['TASK']
        outputs['subject'] =nwbfile.subject.subject_id
        outputs['session'] = nwbfile.session_id
        for field in ['n_units', 'n_keep', 'n_unit_channels']:
           outputs[field] = units_info[field]
        for field in ['n_trials', 'session_length', 'n_chests', 'n_items', '%_correct', 'avg_error']:
           outputs[field] = behav_info[field]

        # Save out session results
        save_json(outputs, nwbfilename, folder=str(PATHS['RESULTS'] / 'sessions' / RUN['TASK']))

        ## CREATE REPORT

        # Initialize figure
        grid = make_grid(4, 3, wspace=0.4, hspace=1.0, figsize=(15, 15),
                         title='TH Subject Report - {}'.format(nwbfile.session_id))

        # 00: units text
        plot_text(create_units_str(units_info), ax=get_grid_subplot(grid, 0, 0))

        # 01: neuron fig
        plot_firing_rates(units_info['frs'], xticks=[], ax=get_grid_subplot(grid, 0, slice(1, None)))

        # 10: unit locations plot
        plot_bar(units_info['location_counts'].values(), units_info['location_counts'].keys(),
                 title='Unit Locations', ax=get_grid_subplot(grid, 1, 0))

        # 11: occupancy map
        plot_heatmap(pos_info['occupancy'], title='Occupancy',
                     ax=get_grid_subplot(grid, slice(1, 3), 1))

        # 12: subject positions overlaid with chest positions
        plot_positions(positions_trials,
                       landmarks={'positions' : pos_info['chests'], 'color' : 'green'},
                       ax=get_grid_subplot(grid, slice(1, 3), 2))

        # 20: position text
        plot_text(create_position_str(pos_info), ax=get_grid_subplot(grid, 2, 0))

        # 30: behaviour text
        plot_text(create_behav_str(behav_info), ax=get_grid_subplot(grid, 3, 0))

        # 31: choice point plot
        plot_bar(behav_info['confidence_counts'].values(), behav_info['confidence_counts'].keys(),
                 title='Confidence Reports', ax=get_grid_subplot(grid, 3, 1))

        # 32: errors plot
        plot_hist(nwbfile.trials.error.data[:], title='Response Error',
                  ax=get_grid_subplot(grid, 3, 2))

        # Save out report
        save_figure('session_report_' + nwbfile.session_id + '.pdf',
                    PATHS['REPORTS'] / 'sessions' / RUN['TASK'], close=True)

        # Close the nwbfile
        io.close()

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED SESSION ANALYSES\n\n', 0)

if __name__ == '__main__':
    main()
