"""Run & collect TH analyses at the group level."""

import numpy as np

from convnwb.io import get_files, load_nwbfile
from convnwb.io.utils import file_in_list
from convnwb.run import print_status

from spiketools.plts.data import plot_hist, plot_text
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure

# Import settings from local file
from settings import RUN, PATHS

# Import local code
import sys
sys.path.append('../code')
from reports import create_group_info, create_group_str, create_group_sessions_str

###################################################################################################
###################################################################################################

def main():
    """Run group level summary analyses."""

    print_status(RUN['VERBOSE'], '\n\nANALYZING GROUP DATA - {}\n\n'.format(RUN['TASK']), 0)

    # Get the list of NWB files
    nwbfiles = get_files(PATHS['DATA'], select=RUN['TASK'])

    # Define summary data to collect
    summary = {
        'ids' : [],
        'n_trials' : [],
        'n_units' : [],
        'n_keep' : [],
        'error' : [],
        'correct' : []
    }

    for nwbfilename in nwbfiles:

        ## LOADING & DATA ACCESSING

        # Check and ignore files set to ignore
        if file_in_list(nwbfilename, RUN['IGNORE']):
            print_status(RUN['VERBOSE'], '\nSkipping file (set to ignore): {}'.format(nwbfilename), 0)
            continue

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfilename, PATHS['DATA'], return_io=True)

        # Collect summary information
        summary['ids'].append(nwbfile.session_id)
        summary['n_trials'].append(len(nwbfile.trials))
        summary['n_units'].append(len(nwbfile.units))
        summary['n_keep'].append(sum(nwbfile.units.keep[:]))
        summary['error'].append(np.mean(nwbfile.trials.error[:]))
        summary['correct'].append(np.mean(nwbfile.trials.correct[:]))

        # Close the nwbfile
        io.close()

    ## CREATE REPORT
    # Initialize figure with grid layout and add title
    grid = make_grid(3, 3, wspace=0.4, hspace=1.0, figsize=(15, 12),
                     title='Group Report - {} - {} sessions'.format(RUN['TASK'], len(summary['ids'])))

    # 00: group text
    plot_text(create_group_str(create_group_info(summary)), ax=get_grid_subplot(grid, 0, 0))

    # 01: neuron firing
    plot_hist(summary['n_keep'], title='Number of Units', ax=get_grid_subplot(grid, 0, 1))

    # 10-12: behavioural data
    plot_hist(summary['n_trials'], title='Number of trials', ax=get_grid_subplot(grid, 1, 0))
    plot_hist(summary['correct'] * 100, title='Percent Correct', ax=get_grid_subplot(grid, 1, 1))
    plot_hist(summary['error'], title='Average Error', ax=get_grid_subplot(grid, 1, 2))

    # 21: detailed session strings
    plot_text('\n'.join(create_group_sessions_str(summary)), ax=get_grid_subplot(grid, 2, 1))

    # Save out report
    save_figure('group_report_' + RUN['TASK'] + '.pdf', PATHS['REPORTS'] / 'group', close=True)

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED GROUP ANALYSES\n\n', 0)


if __name__ == '__main__':
    main()
