"""Run & collect TH analyses at the group level."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pynwb import NWBHDF5IO

from convnwb.io import get_files

from spiketools.plts.data import plot_hist

# Import settings from local file
from settings import TASK, PATHS, IGNORE

# Import local code
import sys
sys.path.append('../code')
from reports import create_group_info, create_group_str, create_group_sessions_str

###################################################################################################
###################################################################################################

def main():
    """Run group level summary analyses."""

    print('\n\nANALYZING GROUP DATA - {} \n\n'.format(TASK))

    # Get the list of NWB files
    nwbfiles = get_files(PATHS['DATA'], select=TASK)

    # Define summary data to collect
    summary = {
        'ids' : [],
        'n_trials' : [],
        'n_units' : [],
        'n_keep' : [],
        'error' : [],
        'correct' : []
    }

    for nwbfile in nwbfiles:

        ## LOADING & DATA ACCESSING
        # Check and ignore files set to ignore
        if nwbfile.split('.')[0] in IGNORE:
            print('Ignoring file: ', nwbfile)
            continue

        # Load NWB file
        io = NWBHDF5IO(str(PATHS['DATA'] / nwbfile), 'r')
        nwbfile = io.read()

        # Get the subject & session ID from file
        subj_id = nwbfile.subject.subject_id
        session_id = nwbfile.session_id

        # Collect summary information
        summary['ids'].append(session_id)
        summary['n_trials'].append(len(nwbfile.trials))
        summary['n_units'].append(len(nwbfile.units))
        summary['n_keep'].append(sum(nwbfile.units.keep[:]))
        summary['error'].append(np.median(nwbfile.trials.error[:]))
        summary['correct'].append(np.mean(nwbfile.trials.correct[:]))

        # Collect information of interest
        group_info = create_group_info(summary)

        ## CREATE REPORT
        # Initialize figure
        _ = plt.figure(figsize=(15, 12))
        grid = gridspec.GridSpec(3, 3, wspace=0.4, hspace=1.0)
        plt.suptitle('Group Report - {} - {} sessions'.format(TASK, len(summary['ids'])),
                     fontsize=24, y=0.95);

        # 00: group text
        ax00 = plt.subplot(grid[0, 0])
        plot_text(create_group_str(group_info), ax=ax00)

        # 01: neuron firing
        ax01 = plt.subplot(grid[0, 1])
        plot_hist(summary['n_units'], title='Number of Units', ax=ax01)

        # 10-12: behavioural data
        ax10 = plt.subplot(grid[1, 0])
        plot_hist(summary['n_trials'], title='Number of trials', ax=ax10)
        ax11 = plt.subplot(grid[1, 1])
        plot_hist(summary['correct'] * 100, title='Percent Correct', ax=ax11)
        ax12 = plt.subplot(grid[1, 2])
        plot_hist(summary['error'], title='Average Error', ax=ax12)

        # 21: detailed session strings
        ax21 = plt.subplot(grid[2, 1])
        plot_text('\n'.join(create_group_sessions_str(summary)), ax=ax21)

        # Save out report
        report_name = 'group_report_' + TASK + '.pdf'
        plt.savefig(PATHS['REPORTS'] / 'group' / report_name)

    print('\n\nCOMPLETED GROUP ANALYSES\n\n')


if __name__ == '__main__':
    main()
