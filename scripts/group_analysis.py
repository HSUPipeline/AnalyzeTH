"""Run & collect TH analyses at the group level."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pynwb import NWBHDF5IO

from convnwb.io import get_files

from spiketools.plts.data import plot_hist

from settings import TASK, DATA_PATH, REPORTS_PATH

import sys
sys.path.append('../code')
from reports import create_group_info, create_group_str, create_group_sessions_str

###################################################################################################
###################################################################################################

def main():
    """Run group level summary analyses."""

    print('\n\nANALYZING GROUP DATA - {} \n\n'.format(TASK))

    # Get the list of NWB files
    nwbfiles = get_files(DATA_PATH, select=TASK)

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

        # Load NWB file
        io = NWBHDF5IO(str(DATA_PATH / nwbfile), 'r')
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

        # Initialize figure
        _ = plt.figure(figsize=(15, 12))
        grid = gridspec.GridSpec(3, 3, wspace=0.4, hspace=1.0)
        plt.suptitle('Group Report - {} - {} sessions'.format(TASK, len(summary['ids'])),
                     fontsize=24, y=0.95);

        # 00: group text
        ax00 = plt.subplot(grid[0, 0])
        subject_text = create_group_str(create_group_info(summary))
        ax00.text(0.5, 0.5, subject_text, fontdict={'fontsize' : 14}, ha='center', va='center');
        ax00.axis('off');

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
        session_text = '\n'.join(create_group_sessions_str(summary))
        ax21.text(0.5, 0.5, session_text, fontdict={'fontsize' : 14}, ha='center', va='center');
        ax21.axis('off');

        # Save out report
        report_name = 'group_report_' + TASK + '.pdf'
        plt.savefig(REPORTS_PATH / 'group' / report_name)


if __name__ == '__main__':
    main()
