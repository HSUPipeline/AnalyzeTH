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
from spiketools.plts.space import plot_heatmap
from spiketools.plts.spikes import plot_unit_frs

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

    nwbfiles = get_files(DATA_PATH, select=TASK)

    for nwbfile in nwbfiles:

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

        # Get position & speed information
        pos = nwbfile.acquisition['position']['xy_position']
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Initialize output unit file name & output dictionary
        name = session_id
        results = {}

        ## ANALYZE SESSION DATA

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
        frs = [compute_spike_rate(nwbfile.units.get_unit_spike_times(uind) / 1000) \
            for uind in keep_inds]

        # Compute occupancy
        occ = compute_occupancy(pos.data[:], pos.timestamps[:],
                                ANALYSIS_SETTINGS['PLACE_BINS'],
                                speed, set_nan=True)

        ## CREATE REPORT
        # Initialize figure
        _ = plt.figure(figsize=(15, 15))
        grid = gridspec.GridSpec(4, 3, wspace=0.4, hspace=1.0)
        plt.suptitle('Subject Report - {}'.format(session_id), fontsize=24, y=0.95);

        # 00: subject text
        ax00 = plt.subplot(grid[0, 0])

        # Collect subject information
        subject_info = create_subject_info(nwbfile)
        subject_text = create_subject_str(subject_info)
        ax00.text(0.5, 0.5, subject_text, fontdict={'fontsize' : 14}, ha='center', va='center');
        ax00.axis('off');

        # 01: neuron fig
        ax01 = plt.subplot(grid[0, 1:])
        plot_unit_frs(frs, ax=ax01)
        ax01.set(xticks=[])

        # 10: position text
        ax10 = plt.subplot(grid[1, 0])
        position_text = create_position_str(ANALYSIS_SETTINGS['PLACE_BINS'], occ)
        ax10.text(0.5, 0.5, position_text, fontdict={'fontsize' : 14}, ha='center', va='center');
        ax10.axis('off');

        # 11: occupancy map
        ax11 = plt.subplot(grid[1:3, 1])
        plot_heatmap(occ, transpose=True, title='Occupancy', ax=ax11)

        # 12: subject positions overlaid with chest positions
        ax12 = plt.subplot(grid[1:3, 2])
        ax12.plot(*nwbfile.acquisition['position']['xy_position'].data[:], alpha=0.5)
        ax12.plot(*nwbfile.acquisition['chest_positions']['chest_positions'].data[:], '.g');
        ax12.set(xticks=[], yticks=[]);

        # 20: head direction
        ax10 = plt.subplot(grid[2, 0], polar=True)
        hd_degrees = nwbfile.acquisition['position']['head_direction'].data[:]
        plot_polar_hist(hd_degrees, ax=ax10)
        ax10.set_title('Head Direction')

        # 30: behaviour text
        ax20 = plt.subplot(grid[3, 0])
        behav_info = create_behav_info(nwbfile)
        behav_text = create_behav_str(behav_info)
        ax20.text(0.5, 0.5, behav_text, fontdict={'fontsize' : 14}, ha='center', va='center');
        ax20.axis('off');

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
