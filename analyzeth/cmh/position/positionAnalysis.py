# -- IMPORTS --
# General
import os
import numpy as np
from pingouin import convert_angles, circ_rayleigh

# NWB
from pynwb import NWBHDF5IO

# Spike Tools
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.stats.permutations import zscore_to_surrogates, compute_empirical_pvalue
from spiketools.utils import restrict_range
from spiketools.plts.trials import plot_rasters

# Local
from analyzeth.cmh.load_nwb import load_nwb
from analyzeth.analysis import get_spike_heading, bin_circular
from analyzeth.plts import plot_polar_hist

# Plots
import matplotlib.pyplot as plt 
import seaborn as sns 
import analyzeth.cmh.settings_plots as PLOTSETTINGS
sns.set()
plt.rcParams.update(PLOTSETTINGS.plot_params)


# Analysis settings
import analyzeth.cmh.settings_analysis as SETTINGS


def plot_positions_TH(
        nwbfile = None,
        task = SETTINGS.TASK,
        subject = SETTINGS.SUBJECT,
        session = SETTINGS.SESSION,
        unit_ix = SETTINGS.UNIT_IX,
        trial_ix = SETTINGS.TRIAL_IX,
        data_folder = SETTINGS.DATA_FOLDER,
        shuffle_approach = SETTINGS.SHUFFLE_APPROACH,
        shuffle_n_surrogates = SETTINGS.N_SURROGATES,
        SHUFFLE = False,
        PLOT = False,
        SAVEFIG = False,
        VERBOSE = False
    ):

    """ Plot position map for session from NWB file """

    # -- LOAD & EXTRACT DATA --
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder)

    # Position data
    pos = nwbfile.acquisition['position']['xy_position']
    position_times = pos.timestamps[:]
    positions = pos.data[:]

    # Get position values for each spike
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    spike_xs, spike_ys = get_spike_positions(spikes, position_times, positions)
    spike_positions = np.array([spike_xs, spike_ys])

