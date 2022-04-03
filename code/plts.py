"""Plotting functions for TH analysis."""

import matplotlib.pyplot as plt

from spiketools.plts.task import plot_task_structure as _plot_task_structure

###################################################################################################
###################################################################################################

def plot_task_structure(trials, ax=None, **plt_kwargs):
    """Plot the task structure for Treasure Hunt.

    Parameters
    ----------
    trials : pynwb.epoch.TimeIntervals
        The TreasureHunt trials structure from a NWB file.
    """

    _plot_task_structure([[trials.encoding_start[:], trials.encoding_end[:]],
                          [trials.distractor_start[:], trials.distractor_end[:]],
                          [trials.recall_start[:], trials.recall_end[:]]],
                         [trials.start_time[:], trials.stop_time[:]],
                         shade_colors=['green', 'orange', 'purple'],
                         line_colors=['red', 'black'],
                         line_kwargs={'lw' : 1.25},
                         ax=ax, **plt_kwargs)
