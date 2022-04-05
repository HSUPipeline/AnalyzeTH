"""Plotting functions for TH analysis."""

import matplotlib.pyplot as plt

from analyzeth.analysis import bin_circular

###################################################################################################
###################################################################################################

def plot_polar_hist(degrees, ax=None):
    """Plot a polar histogram.

    Parameters
    ----------
    degrees : 1d array
        Data to plot in a circular histogram.
    """

    if not ax:
        ax = plt.subplot(111, polar=True)

    bin_edges, counts = bin_circular(degrees)
    bars = ax.bar(bin_edges[:-1], counts)

    return ax


