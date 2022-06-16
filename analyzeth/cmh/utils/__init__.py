""" Module - Utility functions for loading NWB, displaying NWB info, subsetting data in epochs of interest """

from .load_nwb import load_nwb
from .nwb_info import nwb_info
from .subset_data import subset_period_data, subset_period_event_time_data
from .cell_firing_rate import *
from .flatten_dict import flatten_dict