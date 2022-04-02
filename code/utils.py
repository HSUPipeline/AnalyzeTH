"""Utility functions for TH analysis."""

###################################################################################################
###################################################################################################

def select_from_list(lst, select):
    """Select elements from a list based on a boolean mask."""

    return [el for el, sel in zip(lst, select) if sel]

def convert_ms_to_minutes(ms):
    """Convert a time value from milliseconds to minutes."""

    return ms / 1000 / 60
