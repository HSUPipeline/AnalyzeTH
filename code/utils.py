"""Utility functions for TH analysis."""

###################################################################################################
###################################################################################################

def select_from_list(lst, select):
    """Select elements from a list based on a boolean mask.

    Parameters
    ----------
    lst : list
        A list of values to select from.
    select : list of bool
        Indicator for which elements to select.

    Returns
    -------
    lst
        Selected elements from the list.
    """

    return [el for el, sel in zip(lst, select) if sel]
