""""Functions for serial position analyses."""

from functools import partial
from collections import Counter

import numpy as np

from spiketools.stats.anova import create_dataframe, fit_anova
from spiketools.utils.extract import get_range, get_value_by_time, get_value_by_time_range

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(segment)'
FEATURE = 'C(segment)'
COLUMNS = ['segment', 'fr']

# Create functions for serial position model
create_df_serial = partial(create_dataframe, columns=COLUMNS)
fit_anova_serial = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################

def compute_serial_position_fr(spikes, nav_starts, chest_openings, chest_trials):
    """Collect firing rates per segment across all trials"""

    all_frs = np.zeros([len(nav_starts), 4])
    for t_ind in range(len(nav_starts)):

        t_st = nav_starts[t_ind]
        ch_openings = chest_openings[t_ind]
        t_en = ch_openings[-1]

        chest_trials == t_ind

        t_spikes = get_range(spikes, t_st, t_en)

        seg_times = np.diff(np.insert(ch_openings, 0, t_st))
        count = Counter({0 : 0, 1 : 0, 2 : 0, 3 : 0})
        count.update(np.digitize(t_spikes, ch_openings))
        inds = count.keys()
        frs = np.array(list(count.values())) / seg_times

        all_frs[t_ind, :] = frs

    return all_frs
