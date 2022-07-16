""""Functions for serial position analyses."""

from functools import partial

#import numpy as np

from spiketools.stats.anova import create_dataframe_bins, fit_anova

#from spiketools.measures.conversions import convert_times_to_rates
#from spiketools.utils.extract import get_range

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(segment)'
FEATURE = 'C(segment)'
COLUMNS = ['segment', 'fr']

# Create functions for serial position model
create_df_serial = partial(create_dataframe_bins, columns=COLUMNS)
fit_anova_serial = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################

# def compute_serial_position_fr(spikes, nav_starts, chest_openings):
#     """Collect firing rates per segment across all trials"""

#     all_frs = np.zeros([len(nav_starts), 4])
#     for t_ind in range(len(nav_starts)):

#         t_st = nav_starts[t_ind]
#         ch_openings = chest_openings[t_ind]
#         t_en = ch_openings[-1]

#         t_spikes = get_range(spikes, t_st, t_en)
#         bin_times = np.insert(ch_openings, 0, t_st)

#         all_frs[t_ind, :] = convert_times_to_rates(t_spikes, bin_times)

#     return all_frs
