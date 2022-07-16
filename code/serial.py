""""Functions for serial position analyses."""

from functools import partial

from spiketools.stats.anova import create_dataframe_bins, fit_anova

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
