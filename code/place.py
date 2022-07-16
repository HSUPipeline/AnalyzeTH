""""Functions for place analyses."""

from functools import partial

from spiketools.stats.anova import create_dataframe_bins, fit_anova

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(bin)'
FEATURE = 'C(bin)'
COLUMNS = ['bin', 'fr']

# Create functions for place model
create_df_place = partial(create_dataframe_bins, columns=COLUMNS)
fit_anova_place = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################
