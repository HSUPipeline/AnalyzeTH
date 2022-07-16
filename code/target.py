""""Functions for spatial target analyses."""

from functools import partial

from spiketools.stats.anova import create_dataframe_bins, fit_anova

###################################################################################################
###################################################################################################

# Define ANOVA model
MODEL = 'fr ~ C(target_bin)'
FEATURE = 'C(target_bin)'
COLUMNS = ['target_bin', 'fr']

# Create functions for target model
create_df_target = partial(create_dataframe_bins, columns=COLUMNS)
fit_anova_target = partial(fit_anova, formula=MODEL, feature=FEATURE)

###################################################################################################
###################################################################################################
