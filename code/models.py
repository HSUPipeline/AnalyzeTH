""""Helper code & functions to define analysis models."""

from functools import partial

from spiketools.stats.anova import create_dataframe, create_dataframe_bins, fit_anova

###################################################################################################
###################################################################################################

## PLACE MODELS

PLACE = {
    'MODEL' : 'fr ~ C(bin)',
    'FEATURE' : 'C(bin)',
    'COLUMNS' : ['bin', 'fr']
}

create_df_place = partial(create_dataframe_bins)
fit_anova_place = partial(fit_anova, formula=PLACE['MODEL'], feature=PLACE['FEATURE'])

## SPATIAL TARGET MODELS

TARGET = {
    'MODEL' : 'fr ~ C(target_bin)',
    'FEATURE' : 'C(target_bin)',
    'COLUMNS' : ['target_bin', 'fr']
}

create_df_target = create_dataframe
fit_anova_target = partial(fit_anova, formula=TARGET['MODEL'], feature=TARGET['FEATURE'])

## SERIAL POSITION MODELS

SERIAL_POSITION = {
    'MODEL' : 'fr ~ C(segment)',
    'FEATURE' : 'C(segment)',
    'COLUMNS' : ['segment', 'fr']
}

create_df_serial = partial(create_dataframe_bins, bin_columns=SERIAL_POSITION['COLUMNS'])
fit_anova_serial = partial(fit_anova, formula=SERIAL_POSITION['MODEL'],
                           feature=SERIAL_POSITION['FEATURE'])
