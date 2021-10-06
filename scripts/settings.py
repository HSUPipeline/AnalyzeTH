"""Settings for TH analysis run scripts."""

from pathlib import Path

###################################################################################################
###################################################################################################

## PATHS

# Set the data path to load from
DATA_PATH = Path('.')

# Set the path to save out reports & results
REPORTS_PATH = Path('../reports/')
RESULTS_PATH = Path('../results/')


## SUBJECT SETTINGS

# Set whether to skip units that have already been processed
SKIP_ALREADY_RUN = True

# Set files to ignore
IGNORE = []


## ANALYSIS SETTINGS

# Set the spatial bin definition
BINS = ...

# Settings for surrogate analyses
N_SURROGATES = ...
SHUFFLE_APPROACH = ...
