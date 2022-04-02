"""Settings for TH analysis run scripts."""

from pathlib import Path

###################################################################################################
###################################################################################################

## PATHS

# Set which task to process
TASK = 'THO'

# Set the data path to load from
BASE_PATH = Path('/Users/tom/Documents/Data/JacobsLab/TH/')
DATA_PATH = BASE_PATH / TASK / 'NWB'

# Set the path to save out reports & results
REPORTS_PATH = Path('../reports/')
RESULTS_PATH = Path('../results/')


## SUBJECT SETTINGS

# Set whether to skip units that have already been processed
SKIP_ALREADY_RUN = True
SKIP_FAILED = True

# Set files to ignore
IGNORE = []

## ANALYSIS SETTINGS

# Set the time range to analyze
TRIAL_RANGE = [-1000, 1000]

# Set the spatial bin definition
PLACE_BINS = [7, 21]
CHEST_BINS = [5, 7]

# Settings for surrogate analyses
N_SURROGATES = 100
SHUFFLE_APPROACH = 'BINCIRC'
