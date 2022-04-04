"""Settings for TH analysis run scripts."""

from pathlib import Path

###################################################################################################
###################################################################################################

# Set which task to process
TASK = 'THF'

## PATHS

# Set the data path to load from
BASE_PATH = Path('/Users/tom/Documents/Data/JacobsLab/TH/')
DATA_PATH = BASE_PATH / 'NWB'

# Set the path to save out reports & results
REPORTS_PATH = Path('../reports/')
RESULTS_PATH = Path('../results/')

PATHS = {
    'BASE' : BASE_PATH,
    'DATA' : DATA_PATH,
    'REPORTS' : REPORTS_PATH,
    'RESULTS' : RESULTS_PATH
}

## FILE SETTINGS

# Set files to ignore
IGNORE = []

## UNIT SETTINGS

# Set whether to skip units that have already been processed
SKIP_ALREADY_RUN = True
SKIP_FAILED = True
CONTINUE_ON_FAIL = True

UNIT_SETTINGS = {
    'SKIP_ALREADY_RUN' : SKIP_ALREADY_RUN,
    'SKIP_FAILED' : SKIP_FAILED,
    'CONTINUE_ON_FAIL' : CONTINUE_ON_FAIL
}

## ANALYSIS SETTINGS

# Set the time range to analyze
TRIAL_RANGE = [-1000, 1000]

# Set the spatial bin definition
PLACE_BINS = [7, 21]
CHEST_BINS = [5, 7]

ANALYSIS_SETTINGS = {
    'TRIAL_RANGE' : TRIAL_RANGE,
    'PLACE_BINS' : PLACE_BINS,
    'CHEST_BINS' : CHEST_BINS,
}

## SURROGATE SETTINGS

# Settings for surrogate analyses
N_SURROGATES = 100
SHUFFLE_APPROACH = 'BINCIRC'

SURROGATE_SETTINGS = {
    'N_SURROGATES' : N_SURROGATES,
    'SHUFFLE_APPROACH' : SHUFFLE_APPROACH
}
