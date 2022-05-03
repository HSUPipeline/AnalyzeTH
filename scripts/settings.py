"""Settings for TH analysis run scripts."""

from pathlib import Path

###################################################################################################
###################################################################################################

# Set which task to process
TASK = 'THO'

## PATHS

# Set the data path to load from
#BASE_PATH = Path('/Users/tom/Documents/Data/JacobsLab/TH/')
BASE_PATH = Path('/scratch/tom.donoghue/TH/')
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
SKIP_ALREADY_RUN = False
SKIP_FAILED = False
CONTINUE_ON_FAIL = True

UNIT_SETTINGS = {
    'SKIP_ALREADY_RUN' : SKIP_ALREADY_RUN,
    'SKIP_FAILED' : SKIP_FAILED,
    'CONTINUE_ON_FAIL' : CONTINUE_ON_FAIL
}

## METHOD SETTINGS
PLACE_METHOD = 'ANOVA'  # 'INFO', 'ANOVA'
TARGET_METHOD = 'ANOVA'  # 'INFO', 'ANOVA'

# Collect together method settings
METHOD_SETTINGS = {
    'PLACE' : PLACE_METHOD,
    'TARGET' : TARGET_METHOD
}

## ANALYSIS SETTINGS

# Set the time range to analyze
TRIAL_RANGE = [-1, 1]

# Set the spatial bin definitions
#PLACE_BINS = [7, 21]
PLACE_BINS = [9, 12]
#CHEST_BINS = [5, 7]
CHEST_BINS = [6, 8]

# Occupancy settings
MIN_OCCUPANCY = 1

# Collect together all analysis settings
ANALYSIS_SETTINGS = {
    'TRIAL_RANGE' : TRIAL_RANGE,
    'PLACE_BINS' : PLACE_BINS,
    'CHEST_BINS' : CHEST_BINS,
    'MIN_OCCUPANCY' : MIN_OCCUPANCY
}

## SURROGATE SETTINGS

# Settings for surrogate analyses
N_SURROGATES = 500
SHUFFLE_APPROACH = 'CIRCULAR'   # 'CIRCULAR', 'BINCIRC'

SURROGATE_SETTINGS = {
    'N_SURROGATES' : N_SURROGATES,
    'SHUFFLE_APPROACH' : SHUFFLE_APPROACH
}
