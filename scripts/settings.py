"""Settings for TH analysis run scripts."""

from pathlib import Path

###################################################################################################
## FILE / RUN SETTINGS

# Set which task to process
TASK = 'THO'

# Set files to ignore
IGNORE = []

## UNIT SETTINGS

# Set whether to skip units that have already been processed
SKIP_ALREADY_RUN = False
SKIP_FAILED = False
CONTINUE_ON_FAIL = True

UNITS = {
    'SKIP_ALREADY_RUN' : SKIP_ALREADY_RUN,
    'SKIP_FAILED' : SKIP_FAILED,
    'CONTINUE_ON_FAIL' : CONTINUE_ON_FAIL
}

###################################################################################################
## PATHS

# Set the data path to load from
BASE_PATH = Path('/Users/tom/Documents/Data/JacobsLab/TH/')
#BASE_PATH = Path('/scratch/tom.donoghue/TH/')
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

###################################################################################################
## METHOD SETTINGS

PLACE_METHOD = 'anova'  # 'info', 'anova'
TARGET_METHOD = 'anova'  # 'info', 'anova'

METHODS = {
    'place' : PLACE_METHOD,
    'target' : TARGET_METHOD
}

###################################################################################################
## ANALYSIS SETTINGS

## BIN SETTINGS

#PLACE_BINS = [7, 21]
PLACE_BINS = [9, 12]
#CHEST_BINS = [5, 7]
CHEST_BINS = [6, 8]

BINS = {
    'place' : PLACE_BINS,
    'chest' : CHEST_BINS,
}

## OCCUPANCY SETTINGS

OCC_MINIMUM = 1
OCC_SETNAN = True
OCC_SPEED_THRESH = 0

OCCUPANCY = {
    'minimum' : OCC_MINIMUM,
    'set_nan' : OCC_SETNAN,
    'speed_threshold' : OCC_SPEED_THRESH
}

## TIME WINDOW SETTINGS

TRIAL_RANGE = [-1, 1]
PRE_WINDOW = [-1, 0]
POST_WINDOW = [0, 1]

WINDOWS = {
    'trial_range' : TRIAL_RANGE,
    'pre' : PRE_WINDOW,
    'post' : POST_WINDOW,
}

## SURROGATE SETTINGS

SHUFFLE_APPROACH = 'CIRCULAR'   # 'CIRCULAR', 'BINCIRC'
N_SHUFFLES = 500

SURROGATES = {
    'approach' : SHUFFLE_APPROACH,
    'n_shuffles' : N_SURROGATES
}
