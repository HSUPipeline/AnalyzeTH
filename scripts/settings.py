"""Settings for TH analysis run scripts."""

from pathlib import Path

###################################################################################################
## RUN SETTINGS

# Set which task to process
TASK = 'THF'

# Set files to ignore
IGNORE = [
   # 'THF_wv002_session_1',
]

# Set verboseness
VERBOSE = True

RUN = {
    'TASK' : TASK,
    'IGNORE' : IGNORE,
    'VERBOSE' : VERBOSE,
}

###################################################################################################
## PATHS

# Set the data path to load from
#BASE_PATH = Path('/Users/tom/Data/JacobsLab/WVTH/')
BASE_PATH = Path('/data12/jacobs_lab/WVTH/')
DATA_PATH = BASE_PATH / 'nwb'

# Set the path to save out reports & results
REPORTS_PATH = Path('../reports/')
RESULTS_PATH = Path('../results/')

PATHS = {
    'BASE' : BASE_PATH,
    'DATA' : DATA_PATH,
    'REPORTS' : REPORTS_PATH,
    'RESULTS' : RESULTS_PATH,
}

###################################################################################################
## UNIT SETTINGS

# Set whether to skip units that have already been processed
SKIP_ALREADY_RUN = False
SKIP_FAILED = False
CONTINUE_ON_FAIL = True

UNITS = {
    'SKIP_ALREADY_RUN' : SKIP_ALREADY_RUN,
    'SKIP_FAILED' : SKIP_FAILED,
    'CONTINUE_ON_FAIL' : CONTINUE_ON_FAIL,
}

###################################################################################################
## METHOD SETTINGS

# Define which method(s) to run (all within list will be run)
PLACE_METHODS = []# ['ANOVA', 'INFO']    # 'info', 'anova'
TARGET_METHODS = ['ANOVA']#, 'INFO']   # 'info', 'anova'
SERIAL_METHODS = ['ANOVA']           # 'anova'

METHODS = {
    'PLACE' : PLACE_METHODS,
    'TARGET' : TARGET_METHODS,
    'SERIAL' : SERIAL_METHODS,
}

###################################################################################################
## ANALYSIS SETTINGS

## AREA SETTINGS

# Area range for place

# USE BOUNDARY RANGE FROM FILES

# Area range for chests
CH_X_RANGE = [360, 410]
CH_Y_RANGE = [320, 400]
CH_AREA_RANGE = [CH_X_RANGE, CH_Y_RANGE]

## BIN SETTINGS

# # LARGE BINS
PLACE_BINS = [5, 7]
CHEST_BINS = [2, 4]

# ORIGINAL BINS
#PLACE_BINS = [9, 12]
#CHEST_BINS = [6, 8]

BINS = {
    'place' : PLACE_BINS,
    'chest' : CHEST_BINS,
}

## OCCUPANCY SETTINGS

OCC_MINIMUM = 1
OCC_SETNAN = True
OCC_SPEED_THRESH = 5e-6
OCC_TIME_THRESH = 0.25
OCC_NORMALIZE = False

OCCUPANCY = {
    'minimum' : OCC_MINIMUM,
    'set_nan' : OCC_SETNAN,
    'speed_threshold' : OCC_SPEED_THRESH,
    'time_threshold' : OCC_TIME_THRESH,
    'normalize' : OCC_NORMALIZE,
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
N_SHUFFLES = 1000

SURROGATES = {
    'approach' : SHUFFLE_APPROACH,
    'n_shuffles' : N_SHUFFLES,
}
