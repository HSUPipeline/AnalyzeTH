# Imports
from pynwb import NWBHDF5IO
import glob
from thefuzz import process      # Levenshtein distancce fuzzy string matching

# Settings
from analyzeth.cmh import settings_analysis as SETTINGS 


def load_nwb(task = None,
             subject = None,
             session = None,
             data_folder = None
            ):
    """ Function for loading NWB Files for Task + Subject + Session. 
    
    load_nwb will first check for a file with default naming. If this 
    is not found, it will check all files in the given data folder and 
    calculate the Levenshtein distance for fuzzy string matching. This 
    file will be returned and the user will be asked if this is the 
    correct file to load. 

    If parameters are not given at call (i.e. task = None etc.), it will
    pull from the settings_TC.py file for default loading settings
    
    Defaults for now:
        task = 'THF',
        subj = 'wv001',
        session = 2,
        data_folder = '/home1/cameron.holman/data/wvu/'

    NOTE: may make sense to update glob to os.walk to check subdirs

    PARAMETERS
    ----------
    task: str
        name of the task, ex: 
            'THO'
            'THF' 
    
    subj: str
        name of the subject, ex:
            'wv001'
            'wv002'
            ...
    
    session: int
        session of interest

    data_folder: str
        path to folder containing nwb files, ex:
            '/home1/cameron.holman/data/wvu/'

            include the final slash! 

    
    RETURNS
    -------
    nwb_file: .nwb
        NWB file containing data for the given TASK + SUBJ + SESSION
    """

    # Check for given arguments and load from SETTINGS 
    #       @cmh this could be moved to args, but want to print where 
    #       loading data is taken from
    if task == None:
        task = SETTINGS.TASK
        print ('Task not set \t\t | Task from SETTINGS: \t\t {}'.format(task))
    if subject == None:
        subject = SETTINGS.SUBJECT
        print('Subject not set \t | Subject from SETTINGS: \t {}'.format(subject))
    if session == None:
        session = SETTINGS.SESSION
        print('Session not set \t | Session from SETTINGS: \t {}'.format(session))
    if data_folder == None:
        data_folder = SETTINGS.DATA_FOLDER
        print('Data folder not set \t | Data folder from SETTINGS: \t {} \n'.format(data_folder))

    # Define expected NWB file name & full path
    file_name = '_'.join([task, subject, 'session_' + str(session)]) + '.nwb'
    file_path = data_folder + file_name

    # Try to load file
    try:
        # Load NWB file
        io = NWBHDF5IO(str(file_path), 'r')
        nwbfile = io.read()

    except:
        # -- FIND CLOSEST MATCH --
        print ('Could not find exact match. Looking for closest match... \n')

        # collect files in folder
        files = glob.glob(data_folder +'*.nwb')
        print('These are the NWB files in your data folder:')
        print(files, '\n')

        # Find closest match
        #closest_file = thefuzz.process.extractOne(file_name, files)
        closest_file = process.extractOne(file_name, files)
        print('This file was found to have the closest match:')
        print (closest_file[0])
        print('Match Score = {} \n'.format(closest_file[1]))

        # Ask user if they want to load this file
        while True:
            UI = input ('Would you like to load this file (y/n):')
            if UI.isalpha():
                break
            else: #invalid
                print('Invalid Input. Respond y or n.')
        
        if UI == 'y':
            print ('Great, loading the file...')
            file_path = closest_file[0]
            io = NWBHDF5IO(file_path, 'r')
            nwbfile = io.read()

        elif UI == 'n':
            print ('Oh no! Please try again')
            return
    
    print('Loaded File: \t\t {}'.format(file_path))
    
    return nwbfile