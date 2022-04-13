import analyzeth.cmh.settings_analysis as SETTINGS
from analyzeth.cmh.utils.load_nwb import load_nwb
from analyzeth.cmh.utils.subset_data import subset_period_event_time_data, subset_period_data
from spiketools.utils import restrict_range
import numpy as np 

def nwb_info(
    nwbfile = None,
    data_folder = SETTINGS.DATA_FOLDER,
    task = SETTINGS.TASK,
    subject = SETTINGS.SUBJECT,
    session = SETTINGS.SESSION,
    unit_ix = SETTINGS.UNIT_IX,
    trial_ix = SETTINGS.TRIAL_IX,
    experiment_label = SETTINGS.ACQUISITION_LOCATION
    ):

    """ 
    Print relevant info for NWB file of interest 
    
    If file is not provided it will load given data in analysis settings
    """

    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder) 
    
    # -- SUBJECT DATA -- 
    print('\n -- SUBJECT DATA --')
    print(experiment_label)
    print('Subject {}'.format(subject))


    # -- SESSION DATA -- 
    # Get the number of trials & units
    n_trials = len(nwbfile.trials)
    n_units = len(nwbfile.units)

    # Sesion start and end times
    session_start = nwbfile.trials['start_time'][0]
    session_end = nwbfile.trials['stop_time'][-1]
    session_len = session_end - session_start

    print('\n -- SESSION DATA --')
    print('Chosen Session: \t\t\t {}'.format(session))
    print('Session Start Time: \t\t\t {}'.format(session_start))
    print('Session End Time: \t\t\t {}'.format(session_end))
    print('Total Session Length (ms): \t\t {}'.format(np.round(session_len,2))) 
    print('Total Session Length (sec): \t\t {}'.format(np.round((session_len)/1000,2))) 
    print('Total Session Length (min): \t\t {}'.format(np.round((session_len)/60000,2))) 
    print('Number of trials: \t\t\t {}'.format(n_trials))
    print('Number of units: \t\t\t {}'.format(n_units))


    # -- TRIAL DATA --
    # Extract behavioural markers of interest
    trial_starts = nwbfile.trials['start_time'].data[:]
    chest_openings = nwbfile.trials.chest_opening_time_index[:]

    # Trial start and end times
    trial_ix_start = trial_starts[trial_ix]
    trial_ix_end = chest_openings[trial_ix][-1]  # @cmh may want to modify this will see 
    trial_ix_len = trial_ix_end - trial_ix_start
    
    print('\n -- TRIAL DATA --')
    print('Chosen Trial: \t\t\t\t {}'.format(trial_ix))
    print('Trial Start Time: \t\t\t {}'.format(trial_ix_start))
    print('Trial End Time: \t\t\t {}'.format(trial_ix_end))
    print('Total Trial Length (ms): \t\t {}'.format(np.round(trial_ix_len,2))) 
    print('Total Trial Length (sec): \t\t {}'.format(np.round((trial_ix_len)/1000,2))) 
    print('Total Trial Length (min): \t\t {}'.format(np.round((trial_ix_len)/60000,2))) 


    # -- SPIKE DATA --
    # Get all spikes from unit across session 
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    # Get spikes in trial_ix
    spikes_tix = restrict_range(spikes, trial_ix_start, trial_ix_end)
    n_spikes_tix = len(spikes_tix)

    # Get spikes during navigation periods
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    
    if len(navigation_start_times) != len(navigation_end_times):
        # I actually think I caught this in the parser but JIC
        msg = 'Different lengths of Navigation Start and End Times. The subject likely \
               stopped in the middle of a trial. Remove the last trial and try again'
        raise ValueError(msg)

    spikes_navigation = subset_period_event_time_data(spikes, navigation_start_times, navigation_end_times)
    n_spikes_navigation = len(spikes_navigation)

    print('\n -- UNIT DATA --')
    print('Chosen example unit: \t\t\t {}'.format(unit_ix))
    print('Total number of spikes: \t\t {}'.format(n_spikes_tot))
    print('Number of spikes within session: \t {}'.format(n_spikes_ses)) 
    print('Number of spikes within Trial {}: \t {}'.format(trial_ix, n_spikes_tix)) 
    print('Number of spikes within navigation: \t {}'.format(n_spikes_navigation)) 
    

    # -- HEAD DIRECTION DATA --
    head_direction = nwbfile.acquisition['position']['head_direction']
    hd_times = head_direction.timestamps[:]
    hd_degrees = head_direction.data[:]

    print('\n -- HEAD DIRECTION DATA --')
    print('Session | length of HD timestamps: \t {}'. format(len(hd_times)))
    print('Session | length of HD degree array \t {}'.format(len(hd_degrees)))
    print('Head direction degree range: \t\t [{}, {}]'.format(min(hd_degrees), max(hd_degrees)))

    return
