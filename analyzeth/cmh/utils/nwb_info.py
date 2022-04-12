

import analyzeth.cmh.settings_analysis as SETTINGS
from analyzeth.cmh.utils.load_nwb import load_nwb

def nwb_info(
        nwbfile = None,
        data_folder = SETTINGS.DATA_FOLDER,
        task = SETTINGS.TASK,
        subject = SETTINGS.SUBJECT,
        session = SETTINGS.SESSION,
        unit_ix = SETTINGS.UNIT_IX,
        trial_ix = SETTINGS.TRIAL_IX,
        experiment_label = SETTINGS.ACQUISITION_LOCATION,
        ):
    
    # load file if not given
    if nwbfile == None:
        nwbfile = load_nwb(task, subject, session, data_folder) 
    

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

    # -- SPIKE DATA --
    # Get all spikes from unit across session 
    spikes = nwbfile.units.get_unit_spike_times(unit_ix)
    n_spikes_tot = len(spikes)

    # Drop spikes oustside session time
    spikes = restrict_range(spikes, session_start, session_end)
    n_spikes_ses = len(spikes)

    # Get spikes during navigation period 
    navigation_start_times = nwbfile.trials['navigation_start'][:]
    navigation_end_times = nwbfile.trials['navigation_end'][:]
    
    if len(navigation_start_times) != len(navigation_end_times):
        # I actually think I caught this in the parser but JIC
        msg = 'Different lengths of Navigation Start and End Times. The subject likely \
               stopped in the middle of a trial. Remove the last trial and try again'
        raise ValueError(msg)

    spikes_navigation = np.array([])
    for ix in range(len(navigation_start_times)):
        spikes_navigation = np.append(
                            spikes_navigation,
                            spikes[(spikes > navigation_start_times[ix]) \
                                 & (spikes < navigation_end_times[ix])],   # <= ?
                            axis = 0)
    n_spikes_navigation = len(spikes_navigation)

    print('\n -- UNIT DATA --')
    print('Chosen example unit: \t\t\t {}'.format(unit_ix))
    print('Total number of spikes: \t\t {}'.format(n_spikes_tot))
    print('Number of spikes within session: \t {}'.format(n_spikes_ses))  
    print('Number of spikes within navigation: \t {}'.format(n_spikes_navigation)) 

    return