# Local
#from analyzeth.cmh.headDirection.headDirectionStats import nwb_shuffle_spikes_bincirc_navigation
#from analyzeth.cmh.utils import *
#from analyzeth.cmh.headDirection.headDirectionPlots import * 
#from analyzeth.cmh.headDirection.headDirectionUtils import * 
#from analyzeth.cmh.headDirection.headDirectionStats import * 
from analyzeth.cmh.utils.cell_firing_rate import *


class Neuron(object):
    """
    Class for Single Neuron analysis.

    id : str 
        Unique id of neuron. Should have form 'subject_session_unitix'

    spiketrain : list of floats
        Spike train of single neuron in SECONDS (NWB convention)
    
    """
    def __init__(self, spiketrain = None, subject = None, session = None, unit_ix = None):
        self.spiketrain = spiketrain
        self.subject = subject
        self.session = session
        self.unit_ix = unit_ix

        # Metadata
        self.neuron_id = self.session + '_unit' + str(self.unit_ix)
        

    def normalize_spike_train(self, startTime = None):
        """
        Zero spike train times to start of session or given time. For example, if 
        spikes are recorded in UNIX seconds time, then all spike times should be 
        measured realtive to 0 @ UNIX seconds time for the start of the experimental
        session.
        
        If no time is given, the session will initialize with the 
        first spike at 0s 
        """
        if not startTime:
            startTime = self.spiketrain[0]
        self.spiketrain = self.spiketrain - startTime

    def firing_rate_over_time(self, start_time = None, stop_time = None, window = 1, step = 0.1):
        return cell_firing_rate(self.spiketrain, start_time, stop_time, window, step)

    def firing_rate(self):
        return len(self.spiketrain) / (self.spiketrain[-1] - self.spiketrain[0])


class NWB_Neuron(Neuron):
    """
    Class for populating Neuron structure from NWB file

    """    

    def __init__(self, nwbfile, unit_ix):
        self.nwbfile = nwbfile
        Neuron.__init__(self,
            spiketrain = nwbfile.units.get_unit_spike_times(unit_ix),
            subject = nwbfile.subject.subject_id,
            session = nwbfile.session_id,
            unit_ix = unit_ix
        )


class TH_neuron (NWB_Neuron):
    """
    Class for single neurons recorded during treasure hunt 

    Assumes NWB storage, this can be modified later...
    """

    def __init__(self, nwbfile, unit_ix):
        self.nwbfile = None

    

class HeadDirectionCell:
    """
    Class for HD Cell
    """

    def __init__(self, nwbfile, unit_ix):
        self.nwbfile = nwbfile
        self.unit_ix = unit_ix
        self.occupancy = []
    
