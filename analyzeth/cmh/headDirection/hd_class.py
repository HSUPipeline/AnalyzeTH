

class Neuron:
    """
    Class for Single Neuron analysis.

    id : str 
        Unique id of neuron. Should have form 'subject_session_unitix'

    spiketrain : list of floats
        Spike train of single neuron in SECONDS (NWB convention)
    
    """
    def __init__(self, id = '', spiketrain = []):
        self.id = id
        self.spiketrain = spiketrain

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


class NWB_neuron(Neuron):
    """
    Class for populating Neuron structure from NWB file

    """    

    def __init__(self, nwbfile, unit_ix):
        self.nwbfile = nwbfile
        self.unit_ix = unit_ix

    

class HeadDirectionCell:
    """
    Class for HD Cell
    """

    def __init__(self, nwbfile, unit_ix):
        self.nwbfile = nwbfile
        self.unit_ix = unit_ix
        self.occupancy = []
        self.
