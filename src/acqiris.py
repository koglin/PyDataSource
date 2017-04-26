import PyDataSource
import os

from pylab import *

class Acqiris(PyDataSource.Detector):
    """Acqiris Functions.
    """

    def __init__(self,*args,**kwargs):

        PyDataSource.Detector.__init__(self,*args,**kwargs)

        if False and hasattr(self.configData, 'nbrChannels'):
            self.add.parameter(nchannels=self.configData.nbrChannels)
        else: 
            self.add.parameter(nchannels=4)

        self.add.property(peaks)

    def _update_xarray_info(self):

        nchannels = self.nchannels
        xattrs = {'doc': 'Acqiris for 4 diode channel waveforms',
                  'unit': 'ADU'}
        self._xarray_info['dims'].update({'peaks': (['ch'], nchannels, xattrs)}) 
        
def peaks(self):
    """Max value of each waveform.
    """
    return np.array([max(self.waveform[ch]) for ch in range(self.nchannels)])


