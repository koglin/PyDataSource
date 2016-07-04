import PyDataSource
import os

from pylab import *

class Wave8(PyDataSource.Detector):
    """Acqiris Functions.
    """

    def __init__(self,*args,**kwargs):

        PyDataSource.Detector.__init__(self,*args,**kwargs)

        self.add.parameter(nchannels=8)

        self.add.property(peaks)
        self.add.property(waveforms)

    def _update_xarray_info(self):

        xattrs = {'doc': 'BeamMonitor peak intensity for 8 diode channel waveforms',
                  'unit': 'ADU'}
        self._xarray_info['dims'].update({'peaks': (['ch'], 8, xattrs)}) 

def peaks(self):
    return np.array([max(self.waveforms[ch]) for ch in range(self.nchannels)])

def waveforms(self):
    wfs = []
    for ch in range(self.nchannels):
        wf = self.data_u32[ch]
        back = wf[10:110].mean()
        wfs.append(-1.*wf+back)

    return np.array(wfs)


