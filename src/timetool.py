import PyDataSource
import os

from pylab import *

class Timetool(PyDataSource.Detector):
    """Timetool Functions.
    """

    def __init__(self,*args,**kwargs):

        PyDataSource.Detector.__init__(self,*args,**kwargs)
      
        if 'projected_signal' not in self._xarray_info['dims']:
            while self.evtData is None:
                self.next()

            self._add_xarray_evtData(['amplitude', 'nxt_amplitude',
                              'position_fwhm', 'position_pixel', 'position_time',
                              'ref_amplitude'])

            config_attrs = {attr: item for attr, item in self.configData._all_values.items()}
            for attr in self.epicsData._attrs:
                config_attrs.update({attr: getattr(self.epicsData, attr)})

            self._xarray_info['dims'].update(
                        {'projected_signal': (['X'], self.configData.signal_projection_size, config_attrs)} )

            self._xarray_info['coords'].update({'X': np.arange(self.configData.signal_projection_size)})

