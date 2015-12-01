import sys
import operator
import re
import time
import traceback
import psana

def live_source(monshmserver='psana', **kwargs):
    """Returns psana source string for live data from shared memory on the current node.
       The standard convention is to have the shared memry servers either named 'psana'
       or the instrument name in all caps.  This will return the source string for 
       the valid one based on searching the local file system.  The process will be
       found at:

          '/dev/shm/PdsMonitorSharedMemory_'+monshmserver
    """
    from glob import glob
    import os

    shm_srvs = glob('/dev/shm/PdsMonitorSharedMemory_'+monshmserver)
    if shm_srvs == []:
        hostsplit = os.uname()[1].split('-')
        instrument = hostsplit[1]
        monshmserver = instrument.upper()
        shm_srvs = glob('/dev/shm/PdsMonitorSharedMemory_'+monshmserver)
    
    if shm_srvs != []:
        try:
            MPI_RANK = 0
            source_str = 'shmem={:}.0:stop=no'.format(monshmserver)
        except:
            print 'Exception in finding shared memory server: ',shm_srvs
            source_str = None
    else:
        source_str = None

    return source_str


class DataSourceInfo(object):
    """-------------------------------------------------------------------------------
       data_source class built from keyword arguments that can be accessed as attributes

            data_source = 'exp=CXI/cxic0115:run=10'

        The following are equivalent:

            data_source = DataSourceInfo(exp='cxic0115', run=10).data_source
            data_source = str(DataSourceInfo(exp='cxic0115', run=10))

        You can also load a data source with keyword options:

            smd:  small data support -- has become standard for experiments after Oct 2015
            h5:   loads hdf5 data instead of xtc
            ffb:  appends ':one-stream' to data source

        The shared memory data_source can be loaded with the monshmemsrver keyword:

            data_source = str(DataSource(monshmemsrver='psana'))

        But shared memory should alse be automatically detected if no arguments are
        supplied and you are on a shared memery server.
            
            data_source = str(DataSource())

    """
    _exp_defaults = {'instrument':    None, 
                     'exp':           None, 
                     'h5':            None,
                     'run':           0,
                     'smd':           None, 
                     'station':       0,
                     'idx':           None,
                     'ffb':           None,
                     'monshmserver':  None,
                     'indexed':       False,
                     'dir':           None,
                     'cfg':           None}

    def __init__(self, data_source=None, **kwargs):
        self.data_source = self._set_data_source(data_source=data_source, **kwargs)

    def _set_exp_defaults(self, **kwargs):
        """Sets experiment defaults based on kwargs and defaults.
        """
        for key, val in self._exp_defaults.items():
            setattr(self, key, kwargs.get(key, val))

        if self.exp is not None:
            self.instrument = self.exp[0:3]

#        inst_id = '{:}:{:}'.format(self.instrument.upper(), self.station)

    def _set_data_source(self, data_source=None, **kwargs): 
        self._set_exp_defaults(**kwargs)

        if self.monshmserver:
            self.indexed = False
            if not data_source:
                data_source = live_source(**kwargs)

        if data_source:
            opts = data_source.split(':')
            for opt in opts:
                items = opt.split('=')
                key = items[0]
                if key not in self._exp_defaults:
                    self._exp_defaults.update({key: None})
                    setattr(self, key, None)

                if len(items) == 2:
                    value = items[1]
                    setattr(self, key, value)
                else:
                    setattr(self, key, True)
                    if key in ['idx']:
                        self.indexed = True

        else:

            if self.exp and self.run:
                self.instrument = self.exp[0:3]
     
                data_source = "exp={exp}:run={run}".format(exp=self.exp,run=self.run)
                
                if self.ffb:
                    data_source += ":one-stream"
                elif self.h5:
                    data_source += ":h5"
                elif self.smd:
                    data_source += ":smd"
                    self.indexed = False 
                elif self.idx:
                    data_source += ":idx"
                    self.indexed = True

            else:
                print 'No data source specified, so assume this is shared memory.'
                data_source = live_source(**kwargs)
                self.monshmserver = data_source
                self.indexed = False


        if not self.instrument and self.exp is not None:
            self.instrument = self.exp[0:3]

        if self.exp and self.monshmserver:
            calibDir = '/reg/d/psdm/cxi/{:}/calib'.format(self.exp)
            print 'setting calibDir', self.exp, calibDir
            psana.setOption('psana.calib-dir', calibDir)


        return data_source

    def show_info(self):
        print self.__repr__()
        for attr in self._exp_defaults:
            print '{:20} {:}'.format(attr, getattr(self, attr))

    def __str__(self):
        return self.data_source

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name__, self.data_source)


