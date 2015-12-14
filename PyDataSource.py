import sys
import operator
import re
import time
import traceback
import psana
import numpy as np
from DataSourceInfo import *
from psana_doc_info import * 

def get_key_info(psana_obj):
    """Get a dictionary of the (type, src, key) for the data types of each src.
    """
    key_info = {}
    for key in psana_obj.keys():
        typ = key.type()
        src = key.src()
        if typ:
            srcstr = str(src)
            if srcstr not in key_info:
                key_info[srcstr] = [] 
            key_info[srcstr].append((typ, src, key.key()))

    return key_info

def get_key_dict(func):
    """Return a dictionary of a psana evt key.
    """
    if hasattr(func, '__module'):
        m = func.__module__.lstrip('psana.')
        n = func.__class__.__name__
        attrs = psana_doc_info[m][n]
    else:
        attrs = [attr for attr in dir(func) if not attr.startswith('_')]
    
    return {attr: get_func_value(getattr(func,attr)) for attr in attrs}

def _repr_value(value):
    if isinstance(value,str):
        return value
    else:
        if isinstance(value, list):
            return 'list'
        elif hasattr(value, 'mean'):
            return '<{:.4}>'.format(value.mean())
        else:
            try:
                return '{:10.5g}'.format(value)
            except:
                try:
                    return value.__str__()
                except:
                    return value

def get_func_value(func):
    """Return the value of a psana object.
    """
    try:
        func = func()
    except:
        pass

    if isinstance(func, list):
        func = [func_redict(f) for f in func]
    else:
        try:
            func = func_redict(func)
        except:
            pass

    return func

def func_redict(f):
    """Resolve the full function.
    """
    if hasattr(f, '__module__') and f.__module__.startswith('psana'):
        m = f.__module__.lstrip('psana.')
        n = f.__class__.__name__
        if n in psana_doc_info[m]:
            return {attr: get_func_value(getattr(f,attr)) for attr in psana_doc_info[m][n]}

        else:
            attrs = [attr for attr in dir(f) if not attr.startswith('_')]
            try:
                return {attrs: getattr(f,attr) for attrs in attrs}
            except:
                return f
    
    else:
        return f

class DataSource(object):
    """Python version of psana.DataSource with support for event and config
       data as well as PyDetector functions to access calibrated data.
    """

    _ds_attrs = ['empty', 'end', 'env', 'runs', 'steps']
    _env_attrs = ['calibDir', 'instrument', 'experiment','expNum']
    _detectors = {}
    _srcs = {}

    def __init__(self, data_source=None, **kwargs):
        self.load_run(data_source=data_source, **kwargs)
        self._evt_time_last = (0,0)
        self._ievent = -1

    def load_run(self, initialize=None, **kwargs):
        self._evtData = None
        self._current_evt = None
        self._evt_keys = {}
        self.data_source = DataSourceInfo(**kwargs)
        self._ds = psana.DataSource(str(self.data_source))
        self.epicsStore = EpicsDictify(self._ds) 

        if self.data_source.indexed:
            self._Events = RunEvents(self, **kwargs)
        else:
            self._Events = Events(self, **kwargs)

        if not self._detectors.get(str(self.data_source)):
            initialize = True
        
        if initialize: 
            self._init_detectors()

    def _init_detectors(self):
        """Initialize psana.Detector classes based on psana env information.
        """
        self.configStore = ConfigStore(self)
        self._aliases = self.configStore._aliases
        for srcstr, item in self.configStore._sources.items():
            alias = item.get('alias')
            self._add_dets(**{alias: srcstr})

    def _add_dets(self, **kwargs):
        if str(self.data_source) not in self._detectors:
            self._detectors.update({str(self.data_source): {}})

        for alias, srcstr in kwargs.items():
            try:
                det = Detector(self, alias)
                self._detectors[str(self.data_source)].update({alias: det})
            except Exception as err:
                print 'Cannot add {:}:  {:}'.format(alias, srcstr) 
                traceback.print_exc()

    @property
    def _current_dets(self):
        """Current detectors from _detector dictionary.
        """
        return self._detectors.get(str(self.data_source), {})

    def show_info(self):
        print self.__repr__()
        for item in self._current_dets.values():
            print item.__repr__()

    @property
    def current(self):
        return EvtDetectors(self)

    def events(self):
        return self._Events

    def next(self, *args, **kwargs):
        return self.events().next(*args, **kwargs) 
 
    def __iter__(self):
        return self

    def __str__(self):
        return  str(self.data_source)

    def __repr__(self):
        repr_str = '{:}: {:}'.format(self.__class__.__name__,str(self))
        return '< '+repr_str+' >'

    def __getattr__(self, attr):
        if attr in self._ds_attrs:
            return getattr(self._ds, attr)
        
        if attr in self._env_attrs:
            return getattr(self._ds.env(), attr)()
        
    def __dir__(self):
        all_attrs =  set(self._ds_attrs + 
                         self._env_attrs +
                         self.__dict__.keys() + dir(DataSource))
        
        return list(sorted(all_attrs))


class ConfigStore(object):
    """ConfigStore
    """
    _configStore_attrs = ['get','put','keys']
    
    def __init__(self, ds):
        configStore = ds.env().configStore()
        self._ds = ds
        self._configStore = configStore
        self._key_info = get_key_info(configStore)

        # Build _config dictionary for each source
        self._config = {}
        self._modules = {}
        for attr, keys in self._key_info.items():
            config = KeyDict(self._configStore, attr, key_info=self._key_info)
            self._config[attr] = config
            for typ, src, key in keys:
                type_name = typ.__name__
                module = typ.__module__.lstrip('psana.')
                if module:
                    if module not in self._modules:
                        self._modules[module] = {}
                    
                    if type_name not in self._modules[module]:
                        self._modules[module][type_name] = []

                    self._modules[module][type_name].append((typ, src, key))

        #Setup Partition
        if not self._modules.get('Partition'):
            print 'ERROR:  No Partition module in configStore data.'
            return 
        elif len(self._modules['Partition']) != 1:
            print 'ERROR:  More than one Partition config type in configStore data.'
            return 
        
        type_name = self._modules.get('Partition').keys()[0]
        if len(self._modules['Partition'][type_name]) == 1:
            typ, src, key = self._modules['Partition'][type_name][0]
            srcstr = str(src)
            config = self._config[srcstr]
        else:
            print 'ERROR:  More that one Partition module in configStore data.'
            print '       ', self._modules['Partition'][type_name]
            return

        self._ipAddrPartition = src.ipAddr()
        self._bldMask = config.bldMask
        self._partition = {str(item['src']): item for item in config.sources}

        # Find Aliases and update Partition
        self._srcAlias = {}
        if self._modules.get('Alias'):
            for type_name, keys in self._modules['Alias'].items():
                for typ, src, key in keys:
                    srcstr = str(src)
                    config = self._config[srcstr]
                    for item in config.srcAlias:
                        self._srcAlias[item['aliasName']] = (item['src'], src.ipAddr())

        self._aliases = {}
        for alias, item in self._srcAlias.items():
            src = item[0]
            ipAddr = item[1]
            srcstr = str(src)
            alias = re.sub('-|:|\.| ','_', alias)
            if srcstr in self._partition:
                self._partition[srcstr]['alias'] = alias
                self._aliases[alias] = srcstr
            elif ipAddr != self._ipAddrPartition or self._ds.data_source.monshmserver:
                # add data sources not in partition that come from recording nodes
                self._partition[srcstr] = {'src': src, 'group': -1, 'alias': alias}
                self._aliases[alias] = srcstr

        # Determine data sources and update aliases
        self._sources = {}
        for srcstr, item in self._partition.items():
            if not item.get('alias'):
                try:
                    alias = srcstr.split('Info(')[1].rstrip(')')
                except:
                    alias = srcstr
                
                alias = re.sub('-|:|\.| ','_',alias)
                item['alias'] = alias
                self._aliases[alias] = srcstr

            if 'NoDetector' not in srcstr and 'NoDevice' not in srcstr:
                # sources not recorded have group None
                # only include these devices for shared memory
                self._sources[srcstr] = item

        # Make dictionary of src: alias for sources with config objects 
        self._config_srcs = {}
        for attr, item in self._sources.items():
            config = self._config.get(attr)
            if config:
                self._config_srcs[item['alias']] = attr
    
        self._output_maps = {}
        self._evr_pulses = {}
        self._eventcodes = {}
        self._readoutGroup = {}

        for type_name in self._modules['EvrData'].keys():
            if type_name.startswith('IOConfig'):
                IOCconfig_type = type_name
            elif type_name.startswith('Config'):
                config_type = type_name

        # get eventcodes and combine output_map info from all EvrData config keys
        map_attrs = ['map', 'conn_id', 'module', 'value', 'source_id']
        for typ, src, key in self._modules['EvrData'][config_type]:
            srcstr = str(src)
            config = self._config[srcstr]
            for eventcode in config.eventcodes:
                self._eventcodes.update({eventcode['code']: eventcode})
                self._readoutGroup.update({eventcode['readoutGroup']: eventcode})

            for output_map in config.output_maps:
                map_key = (output_map['module'],output_map['conn_id'])
                if output_map['source'].get('Pulse'):
                    pulse_id = output_map['source_id']
                    pulse = config.pulses[pulse_id]
                    evr_info = { 'evr_width': pulse['width']*pulse['prescale']/119.e6, 
                                 'evr_delay': pulse['delay']*pulse['prescale']/119.e6, 
                                 'evr_polarity': pulse['polarity']}
                else:
                    pulse_id = None
                    pulse = None
                    evr_info = {'evr_width': None, 'evr_delay': None, 'evr_polarity': None}

                self._output_maps[map_key] = {attr: output_map[attr] for attr in map_attrs} 
                self._output_maps[map_key].update(**evr_info) 

        # Assign evr info to the appropriate sources
        if len(self._modules['EvrData'][IOCconfig_type]) > 1:
            print 'WARNING: More than one EvrData.{:} objects'.format(IOCconfig_type)

        typ, src, key = self._modules['EvrData'][IOCconfig_type][0]
        srcstr = str(src)
        config = self._config[srcstr]
        for ch in config.channels:
            map_key = (ch['output']['module'], ch['output']['conn_id'])
            for i in range(ch['ninfo']):
                src = ch['infos'][i]
                srcstr = str(src)
                self._sources[srcstr].update(**self._output_maps[map_key]) 

        for srcstr, item in self._sources.items():
            group = item.get('group')
            if group and group > 1:
                item['eventCode'] = self._readoutGroup[group]
            else:
                item['eventCode'] = 0

    def __getattr__(self, attr):
        if attr in self._config_srcs:
            return self._config[self._config_srcs[attr]]

        if attr in self._configStore_attrs:
            return getattr(self._configStore, attr)
        
    def __dir__(self):
        all_attrs = set(self._configStore_attrs +
                        self._config_srcs.keys() + 
                        self.__dict__.keys() + dir(ConfigStore))
        return list(sorted(all_attrs))


class RunEvents(object):
    """Event iterator from ds.runs() for indexed data 

       No support yet for multiple runs in a data_source
    """

    _ds_runs = []

    def __init__(self, ds, **kwargs):
        self._kwargs = kwargs
        self._ds = ds
        self.next_run()

    def next_run(self):
        self._ds_run = self._ds.runs().next()
        self._ds_runs.append(self._ds_run)
        self.times = self._ds_run.times()

    def __iter__(self):
        return self

    def next(self, evt_time=None):
        """Optionally pass either an integer for the event number in the data_source
           or a psana.EventTime time stamp to jump to an event.
        """
        if evt_time is not None:
            if isinstance(evt_time, int):
                self._ds._ievent = evt_time
            else:
                self._ds._ievent = self.times.index(evt_time)
        else:
            self._ds._ievent += 1
        
        if self._ds._ievent >= len(self.times):
            print 'No more events in run.'
        else:
            evt = self._ds_run.event(self.times[self._ds._ievent]) 
            self._ds._evt_keys = get_key_info(evt)
            self._ds._current_evt = evt

        return EvtDetectors(self._ds)


class Events(object):
    """Event iterator
    """

    def __init__(self, ds, **kwargs):
        self._kwargs = kwargs
        self._ds = ds

    def __iter__(self):
        return self

    def next(self):
        self._ds._ievent += 1
        evt = self._ds._ds.events().next()
        self._ds._evt_keys = get_key_info(evt)
        self._ds._current_evt = evt 

        return EvtDetectors(self._ds)


class EvtDetectors(object):
    """Psana tab accessible event detectors.
       All detectors in Partition or defined in any configStore Alias object 
       (i.e., recording nodes as well as daq) return the relevant attributes of 
       a PyDetector object for that src, but only the sources in the evt.keys()
       show up in the ipython tab accessible dir.
       Preserves get, keys and run method of items in psana events iterators.
    """

    _init_attrs = ['get', 'keys', 'run']
    
    def __init__(self, ds): 
        self._ds = ds
        self._EventId = EventId(self._ds._current_evt)
        self.ievent = ds._ievent

    @property
    def EventId(self):
        return self._EventId

    @property
    def _attrs(self):
        """List of detector names in current evt data.
        """
        return [alias for alias, srcstr in self._ds._aliases.items() \
                                        if srcstr in self._ds._evt_keys]

    @property
    def _dets(self):
        """Dictionary of detectors.
        """
        return self._ds._current_dets

    @property
    def Evr(self):
        """Master evr from psana evt data.
        """
        if self._evr_typ is None:
            self._evr_typ, self._evr_src = self._get_evr_typ_src()

        if self._evr_typ:
            return MasterEvr(self.get(self._evr_typ, self._evr_src))
        else:
            return []

    @property
    def L3T(self):
        """L3T Level 3 trigger.
        """
        if self._l3t_typ is None:
            self._l3t_typ, self._l3t_src = self._get_l3t_typ_src()

        if self._l3t_typ:
            return L3Tdata(self.get(self._l3t_typ, self._l3t_src))
        else:
            return True

    def _get_l3t_typ_src(self):
        """Set the L3T type and source.
        """
        for key in self._ds._current_evt.keys():
            typ = key.type()
            if typ and typ.__module__ == 'psana.L3T':
                return (typ, key.src())
                
        return (False, False)

    def _get_evr_typ_src(self):
        """Set the maste evr. By default automated as there should only be one in the evt keys.
        """
        for key in self._ds._current_evt.keys():
            if hasattr(key.src(),'devName') and getattr(key.src(),'devName')() == 'Evr':
                return (key.type(), key.src())
        
        return (False, False)

    def __str__(self):
        return  '{:}, Run {:}, Event {:}, {:}, {:}'.format(self._ds.data_source.exp, 
                self.run(), self.ievent, str(self.EventId), str(self.Evr))

    def __repr__(self):
        repr_str = '{:}: {:}'.format(self.__class__.__name__, str(self))
        return '< '+repr_str+' >'

    def __getattr__(self, attr):
        if attr in self._init_attrs:
            return getattr(self._ds._current_evt, attr)
        
        if attr in self._dets:
            return self._dets[attr]

    def __dir__(self):
        all_attrs =  set(self._attrs +
                         self._init_attrs +
                         self.__dict__.keys() + dir(EvtDetectors))
        
        return list(sorted(all_attrs))


class L3Tdata(object):
    """L3 Trigger.
    """

    _attrs = ['accept', 'bias', 'result']
    _properties = ['TypeId', 'Version']

    def __init__(self, l3t):
        self._l3t = l3t

    def show_info(self):
        for attr in self._attrs:
            print '{:18s} {:>12}'.format(attr, getattr(self, attr))

    def __str__(self):
        return str(self.result)

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name__, str(self))

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(self._l3t, attr)()

    def __dir__(self):
        all_attrs =  set(self._attrs+
                         self.__dict__.keys() + dir(L3T))
        
        return list(sorted(all_attrs))


class MasterEvr(object):
    """Tab Accessible event Evr information for psana event.
    """

    _attrs = ['fifoEvents', 'numFifoEvents']

    def __init__(self, evr):

        self._evr = evr

    @property
    def eventCodes(self):
        """Event codes
        """
        return [a.eventCode() for a in self.fifoEvents]

    def preset(self, eventCode):
        """True if event code is present in event.
        """
        try:
            return self._evr.present(eventCode)
        except:
            return False

    def show_info(self):
        print '{:18s} {:>12}'.format('eventCodes', self.eventCodes)

    def __str__(self):
        try:
            eventCodeStr = '{:}'.format(self.eventCodes)
        except:
            eventCodeStr = ''
        
        return eventCodeStr

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name__, str(self))

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(self._evr, attr)()

    def __dir__(self):
        all_attrs =  set(self._attrs+
                         self.__dict__.keys() + dir(MasterEvr))
        
        return list(sorted(all_attrs))


class EventId(object):
    """Time stamp information from psana EventId. 
    """

    _attrs = ['fiducials', 'idxtime', 'run', 'ticks', 'time', 'vector']
    _properties = ['timef64']

    def __init__(self, evt):

        self._EventId = evt.get(psana.EventId)

    @property
    def timef64(self):
        return np.float64(self.time[0])+np.float64(self.time[1])/1.e9 

    def show_info(self):
        print self.__repr__()
        for attr in self._attrs:
            if attr != 'idxtime': 
                print '{:18s} {:>12}'.format(attr, getattr(self, attr))

    def __str__(self):
        try:
            EventTimeStr = time.strftime('%H:%M:%S',
                    time.localtime(self.time[0]))
            EventTimeStr += '.{:04}'.format(int(self.time[1]/1e5))
        except:
            EventTimeStr = 'NA'

        return '{:}'.format(EventTimeStr)

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name__, str(self))

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(self._EventId, attr)()

    def __dir__(self):
        all_attrs =  set(self._attrs+
                         self.__dict__.keys() + dir(EventId))
        
        return list(sorted(all_attrs))


class Detector(object):
    """Includes epicsStore, configStore, evrConfig info 
       Uses full ds in order to be able to access epicsStore info on
       an event basis.
    """
    
    def __init__(self, ds, alias, **kwargs):
        """Initialize a psana Detector class for a given detector alias.
           Provides the attributes of the PyDetector functions for the current 
           event if applicable.  Otherwise provides the attributes from the
           raw data in the psana event keys for the given detector.
        """

        self._alias = alias
        self._ds = ds
        self.src = ds._aliases.get(alias)

        if self.src:
            print 'Adding Detector: {:20} {:40}'.format(alias, psana.Source(self.src))
        else:
            print 'ERROR No Detector with alias {:20}'.format(alias)
            return

        self._srcstr = str(self.src)
        self._srcname = self._srcstr.split('(')[1].split(')')[0]
       
        self.configStore = getattr(ds.configStore, self._alias)

        try:
            self._pydet = psana.Detector(self._srcname, ds._ds.env())
        except:
            self._pydet = None

        if not hasattr(self._pydet, 'dettype'):
            self._det_class = None
            self._tabclass = 'evtData'
        elif self._pydet.dettype in [16, 17]:
            self._det_class = WaveformDict
            self._tabclass = 'detector'
        elif self._pydet.dettype:
            self._det_class = ImageDict
            self._tabclass = 'detector'
        else:
            self._det_class = None
            self._tabclass = 'evtData'
    
    @property
    def _attrs(self):
        """Attributes of psana.Detector functions if relevant, and otherwise
           attributes of raw psana event keys for the given detector.
        """
        if self._tabclass:
            attrs = getattr(self, self._tabclass)._attrs
        
        return attrs

    def monitor(self, nevents=-1):
        """Monitor detector attributes continuously with show_info function.
        """ 
        ievent = nevents
        try:
            while ievent != 0:
                self._ds.next()
                ievent -= 1
                try:
                    self.show_info()
                except:
                    pass

        except KeyboardInterrupt:
            ievent = 0

    def show_info(self):
        print '-'*80
        print '{:}: {:}'.format(self._alias, str(self._ds.current))
        print '-'*80
        getattr(self, self._tabclass).show_info()

    @property
    def evtData(self):
        """Tab accessible raw data from psana event keys.
        """
        return KeyDict(self._ds._current_evt, self._srcstr, key_info=self._ds._evt_keys)

    @property
    def epicsStore(self):
        return getattr(self._ds.epicsStore, self._alias)

    @property
    def detector(self):
        """Tab accessible psana.Detector class
        """
        if self._pydet:
            return self._det_class(self._pydet, self._ds._current_evt)
        else:
            return None

    def __str__(self):
        return '{:} {:}'.format(self._alias, self.src)

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name__, str(self))

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(getattr(self, self._tabclass), attr)

    def __dir__(self):
        all_attrs =  set(self._attrs+
                         self.__dict__.keys() + dir(Detector))
        
        return list(sorted(all_attrs))


class WaveformDict(object):
    """Tab accessibile dictified psana.Detector object.
       
       Attributes come from psana.Detector 
       with low level implementation done in C++ or python.  
       Boost is used for the C++.
    """

    _attrs = ['raw', 'waveform', 'wftime'] 

    _attr_docs = {
#            'raw': 'Raw waveform Volts vs time in sec', 
            'waveform': 'Returns np.array waveform [Volts]',
            'wftime': 'Returns np.array waveform sample time [s]',
            } 

    def __init__(self, det, evt):
        self._evt = evt
        self._det = det

    @property
    def instrument(self):
        """Instrument to which this detector belongs.
        """
        return self._det.instrument()

    def print_attributes(self):
        """Print detector attributes.
        """
        self._det.print_attributes()

    def set_calibration(self):
        """On/off correction of time.'
        """
        if self._det.dettype == 16:
            self._det.set_correct_acqiris_time()
        elif self._det.dettype == 17:
            self._det.set_calib_imp()

    def show_info(self):
        """Show information for relevant detector attributes.
        """
        try:
            items = sorted(self._attr_docs.items(), key = operator.itemgetter(0))
            for attr, doc in items:
                fdict = {'attr': attr, 'unit': '', 'doc': doc}
                value = getattr(self, attr)
                if isinstance(value, str):
                    fdict['str'] = value
                elif isinstance(value, list):
                    if len(value) < 5:
                        fdict['str'] = str(value)
                    else:
                        fdict['str'] = 'list'
                elif hasattr(value,'mean'):
                    if value.size < 5:
                        fdict['str'] = str(value)
                    else:
                        fdict['str'] = '<{:.5}>'.format(value.mean())
                else:
                    try:
                        fdict['str'] = '{:12.5g}'.format(value)
                    except:
                        fdict['str'] = str(value)

                print '{attr:18s} {str:>12} {unit:7} {doc:}'.format(**fdict)
        except:
            print 'No Event'

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(self._det, attr)(self._evt)

    def __dir__(self):
        all_attrs =  set(self._attrs +
                         self.__dict__.keys() + dir(WaveformDict))
        
        return list(sorted(all_attrs))


class ImageDict(object):
    """Tab accessibile dictified psana Detector object.
       
       Attributes come from psana.Detector  
       with low level implementation done in C++ or python.  
       Boost is used for the C++.
    """

    _attrs = ['shape', 'size', 'ndim', 'pedestals', 'rms', 'gain', 'bkgd', 'status',
              'status_as_mask', 'mask_calib', 'common_mode', 'raw', 'calib',
              'areas', 'indexes_x', 'indexes_y', 'pixel_size',
              'coords_x', 'coords_y', 'coords_z', 
              'image',
             ] 

    _attr_docs = {
            'shape': 'Shape of raw data array', 
            'size': 'Total size of raw data', 
            'ndim': 'Number of dimensions of raw data', 
            'pedestals': 'Pedestals from calibStore', 
            'rms': '', 
            'gain': 'Pixel Gain factor from calibStore', 
            'bkgd': '', 
            'status': '',
            'common_mode': 'Common mode parameters', 
            'raw': 'Raw data', 
            'calib': 'Calibrated data',
            'areas': 'Pixel area correction factor', 
            'indexes_x': 'Pixel X index', 
            'indexes_y': 'Pixel Y index', 
            'pixel_size': 'Pixel Size',
            'coords_x': 'Pixel X coordinate', 
            'coords_y': 'Pixel Y coordinate', 
            'coords_z': 'Pixel Z coordinate', 
            'image': 'Reconstruced 2D image from calibStore geometry',
            } 

    def __init__(self, det, evt):
        self._evt = evt
        self._det = det

    @property
    def instrument(self):
        """Instrument to which this detector belongs.
        """
        return self._det.instrument()

    def make_image(self, nda):
        """Make an image from the input numpy array based on the 
           geometry in the calib directory for this event.
        """
        return self._det.image(self._evt, nda)

    def set_do_offset(do_offset=True):
        """Not sure what do offset does?
        """
        self._det.set_do_offset(do_offset=do_offset)

    def common_mode_correction(self, nda):
        """Return the common mode correction for the input numpy 
           array (pedestal-subtracted). 
        """
        return self._det.common_mode_correction(self._evt, nda)
        
    def common_mode_apply(self, nda):
        """Apply in place the common mode correction for the input 
           numpy array (pedestal-subtracted). 
        """
        self._det.common_mode_apply(self._evt, nda)

    def mask(self, calib=False, status=False, 
                   edges=False, central=False, 
                   unbond=False, unbondnbrs=False):
        """Returns combined mask.
                calib:      mask from file in calib directory.
                status:     pixel status from file in calib director.
                edges:      mask detector module edge pixels (mbit +1 in mask_geo).
                central:    mask wide central columns (mbit +2 in mask_geo).
                unbond:     mask unbonded pixels (mbit +4 in mask_geo).
                unbondnbrs: mask unbonded neighbour pixels (mbit +8 in mask_geo).
        """
        return self._det.mask(self._evt, calib=False, status=False, edges=False, 
                              central=False, unbond=False, unbondnbrs=False)

    def mask_geo(self, mbits=15): 
        """Return geometry mask for given mbits keyword.
           Default is mbits=15 to mask edges, wide central columns,
             non-bo pixels and their neighbors

           mbits =  +1-edges; 
                    +2-wide central cols; 
                    +4 unbonded pixel; 
                    +8-unbonded neighbour pixels;
        """
        return self._det.mask_geo(self._evt, mbits=mbits)

    def print_attributes(self):
        """Print detector attributes.
        """
        self._det.print_attributes()

    def show_info(self):
        """Show information for relevant detector attributes.
        """
        if self.size > 0:
            items = sorted(self._attr_docs.items(), key = operator.itemgetter(0))
            for attr, doc in items:
                fdict = {'attr': attr, 'unit': '', 'doc': doc}
                value = getattr(self, attr)
                if isinstance(value, str):
                    fdict['str'] = value
                elif isinstance(value, list):
                    if len(value) < 5:
                        fdict['str'] = str(value)
                    else:
                        fdict['str'] = 'list'
                elif hasattr(value,'mean'):
                    if value.size < 5:
                        fdict['str'] = str(value)
                    else:
                        fdict['str'] = '<{:.5}>'.format(value.mean())
                else:
                    try:
                        fdict['str'] = '{:12.5g}'.format(value)
                    except:
                        fdict['str'] = str(value)

                print '{attr:18s} {str:>12} {unit:7} {doc:}'.format(**fdict)
        else:
            print 'No Event'

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(self._det, attr)(self._evt)
        
    def __dir__(self):
        all_attrs =  set(self._attrs +
                         self.__dict__.keys() + dir(ImageDict))
        
        return list(sorted(all_attrs))

# to convert ipAddr int to address 
# import socket, struct
# s = key.src()
# socket.inet_ntoa(struct.pack('!L',s.ipAddr()))

class KeyDict(object):
    """Dictify psana data for a given detector source.
       data_type: 'evt' or 'config' (default = 'evt')
    """
    def __init__(self, objclass, srcstr, key_info=None):
        self._srcstr = srcstr
        if not key_info:
            key_info = get_key_info(objclass)

        self._data = {}
        self._keys = key_info.get(srcstr)
        if self._keys:
            for (typ, src, key) in self._keys:
                if key:
                    typ_func = objclass.get(*item)
                else:
                    typ_func = objclass.get(typ, src)

                self._data.update(get_key_dict(typ_func))
                
    @property
    def _attr_info(self):
        """
        """
        _info = {}
        for (typ, src, key) in self._keys:
            _info[typ] = {}
            type_name = typ.__name__
            module = typ.__module__.lstrip('psana.')
            for attr, item in psana_doc_info[module][type_name].items():
                value = self._data[attr]
                if isinstance(value, dict):
                    try:
                        for a,b in psana_doc_info[module][attr].items():
                            val = value.get(a)
                            name =  '_'.join([attr,a])
                            b['attr'] = name
                            b['value'] = val
                            b['str'] = _repr_value(val)
                            _info[typ][name] = b
                    except:
                        for a,val in value.items():
                            b = {'unit': '', 'doc': ''}
                            name =  '_'.join([attr,a])
                            b['attr'] = name
                            b['value'] = val
                            b['str'] = _repr_value(val)
                            _info[typ][name] = b

                else:
                    item['attr'] = attr
                    item['value'] = value
                    item['str'] = _repr_value(value)
                    _info[typ][attr] = item
        
        return _info

    @property
    def _attrs(self):
        return self._data.keys()

    def show_info(self):
        for typ, type_info in self._attr_info.items():
            type_attrs = sorted(type_info)
            for attr in type_attrs:
                if not attr[0].isupper() or attr in ['TypeId','Version']:
                    item = type_info.get(attr)
                    print '{attr:24s} {str:>12} {unit:7} {doc:}'.format(**item)

    def __getattr__(self, attr):
        if attr in self._data:
            return self._data[attr]
        
    def __dir__(self):
        all_attrs = set(self._data.keys() +
                        self.__dict__.keys() + dir(KeyDict))
        return list(sorted(all_attrs))


class EpicsConfig(object):
    """Tab Accessible configStore Epics information.
       Currently relatively simple, but expect this to be expanded
       at some point with more PV config info with daq update.
    """

    _pv_attrs = ['description', 'interval', 'pvId']

    def __init__(self, configStore):

        # move to KeyDict objects
        self._pvs = {}
        for key in configStore.keys():
            if key.type() and key.type().__module__ == 'psana.Epics':
                a = configStore.get(key.type(),key.src())
                for pv in a.getPvConfig():
                    pvdict = {attr: getattr(pv, attr)() for attr in self._pv_attrs} 
                    self._pvs[pv.description()] = pvdict

    def show_info(self):
        for alias, items in self._pvs.items():
            print '{:18s} {:}'.format(alias, item.pvId)

    def __getattr__(self, attr):
        if attr in self._pvs:
            return self._pvs.get(attr)

    def __dir__(self):
        all_attrs =  set(self._pvs.keys() +
                         self.__dict__.keys() + dir(EpicsConfig))
        
        return list(sorted(all_attrs))


class EpicsDictify(object):
    """Tab accessible dictified epics data from psana epicsStore.
       e.g., 
         epicsStore = EpicsDictify(ds)
         returns dictified representation of ds.env().epicsStore()
    """

    def __init__(self, ds):

        self._ds = ds

        pv_dict = {}
        epicsStore = self._ds.env().epicsStore()
        self.epicsConfig = EpicsConfig(self._ds.env().configStore())

        for pv in  epicsStore.names():
            name = re.sub(':|\.','_',pv)
            #check if valid -- some old data had aliases generated from comments in epicsArch files.
            if re.match("[_A-Za-z][_a-zA-Z0-9]*$", name) and not ' ' in name and not '-' in name:
                pvname = epicsStore.pvName(pv)
                if pvname:
                    pvalias = pv
                else:
                    pvalias = epicsStore.alias(pv)
                    pvname = pv

                pvalias = re.sub(':|\.|-| ','_',pvalias)

                components = re.split(':|\.|-| ',pv)
                if len(components) == 1:
                    components = re.split('_',pv,1)
                
                # check if alias has 2 components -- if not fix
                if len(components) == 1:
                    pv = '_'.join([components[0], components[0]])
                    components = re.split('_',pv,1)

                for i,item in enumerate(components):
                    if item[0].isdigit():
                         components[i] = 'n'+components[i]

                pv_dict[name] =  { 'pv': pvname,
                                   'alias': pvalias,
                                   'components': components,
                                 }
        self._pv_dict = pv_dict
        self._attrs = list(set([val['components'][0] for val in self._pv_dict.values()]))

    def __getattr__(self, attr):
        if attr in self._attrs:
            attr_dict = {key: pdict for key,pdict in self._pv_dict.items()
                         if pdict['components'][0] == attr}
            return PVdictify(attr_dict, self._ds, level=1)
        
        if attr in dir(self._ds.env().epicsStore()):
            return getattr(self._ds.env().epicsStore(),attr)

    def __dir__(self):
        all_attrs = set(self._attrs +
                        dir(self._ds.env().epicsStore()) +
                        self.__dict__.keys() + dir(EpicsDictify))
        return list(sorted(all_attrs))


class PVdictify(object):
    """Dot.dictifies a dictionary of {PVnames: values}.
    """

    def __init__(self, attr_dict, ds, level=0):
        self._attr_dict = attr_dict
        self._ds = ds
        self._level = int(level)
        self._attrs = list(set([pdict['components'][level]
                                for key,pdict in attr_dict.items()]))

    def _get_pv(self, pv):
        return EpicsStorePV(self._ds.env().epicsStore(), pv)

    def show_info(self):
        """Show information from PVdictionary for all PV's starting with 
           the specified dictified base.
           (i.e. ':' replaced by '.' to make them tab accessible in python)
        """
        print self.get_info()

    def get_info(self):
        """Return string representation of all PV's starting with 
           the specified dictified base.
           (i.e. ':' replaced by '.' to make them tab accessible in python)
        """
        info = ''
        items = sorted(self._attr_dict.items(), key=operator.itemgetter(0))
        for key,pdict in items:
            alias = pdict['alias']
            if alias:
                name = alias
                pv = pdict['pv']
            else:
                name = pdict['pv']
                pv = ''

            pvfunc = self._get_pv(pdict['pv'])
            value = pvfunc.value
            if pvfunc.isCtrl:
                comment = 'isCtrl'
            else:
                comment = ''

            try:
                info += '{:30s} {:12.4g} -- {:30s} {:10}\n'.format( \
                        name, value, pv, comment)
            except:
                info += '{:30s} {:>12} -- {:30s} {:10}\n'.format( \
                        name, value, pv, comment)
        return info

    def __getattr__(self, attr):
        if attr in self._attrs:
            attr_dict = {key: pdict for key,pdict in self._attr_dict.items()
                         if pdict['components'][self._level] == attr}
            if len(attr_dict) == 1:
                key = attr_dict.keys()[0]
                if len(self._attr_dict[key]['components']) == (self._level+1):
                    pv = self._attr_dict[key]['pv']
                    return self._get_pv(pv)
            if len(attr_dict) > 0:
                return PVdictify(attr_dict, self._ds, level=self._level+1)

    def __repr__(self):
        return self.get_info()

    def __dir__(self):
        all_attrs = set(self._attrs +
                        self.__dict__.keys() + dir(PVdictify))
        return list(sorted(all_attrs))


class EpicsStorePV(object):
    """Dictified psana class for epicsStore PV's. 
    """

    def __init__(self, epicsStore, pv):
        self._epicsStore = epicsStore
        self._pvname = pv
        self._store = epicsStore.getPV(pv)
        self._attrs = [attr for attr in dir(self._store) \
                if not attr.startswith('_')]
        self._show_attrs = [attr for attr in self._attrs \
                if attr not in ['dbr','stamp']]

    def get_info(self):
        info = '-'*80+'\n'
        info += '{:} = {:} -- {:}\n'.format(self._pvname, \
                                self.value, self.stamp)
        info += '-'*80+'\n'
        for attr in self._show_attrs:
            val = self.get(attr)
            info += '{:20} {:12}\n'.format(attr, val)
        
        return info

    def show_info(self):
        print self.get_info()

    def get(self, attr):
        if attr in self._attrs:
            if attr is 'value':
                return self._epicsStore.value(self._pvname)
            else:
                val = getattr(self._store,attr)
                try:
                    if attr is 'stamp':
                        return TimeStamp(val())
                    else:
                        return val() 
                except:
                    return val
        else:
            return None

    def __str__(self):
        return '{:}'.format(self.value)

    def __repr__(self):
        return '< {:} = {:}, {:} -- {:} >'.format(self._pvname, \
                self.value, self.stamp, \
                self.__class__.__name__)

    def __getattr__(self, attr):
        if attr in self._attrs:
            return self.get(attr)

    def __dir__(self):
        all_attrs = set(self._attrs +
                        self.__dict__.keys() + dir(EpicsStorePV))
        return list(sorted(all_attrs))


class TimeStamp(object):

    def __init__(self, stamp):
        self.sec = stamp.sec()
        self.nsec = stamp.nsec()

    @property
    def date(self):
        return time.strftime('%Y-%m-%d', 
                time.localtime(self.sec))

    @property
    def time(self): 
        EventTimeStr = time.strftime('%H:%M:%S',
                time.localtime(self.sec))
        EventTimeStr += '.{:04}'.format(int(self.nsec/1e5))
        return EventTimeStr

    def __str__(self):
        return '{:}.{:} sec'.format(self.sec, self.nsec)

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name_, _self.__str__)


