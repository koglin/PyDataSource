#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: PyDataSource.py  koglin@SLAC.STANFORD.EDU $
#
# Description:
#  module PyDataSource
#--------------------------------------------------------------------------
"""Python implementation of psana DataSource object.

Example:

    # Import the PyDataSource module
    In [1]: import PyDataSource

    # Load example run
    In [2]: ds = PyDataSource.DataSource('exp=xpptut15:run=54')
    Adding Detector: FEEGasDetEnergy      Source("BldInfo(FEEGasDetEnergy)")      
    Adding Detector: XppSb3_Ipm           Source("BldInfo(XppSb3_Ipm)")           
    Adding Detector: PhaseCavity          Source("BldInfo(PhaseCavity)")          
    Adding Detector: EBeam                Source("BldInfo(EBeam)")                
    Adding Detector: XppEnds_Ipm0         Source("BldInfo(XppEnds_Ipm0)")         
    Adding Detector: XppSb2_Ipm           Source("BldInfo(XppSb2_Ipm)")           
    Adding Detector: yag_lom              Source("DetInfo(XppMonPim.1:Tm6740.1)") 
    Adding Detector: cspad                Source("DetInfo(XppGon.0:Cspad.0)")     
    Adding Detector: yag2                 Source("DetInfo(XppSb3Pim.1:Tm6740.1)") 

    # Access the first event
    In [3]: evt = ds.next()

    # Tab to see Data objects in current event
    In [4]: evt.
    evt.EBeam            evt.FEEGasDetEnergy  evt.XppEnds_Ipm0     evt.cspad            evt.next
    evt.EventId          evt.L3T              evt.XppSb2_Ipm       evt.get              evt.yag2
    evt.Evr              evt.PhaseCavity      evt.XppSb3_Ipm       evt.keys             evt.yag_lom

    # Tab to see EBeam attributes
    In [4]: evt.EBeam.
    evt.EBeam.EventId            evt.EBeam.ebeamEnergyBC1     evt.EBeam.ebeamPhotonEnergy  evt.EBeam.epicsData
    evt.EBeam.Evr                evt.EBeam.ebeamEnergyBC2     evt.EBeam.ebeamPkCurrBC1     evt.EBeam.evtData
    evt.EBeam.L3T                evt.EBeam.ebeamL3Energy      evt.EBeam.ebeamPkCurrBC2     evt.EBeam.monitor
    evt.EBeam.calibData          evt.EBeam.ebeamLTU250        evt.EBeam.ebeamUndAngX       evt.EBeam.next
    evt.EBeam.configData         evt.EBeam.ebeamLTU450        evt.EBeam.ebeamUndAngY       evt.EBeam.show_all
    evt.EBeam.damageMask         evt.EBeam.ebeamLTUAngX       evt.EBeam.ebeamUndPosX       evt.EBeam.show_info
    evt.EBeam.detector           evt.EBeam.ebeamLTUAngY       evt.EBeam.ebeamUndPosY       evt.EBeam.src
    evt.EBeam.ebeamCharge        evt.EBeam.ebeamLTUPosX       evt.EBeam.ebeamXTCAVAmpl     
    evt.EBeam.ebeamDumpCharge    evt.EBeam.ebeamLTUPosY       evt.EBeam.ebeamXTCAVPhase    

    # Print a table of the EBeam data for the current event
    In [4]: evt.EBeam.show_info()
    --------------------------------------------------------------------------------
    EBeam: xpptut15, Run 54, Event 0, 11:37:12.4517, [140, 141, 41, 40]
    --------------------------------------------------------------------------------
    damageMask                 1.0486e+06         Damage mask.
    ebeamCharge                0.00080421 nC      Beam charge in nC.
    ebeamDumpCharge                     0 e-      Bunch charge at Dump in num. electrons
    ebeamEnergyBC1                -13.772 mm      Beam position in mm (related to beam energy).
    ebeamEnergyBC2               -0.38553 mm      Beam position in mm (related to beam energy).
    ebeamL3Energy                       0 MeV     Beam energy in MeV.
    ebeamLTU250                         0 mm      LTU250 BPM value in mm, used to compute photon energy. from BPMS:LTU1:250:X
    ebeamLTU450                         0 mm      LTU450 BPM value in mm, used to compute photon energy. from BPMS:LTU1:450:X
    ebeamLTUAngX                        0 mrad    LTU beam angle in mrad.
    ebeamLTUAngY                        0 mrad    LTU beam angle in mrad.
    ebeamLTUPosX                        0 mm      LTU beam position (BPMS:LTU1:720 through 750) in mm.
    ebeamLTUPosY                        0 mm      LTU beam position in mm.
    ebeamPhotonEnergy                   0 eV      computed photon energy, in eV
    ebeamPkCurrBC1                 33.661 Amps    Beam current in Amps.
    ebeamPkCurrBC2             2.9709e+08 Amps    Beam current in Amps.
    ebeamUndAngX                        0 mrad    Undulator launch feedback beam x-angle in mrad.
    ebeamUndAngY                        0 mrad    Undulator launch feedback beam y-angle in mrad.
    ebeamUndPosX                        0 mm      Undulator launch feedback (BPMs U4 through U10) beam x-position in mm.
    ebeamUndPosY                        0 mm      Undulator launch feedback beam y-position in mm.
    ebeamXTCAVAmpl                      0 MVolt   XTCAV Amplitude in MVolt.
    ebeamXTCAVPhase                     0 degrees XTCAV Phase in degrees.

    # Print summary of the cspad detector (uses PyDetector methods for creatining calib and image data)
    In [5]: evt.cspad.show_info()
    --------------------------------------------------------------------------------
    cspad: xpptut15, Run 54, Event 0, 11:37:12.4517, [140, 141, 41, 40]
    --------------------------------------------------------------------------------
    calib                 <0.01035> ADU     Calibrated data
    image               <0.0079081> ADU     Reconstruced 2D image from calibStore geometry
    raw                    <1570.2> ADU     Raw data
    size                  2.297e+06         Total size of raw data

    # Print summary of cspad detector calibration data (using PyDetector access methods) 
    In [6]: evt.cspad.calibData.show_info()
    areas                  <1.0077>         Pixel area correction factor
    bkgd                      <0.0>         
    common_mode        [   1.   25.   10.  100.]         Common mode parameters
    coords_x               <281.44> um      Pixel X coordinate
    coords_y               <753.19> um      Pixel Y coordinate
    coords_z                <1e+06> um      Pixel Z coordinate
    gain                      <1.0>         Pixel Gain factor from calibStore
    indexes_x              <863.22>         Pixel X index
    indexes_y              <869.53>         Pixel Y index
    ndim                          3         Number of dimensions of raw data
    pedestals              <1572.1> ADU     Pedestals from calibStore
    pixel_size               109.92 um      Pixel Size
    rms                    <4.9305> ADU     
    runnum                       54         Run number
    shape              (32, 185, 388)         Shape of raw data array
    size                  2.297e+06         Total size of raw data
    status             <0.00069396>         

    # Print summary of cspad detector calibration data (using PyDetector access methods) 
    In [7]: evt.cspad.configData.show_info()
    activeRunMode                       3         
    asicMask                           15         
    badAsicMask0                        0         
    badAsicMask1                        0         
    concentratorVersion        3.4055e+09         
    eventCode                          40         
    inactiveRunMode                     1         
    internalTriggerDelay                0         
    numAsicsRead                       16         
    numAsicsStored           <bound method ConfigV5.numAsicsStored of <psana.CsPad.ConfigV5 object at 0x7f4b0cd36cf8>>         Number of ASICs in given quadrant
    numQuads                            4         Total number of quadrants in setup
    numSect                            32         Total number of sections (2x1) in all quadrants
    payloadSize                1.1485e+06         
    protectionEnable                    1         
    protectionThresholds             list         
    quadMask                           15         
    quads                    <bound method ConfigV5.quads of <psana.CsPad.ConfigV5 object at 0x7f4b0cd36cf8>>         
    quads_shape                      list         
    roiMask                  <bound method ConfigV5.roiMask of <psana.CsPad.ConfigV5 object at 0x7f4b0cd36cf8>>         ROI mask for given quadrant
    roiMasks                    4.295e+09         
    runDelay                        58100         
    tdi                                 4         

This software was developed for the LCLS project.
If you use all or part of it, please give an appropriate acknowledgment.

@version $Id: PyDataSource.py  koglin@SLAC.STANFORD.EDU $

@author Koglin, Jason
"""
#------------------------------
__version__ = "$Revision:  $"
##-----------------------------

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

def _repr_value(value):
    """Represent a value for use in show_info method.
    """
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

def _is_psana_type(value):
    """True if the input is a psana data type
    """
    return hasattr(value, '__module__') and value.__module__.startswith('psana')

class PsanaTypeData(object):
    """Python representation of a psana data object (event or configStore data).
    """

    def __init__(self, typ_func):
        self._typ_func = typ_func
        module = self._typ_func.__module__.lstrip('psana.')
        type_name = self._typ_func.__class__.__name__

        if type_name in psana_doc_info[module]:
            self._info = psana_doc_info[module][type_name]
            self._attrs = [key for key in self._info.keys() if not key[0].isupper()]
        else:
            self._attrs = [attr for attr in dir(self._typ_func) if not attr.startswith('_')]
            self._info = {}

    @property
    def _attr_info(self):
        """Attribute information including the unit and doc information 
           and a str representation of the value.
        """
        _data = {}
        for attr in self._attrs:
            value = getattr(self, attr)
            if hasattr(value, '__class__') and value.__class__.__name__ == 'PsanaTypeData':
                _data.update({'_'.join([attr, a]): item for a, item in value._attr_info.items()})
            else:
                info = self._info.get(attr, {'unit': '', 'doc': ''})
                info['attr'] = attr
                info['value'] = value
                info['str'] = _repr_value(value)
                _data[attr] = info

        return _data

    @property
    def _values(self):
        """Dictionary of attributes: values. 
        """
        return {attr: getattr(self, attr) for attr in self._attrs}

    def show_info(self):
        """Show a table of the attribute, value, unit and doc information
        """
        items = sorted(self._attr_info.items(), key = operator.itemgetter(0))
        for attr, item in items:
            print '{attr:24s} {str:>12} {unit:7} {doc:}'.format(**item)
    
    def __getattr__(self, attr):
        if attr in self._attrs:
            value = getattr(self._typ_func, attr)
            try:
                value = value()
            except:
                pass

            if isinstance(value, list):
                values = []
                for val in value:
                    if _is_psana_type(val):
                        val = PsanaTypeData(val)
                    
                    values.append(val)
            
                return values

            elif _is_psana_type(value):
                return PsanaTypeData(value)

            else:
                return value

    def __dir__(self):
        all_attrs = set(self._attrs +
                        self.__dict__.keys() + dir(PsanaTypeData))
        return list(sorted(all_attrs))


class PsanaSrcData(object):
    """Dictify psana data for a given detector source.
       key_info: get_key_info(objclass) for faster evt data access.
    """
    def __init__(self, objclass, srcstr, key_info=None):
        self._srcstr = srcstr
        if not key_info:
            key_info = get_key_info(objclass)

        self._types = {}
        self._type_attrs = {}
        self._keys = key_info.get(srcstr)
        if self._keys:
            for (typ, src, key) in self._keys:
                if key:
                    typ_func = objclass.get(*item)
                else:
                    typ_func = objclass.get(typ, src)

                type_data = PsanaTypeData(typ_func)
                self._types[(typ,key)] = type_data 
                self._type_attrs.update({attr: (typ,key) for attr in type_data._attrs})

    @property
    def _attrs(self):
        attrs = []
        for type_data in self._types.values():
            attrs.extend(type_data._attrs)

        return attrs

    @property
    def _attr_info(self):
        """Attribute information including the unit and doc information 
           and a str representation of the value for all data types.
        """
        attr_info = {}
        for type_data in self._types.values():
            attr_info.update(**type_data._attr_info)

        return attr_info

    @property
    def _values(self):
        """Dictionary of attributes: values for all data types.
        """
        values = {}
        for type_data in self._types.values():
            values.update(**type_data._values)

        return values

    def show_info(self):
        """Show a table of the attribute, value, unit and doc information
           for all data types of the given source.
        """
        for type_data in self._types.values():
            type_data.show_info()

    def __getattr__(self, attr):
        item = self._type_attrs.get(attr)
        if item:
            return getattr(self._types.get(item), attr)

    def __dir__(self):
        all_attrs = set(self._type_attrs.keys() +
                        self.__dict__.keys() + dir(PsanaSrcData))
        return list(sorted(all_attrs))


class DataSource(object):
    """Python version of psana.DataSource with support for event and config
       data as well as PyDetector functions to access calibrated data.
    """

    _ds_attrs = ['empty', 'end', 'env', 'runs', 'steps']
    _env_attrs = ['calibDir', 'instrument', 'experiment','expNum']

    def __init__(self, data_source=None, **kwargs):
        self.load_run(data_source=data_source, **kwargs)

    def load_run(self, data_source=None, **kwargs):
        """Load a run with psana.
        """
        self._evtData = None
        self._current_evt = None
        self._evt_keys = {}
        self.data_source = DataSourceInfo(data_source=data_source, **kwargs)
        self._ds = psana.DataSource(str(self.data_source))
        self.epicsData = EpicsData(self._ds) 

        if self.data_source.indexed:
            self._Events = RunEvents(self, **kwargs)
        else:
            self._Events = Events(self, **kwargs)

        self._init_detectors()
        self._evt_time_last = (0,0)
        self._ievent = -1

    def reload(self):
        """Reload the current run.
        """
        self.load_run(str(self.data_source))

    def _init_detectors(self):
        """Initialize psana.Detector classes based on psana env information.
        """
        self._detectors = {}
        self.configData = ConfigData(self)
        self._aliases = self.configData._aliases
        for srcstr, item in self.configData._sources.items():
            alias = item.get('alias')
            self._add_dets(**{alias: srcstr})

    def _add_dets(self, **kwargs):
        for alias, srcstr in kwargs.items():
            try:
                det = Detector(self, alias)
                self._detectors.update({alias: det})
            except Exception as err:
                print 'Cannot add {:}:  {:}'.format(alias, srcstr) 
                traceback.print_exc()
    
    def show_info(self):
        print self.__repr__()
        for item in self._detectors.values():
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


class ConfigData(object):
    """ConfigData
    """
    _configStore_attrs = ['get','put','keys']
    
    def __init__(self, ds):
        configStore = ds.env().configStore()
        if (hasattr(ds, 'data_source') and ds.data_source.monshmserver):
            self._monshmserver = ds.data_source.monshmserver
        else:
            self._monshmserver = None 
        
        self._configStore = configStore
        self._key_info = get_key_info(configStore)

        # Build _config dictionary for each source
        self._config = {}
        self._modules = {}
        for attr, keys in self._key_info.items():
            config = PsanaSrcData(self._configStore, attr, key_info=self._key_info)
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

# to convert ipAddr int to address 
# import socket, struct
# s = key.src()
# socket.inet_ntoa(struct.pack('!L',s.ipAddr()))

        self._ipAddrPartition = src.ipAddr()
        self._bldMask = config.bldMask
        self._partition = {} 
        self._readoutGroup = {}
        self._sources = {}
        for source in config.sources:
            self._partition[str(source.src)] = source._values
            group = source.group
            srcstr = str(source.src)
            if group not in self._readoutGroup:
                self._readoutGroup[group] = {'srcs': [], 'eventCodes': []}

            self._readoutGroup[group]['srcs'].append(srcstr)

        # Find Aliases and update Partition
        self._srcAlias = {}
        if self._modules.get('Alias'):
            for type_name, keys in self._modules['Alias'].items():
                for typ, src, key in keys:
                    srcstr = str(src)
                    config = self._config[srcstr]
                    for item in config.srcAlias:
                        self._srcAlias[item.aliasName] = (item.src, src.ipAddr())

        self._aliases = {}
        for alias, item in self._srcAlias.items():
            src = item[0]
            ipAddr = item[1]
            srcstr = str(src)
            alias = re.sub('-|:|\.| ','_', alias)
            if srcstr in self._partition:
                self._partition[srcstr]['alias'] = alias
                if srcstr.find('NoDetector') == -1:
                    self._aliases[alias] = srcstr
            elif ipAddr != self._ipAddrPartition or self._monshmserver:
                # add data sources not in partition that come from recording nodes
                group = -1
                self._partition[srcstr] = {'src': src, 'group': group, 'alias': alias}
                self._aliases[alias] = srcstr
                if group not in self._readoutGroup:
                    self._readoutGroup[group] = {'srcs': [], 'eventCodes': []}

                self._readoutGroup[group]['srcs'].append(srcstr)
                self._sources[srcstr] = {'group': group}

        # Determine data sources and update aliases
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
                if srcstr not in self._sources:
                    self._sources[srcstr] = {}
                self._sources[srcstr].update(**item)

        # Make dictionary of src: alias for sources with config objects 
        self._config_srcs = {}
        for attr, item in self._sources.items():
            config = self._config.get(attr)
            if config:
                self._config_srcs[item['alias']] = attr
    
        self._output_maps = {}
        self._evr_pulses = {}
        self._eventcodes = {}

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
                self._eventcodes.update({eventcode.code: eventcode._values})
                if eventcode.isReadout:
                    group = eventcode.readoutGroup
                    if group not in self._readoutGroup:
                        self._readoutGroup[group] = {'srcs': [], 'eventCodes': []}
                    self._readoutGroup[group]['eventCodes'].append(eventcode.code)

            for output_map in config.output_maps:
                map_key = (output_map.module,output_map.conn_id)
                if output_map.source.Pulse:
                    pulse_id = output_map.source_id
                    pulse = config.pulses[pulse_id]
                    evr_info = { 'evr_width': pulse.width*pulse.prescale/119.e6, 
                                 'evr_delay': pulse.delay*pulse.prescale/119.e6, 
                                 'evr_polarity': pulse.polarity}
                else:
                    pulse_id = None
                    pulse = None
                    evr_info = {'evr_width': None, 'evr_delay': None, 'evr_polarity': None}

                self._output_maps[map_key] = {attr: getattr(output_map,attr) for attr in map_attrs} 
                self._output_maps[map_key].update(**evr_info) 

        # Assign evr info to the appropriate sources
        if len(self._modules['EvrData'][IOCconfig_type]) > 1:
            print 'WARNING: More than one EvrData.{:} objects'.format(IOCconfig_type)

        typ, src, key = self._modules['EvrData'][IOCconfig_type][0]
        srcstr = str(src)
        config = self._config[srcstr]
        for ch in config.channels:
            map_key = (ch.output.module, ch.output.conn_id)
            for i in range(ch.ninfo):
                src = ch.infos[i]
                srcstr = str(src)
                self._sources[srcstr]['map_key'] = map_key
                for attr in ['evr_width', 'evr_delay', 'evr_polarity']:
                    self._sources[srcstr][attr] = self._output_maps[map_key][attr]

        for group, item in self._readoutGroup.items():
            for srcstr in item['srcs']:
                if srcstr in self._sources:
                    self._sources[srcstr]['eventCodes'] = item['eventCodes']
        
        # Get control data
        if self._modules.get('ControlData'):
            type_name, keys = self._modules['ControlData'].items()[0]
            typ, src, key = keys[0]
            config = self._config[str(src)]
            self._controlData = config._values

        if self._modules.get('SmlData'):
            type_name, keys = self._modules['SmlData'].items()[0]
            typ, src, key = keys[0]
            config = self._config[str(src)]
            self._smlData = config._values


    def __getattr__(self, attr):
        if attr in self._config_srcs:
            return self._config[self._config_srcs[attr]]

        if attr in self._configStore_attrs:
            return getattr(self._configStore, attr)
        
    def __dir__(self):
        all_attrs = set(self._configStore_attrs +
                        self._config_srcs.keys() + 
                        self.__dict__.keys() + dir(ConfigData))
        return list(sorted(all_attrs))


class RunEvents(object):
    """Event iterator from ds.runs() for indexed data 

       No support yet for multiple runs in a data_source
    """
    def __init__(self, ds, **kwargs):
        self._ds_runs = []
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

    _init_attrs = ['get', 'keys'] #  'run' depreciated
    _event_attrs = ['EventId', 'Evr', 'L3T']

    def __init__(self, ds): 
        self._ds = ds
        
    @property
    def EventId(self):
        return EventId(self._ds._current_evt)

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
        return self._ds._detectors

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

    def next(self, *args, **kwargs):
        return self._ds.next(*args, **kwargs)
 
    def __iter__(self):
        return self

    def __str__(self):
        return  '{:}, Run {:}, Event {:}, {:}, {:}'.format(self._ds.data_source.exp, 
                self._ds.data_source.run, self._ds._ievent, str(self.EventId), str(self.Evr))

    def __repr__(self):
        repr_str = '{:}: {:}'.format(self.__class__.__name__, str(self))
        return '< '+repr_str+' >'

    def __getattr__(self, attr):
        if attr in self._dets:
            return self._dets[attr]
        
        if attr in self._init_attrs:
            return getattr(self._ds._current_evt, attr)

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
                         self.__dict__.keys() + dir(L3Tdata))
        
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
    _properties = ['timef64', 'nsec', 'sec']

    def __init__(self, evt):

        self._EventId = evt.get(psana.EventId)

    @property
    def timef64(self):
        return np.float64(self.time[0])+np.float64(self.time[1])/1.e9 

    @property
    def nsec(self):
        """nanosecond part of event time.
        """
        return self.time[0]

    @property
    def sec(self):
        """second part of event time.
        """
        return self.time[1]

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
    """Includes epicsData, configStore, evrConfig info 
       Uses full ds in order to be able to access epicsData info on
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
       
        self.configData = getattr(ds.configData, self._alias)

        try:
            self._pydet = psana.Detector(self._srcname, ds._ds.env())
        except:
            self._pydet = None

        if not hasattr(self._pydet, 'dettype'):
            self._det_class = None
            self._tabclass = 'evtData'
        elif self._pydet.dettype in [16, 17]:
            self._det_class = WaveformData
            self._calib_class = WaveformCalibData
            self._tabclass = 'detector'
        elif self._pydet.dettype:
            self._det_class = ImageData
            self._calib_class = ImageCalibData
            self._tabclass = 'detector'
        else:
            self._det_class = None
            self._tabclass = 'evtData'

    @property
    def _xray_attrs(self):
        """Attributes
        """
        attrs = {}
        for attr, item in self.configData._attr_info.items():
            if item['str'] != 'list' and not item['str'].startswith('<bound'):
                attrs.update({attr: item['value']})
        
        return attrs

    @property
    def _xray_dims(self):
        """Dimensions of data attributes.
        """
        if self._det_class == WaveformData:
            dims_dict = {
                    'waveform':  (['channel', 't'], self.wftime),
                    }
        
        elif self._det_class == ImageData:
            raw_dims = (['sensor', 'row', 'column'],
                        [])
            if self.image is not None:
                xaxis = ((np.arange(self.image.shape[0])-self.image.shape[0]/2.) \
                          *self.calibData.pixel_size/1000.)
                yaxis = ((np.arange(self.image.shape[1])-self.image.shape[1]/2.) \
                          *self.calibData.pixel_size/1000.)
            else:
                xaxis = None
                yaxis = None

            image_dims = (['X', 'Y'],
                          [xaxis, yaxis])
            dims_dict = {
                    'image':     image_dims,
                    'calib':     raw_dims,
                    'raw':       raw_dims,
                    'areas':     raw_dims,
                    'bkgd':      raw_dims,
                    'coords_x':  raw_dims,
                    'coords_y':  raw_dims,
                    'coords_z':  raw_dims,
                    'gain':      raw_dims,
                    'indexes_x': raw_dims,
                    'indexes_y': raw_dims,
                    'pedestals': raw_dims,
                    'rms':       raw_dims,
                    }
        else:
            dims_dict = {}
                    
        return dims_dict

    @property
    def _attrs(self):
        """Attributes of psana.Detector functions if relevant, and otherwise
           attributes of raw psana event keys for the given detector.
        """
        if self._tabclass:
            attrs = getattr(self, self._tabclass)._attrs
        
        return attrs

    def next(self, *args, **kwargs):
        return self._ds.next(*args, **kwargs)
 
    def __iter__(self):
        return self

    def monitor(self, nevents=-1, sleep=0.2):
        """Monitor detector attributes continuously with show_info function.
        """ 
        ievent = nevents
        try:
            while ievent != 0:
                self.next()
                try:
                    self.show_info()
                except:
                    pass
                
                if ievent < nevents and sleep:
                    time.sleep(sleep)

                ievent -= 1

        except KeyboardInterrupt:
            ievent = 0

    def show_all(self):
        print '-'*80
        print '{:}: {:}'.format(self._alias, str(self._ds.current))
        print '-'*80
        print 'Event Data:'
        print '-'*18
        self.evtData.show_info()
        if self._tabclass == 'detector':
            print '-'*80
            print 'Processed Data:'
            print '-'*18
            self.detector.show_info()
            print '-'*80
            print 'Calibration Data:'
            print '-'*18
            self.calibData.show_info()

        if self.epicsData:
            print '-'*80
            print 'Epics Data:'
            print '-'*18
            self.epicsData.show_info()

    def show_info(self):
        print '-'*80
        print '{:}: {:}'.format(self._alias, str(self._ds.current))
        print '-'*80
        getattr(self, self._tabclass).show_info()

    @property
    def evtData(self):
        """Tab accessible raw data from psana event keys.
        """
        return PsanaSrcData(self._ds._current_evt, self._srcstr, key_info=self._ds._evt_keys)

    @property
    def epicsData(self):
        return getattr(self._ds.epicsData, self._alias)

    @property
    def detector(self):
        """Raw, calib and image data using psana.Detector class
        """
        if self._pydet:
            return self._det_class(self._pydet, self._ds._current_evt)
        else:
            return None

    @property
    def calibData(self):
        """Calibration data using psana.Detector class
        """
        if self._pydet:
            return self._calib_class(self._pydet, self._ds._current_evt)
        else:
            return None

    def __str__(self):
        return '{:} {:}'.format(self._alias, str(self._ds.current))

    def __repr__(self):
        return '< {:}: {:} >'.format(self.__class__.__name__, str(self))

    def __getattr__(self, attr):
        if attr in self._attrs:
            return getattr(getattr(self, self._tabclass), attr)
        
        if attr in self._ds.current._event_attrs:
            return getattr(self._ds.current, attr)

    def __dir__(self):
        all_attrs =  set(self._attrs+
                         self._ds.current._event_attrs +
                         self.__dict__.keys() + dir(Detector))
        
        return list(sorted(all_attrs))


class WaveformData(object):
    """Tab accessibile dictified psana.Detector object.
       
       Attributes come from psana.Detector 
       with low level implementation done in C++ or python.  
       Boost is used for the C++.
    """

    _attrs = ['raw', 'waveform', 'wftime'] 

    _attr_info = {
            'waveform':    {'doc': 'Waveform array',
                            'unit': 'V'},
            'wftime':      {'doc': 'Waveform sample time',
                            'unit': 's'},
            } 

    def __init__(self, det, evt):
        self._evt = evt
        self._det = det

    @property
    def instrument(self):
        """Instrument to which this detector belongs.
        """
        return self._det.instrument()

    def show_info(self):
        """Show information for relevant detector attributes.
        """
        try:
            items = sorted(self._attr_info.items(), key = operator.itemgetter(0))
            for attr, item in items:
                fdict = {'attr': attr, 'unit': '', 'doc': ''}
                fdict.update(**item)
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
                         self.__dict__.keys() + dir(WaveformData))
        
        return list(sorted(all_attrs))


class WaveformCalibData(object):
    """Calibration data using psana.Detector access.
    """

    _attrs = ['runnum'] 

    _attr_info = {
            'runnum':      {'doc': 'Run number',
                            'unit': ''}
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
            items = sorted(self._attr_info.items(), key = operator.itemgetter(0))
            for attr, item in items:
                fdict = {'attr': attr, 'unit': '', 'doc': ''}
                fdict.update(**item)
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
                         self.__dict__.keys() + dir(WaveformCalibData))
        
        return list(sorted(all_attrs))


class ImageData(object):
    """Tab accessibile dictified psana Detector object.
       
       Attributes come from psana.Detector  
       with low level implementation done in C++ or python.  
       Boost is used for the C++.
    """
    _attrs = ['image', 'raw', 'calib', 'size'] 
    _attr_info = {
            'size':        {'doc': 'Total size of raw data', 
                            'unit': ''},
            'raw':         {'doc': 'Raw data', 
                            'unit': 'ADU'},
            'calib':       {'doc': 'Calibrated data',
                            'unit': 'ADU'},
            'image':       {'doc': 'Reconstruced 2D image from calibStore geometry',
                            'unit': 'ADU'},
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

    def show_info(self):
        """Show information for relevant detector attributes.
        """
        if self.size > 0:
            items = sorted(self._attr_info.items(), key = operator.itemgetter(0))
            for attr, item in items:
                fdict = {'attr': attr, 'unit': '', 'doc': ''}
                fdict.update(**item)
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
                         self.__dict__.keys() + dir(ImageData))
        
        return list(sorted(all_attrs))


class ImageCalibData(object):
    """Calibration Data from psana Detector object.
    """

    _attrs = ['shape', 'size', 'ndim', 'pedestals', 'rms', 'gain', 'bkgd', 'status',
              'common_mode', 'runnum',
              'areas', 'indexes_x', 'indexes_y', 'pixel_size',
              'coords_x', 'coords_y', 'coords_z', 
             ] 
    _attr_info = {
            'runnum':      {'doc': 'Run number',
                            'unit': ''},
            'shape':       {'doc': 'Shape of raw data array', 
                            'unit': ''},
            'size':        {'doc': 'Total size of raw data', 
                            'unit': ''},
            'ndim':        {'doc': 'Number of dimensions of raw data', 
                            'unit': ''},
            'pedestals':   {'doc': 'Pedestals from calibStore', 
                            'unit': 'ADU'},
            'rms':         {'doc': '', 
                            'unit': 'ADU'},
            'gain':        {'doc': 'Pixel Gain factor from calibStore', 
                            'unit': ''},
            'bkgd':        {'doc': '', 
                            'unit': ''},
            'status':      {'doc': '',
                            'unit': ''},
            'common_mode': {'doc': 'Common mode parameters', 
                            'unit': ''},
            'areas':       {'doc': 'Pixel area correction factor', 
                            'unit': ''},
            'indexes_x':   {'doc': 'Pixel X index', 
                            'unit': ''},
            'indexes_y':   {'doc': 'Pixel Y index', 
                            'unit': ''},
            'pixel_size':  {'doc': 'Pixel Size',
                            'unit': 'um'},
            'coords_x':    {'doc': 'Pixel X coordinate', 
                            'unit': 'um'},
            'coords_y':    {'doc': 'Pixel Y coordinate', 
                            'unit': 'um'},
            'coords_z':    {'doc': 'Pixel Z coordinate', 
                            'unit': 'um'},
            } 

    def __init__(self, det, evt):
        self._evt = evt
        self._det = det

    @property
    def instrument(self):
        """Instrument to which this detector belongs.
        """
        return self._det.instrument()

    def set_do_offset(do_offset=True):
        """Not sure what do offset does?
        """
        self._det.set_do_offset(do_offset=do_offset)

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
            items = sorted(self._attr_info.items(), key = operator.itemgetter(0))
            for attr, item in items:
                fdict = {'attr': attr, 'unit': '', 'doc': ''}
                fdict.update(**item)
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
                         self.__dict__.keys() + dir(ImageCalibData))
        
        return list(sorted(all_attrs))


class EpicsConfig(object):
    """Tab Accessible configStore Epics information.
       Currently relatively simple, but expect this to be expanded
       at some point with more PV config info with daq update.
    """

    _pv_attrs = ['description', 'interval', 'pvId']

    def __init__(self, configStore):

        # move to PsanaSrcData objects
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


class EpicsData(object):
    """Epics data from psana epicsStore.
       e.g., 
         epicsStore = EpicsData(ds)
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
            return PvData(attr_dict, self._ds, level=1)
        
        if attr in dir(self._ds.env().epicsStore()):
            return getattr(self._ds.env().epicsStore(),attr)

    def __dir__(self):
        all_attrs = set(self._attrs +
                        dir(self._ds.env().epicsStore()) +
                        self.__dict__.keys() + dir(EpicsData))
        return list(sorted(all_attrs))


class PvData(object):
    """Epics PV Data.
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
                return PvData(attr_dict, self._ds, level=self._level+1)

    def __repr__(self):
        return self.get_info()

    def __dir__(self):
        all_attrs = set(self._attrs +
                        self.__dict__.keys() + dir(PvData))
        return list(sorted(all_attrs))


class EpicsStorePV(object):
    """Epics PV access from epicsStore. 
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


