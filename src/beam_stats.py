"""
Beam statistics methods
"""
import logging
import traceback
from IPython.core.debugger import Tracer

def get_beam_stats(exp, run, default_modules={}, 
        flatten=True, refresh=True,
        alert=True, to_name=None, from_name=None, html_path=None,
        drop_code='ec162', drop_attr='delta_drop', nearest=5,  
        report_name=None, path=None, engine='h5netcdf', 
        pulse=None, 
        **kwargs):
    """
    Get drop shot statistics to detected dropped shots and beam correlated detectors.

    Parameters
    ----------
    """
    from xarray_utils import set_delta_beam
    import PyDataSource
    import xarray as xr
    import numpy as np
    import pandas as pd
    import time
    import os
    time0 = time.time() 
    logger = logging.getLogger(__name__)
    logger.info(__name__)
    
    ds = PyDataSource.DataSource(exp=exp, run=run, default_modules=default_modules)
    x = load_small_xarray(ds, refresh=refresh, path=path)
    set_delta_beam(x, code=drop_code, attr=drop_attr)
    xdrop = x.where(abs(x[drop_attr]) <= nearest, drop=True)
    ntimes = xdrop.time.size
    dets = {}
    flatten_list = []
    for det, detector in ds._detectors.items():
        methods = {}
        try:
            det_info = detector._xarray_info.get('dims',{})
            if detector._pydet is None: 
                logger.info(det, 'pydet not implemented')
            elif detector._pydet.__module__ == 'Detector.AreaDetector':
                srcstr = detector._srcstr 
                srcname = srcstr.split('(')[1].split(')')[0]
                devName = srcname.split(':')[1].split('.')[0]
                if devName.startswith('Opal'):
                    method = 'rawsum'
                    name = '_'.join([det, method])
                    methods[name] = method
                    xdrop[name] = (('time'), np.zeros([ntimes]))
                    xdrop[name].attrs['doc'] = '{:} sum of raw data'.format(det)
                    xdrop[name].attrs['unit'] = 'ADU'
                    xdrop[name].attrs['alias'] = det
                else:
                    method = 'count'
                    name = '_'.join([det, method])
                    methods[name] = method
                    xdrop[name] = (('time'), np.zeros([ntimes]))
                    xdrop[name].attrs['doc'] = '{:} sum of pedestal corrected data'.format(det)
                    xdrop[name].attrs['unit'] = 'ADU'
                    xdrop[name].attrs['alias'] = det

            elif detector._pydet.__module__ == 'Detector.WFDetector':
                srcstr = detector._srcstr 
                srcname = srcstr.split('(')[1].split(')')[0]
                devName = srcname.split(':')[1].split('.')[0]
                if devName == 'Acqiris':
                    nch = detector.configData.nbrChannels 
                    for method in ['peak_height', 'peak_time']:
                        name = '_'.join([det, method])
                        methods[name] = method
                        xdrop[name] = (('time', det+'_ch',), np.zeros([ntimes, nch]))
                        xdrop[name].attrs['doc'] = 'Acqiris {:}'.format(method.replace('_',' '))
                        xdrop[name].attrs['unit'] = 'V'
                        xdrop[name].attrs['alias'] = det
                
                elif devName == 'Imp':
                    nch = 4
                    method = 'amplitudes'
                    name = '_'.join([det, method])
                    methods[name] = method
                    xdrop[name] = (('time', det+'_ch'), np.zeros([ntimes, nch]))
                    attr = 'waveform'
                    xdrop[name].attrs['doc'] = 'IMP filtered amplitudes'
                    xdrop[name].attrs['unit'] = 'V'
                    xdrop[name].attrs['alias'] = det
 
            elif detector._pydet.__module__ == 'Detector.UsdUsbDetector':
                srcstr = detector._srcstr 
                srcname = srcstr.split('(')[1].split(')')[0]
                devName = srcname.split(':')[1].split('.')[0]
                if devName == 'USDUSB':
                    method = 'encoder_values'
                    name = '_'.join([det, method])
                    flatten_list.append(name)
            
            elif detector._pydet.__module__ == 'Detector.DdlDetector':
                srcstr = detector._srcstr 
                srcname = srcstr.split('(')[1].split(')')[0]
                devName = srcname.split(':')[1].split('.')[0]
                if devName == 'Gsc16ai':
                    method = 'channelValue'
                    name = '_'.join([det, method])
                    flatten_list.append(name)

            elif detector._pydet.__module__ == 'Detector.IpimbDetector':
                pass
            
            else:
                logger.info('{:} Not implemented'.format(det))
        
        except:
            logger.info('Error with config of {:}'.format(det))
        
        if methods:
            dets[det] = methods
    
    ds.reload()
    times = zip(xdrop.sec.values,xdrop.nsec.values,xdrop.fiducials.values)
    nupdate = 100
    time_last = time0
    logger.info('Loading: {:}'.format(dets.keys()))
    for itime, t in enumerate(times):
        if itime % nupdate == nupdate-1:
            time_next = time.time()
            dtime = time_next-time_last
            logger.info('{:8} of {:8} -- {:8.3f} sec, {:8.3f} events/sec'.format(itime+1, 
                    ntimes, time_next-time0, nupdate/dtime))
            time_last = time_next 

        evt = ds.events.next(t)
        for det, methods in dets.items():
            detector = evt._dets.get(det)
            if detector:
                for name, method in methods.items():
                    try:
                        xdrop[name][itime] = globals()[method](detector) 
                    except:
                        traceback.print_exc('Error with {:} {:}'.format(name, method))

    # Flatten waveform data with channels
    if flatten:
        #for name in xdrop.data_vars:
        for det, methods in dets.items():
            for name, method in methods.items():
                if len(xdrop[name].dims) == 2:
                    flatten_list.append(name)
        
        for name in flatten_list:
            if name in xdrop and len(xdrop[name].dims) == 2:
                nch = xdrop[name].shape[1]
                if nch <= 16:
                    for ich in range(nch):
                        xdrop['{:}_ch{:}'.format(name,ich)] = xdrop[name][:,ich]
                    del xdrop[name]

    if not path:
        path = os.path.join(ds.data_source.res_dir,'nc')
    elif path == 'home':
        path = os.path.join(os.path.expanduser('~'), 'RunSummary', 'nc')

    if not os.path.isdir(path):
        os.mkdir(path)
    
    if not report_name:
        report_name = 'run{:04}_drop_stats'.format(ds.data_source.run)

    filename = '{:}'.format(report_name)
    h5file = os.path.join(path,report_name+'.nc')
      
    try:
        from build_html import Build_html
        b = Build_html(xdrop, h5file=h5file, filename=report_name, path=html_path)
        b.add_delta_beam(pulse=pulse)
        b.to_html(h5file=h5file)
        #xdrop = b._xdat

        try:
            xdrop.to_netcdf(h5file, engine=engine)
            logger.info('Saving file to {:}'.format(h5file))
        except:
            traceback.print_exc('Cannot save to {:}/{:}'.format(path,filename))

        if not to_name:
            if isinstance(alert, list) or isinstance(alert, str):
                to_name = alert
            else:
                to_name = from_name
            
        if alert and to_name is not 'None':
            try:
                alert_items = {name.lstrip('Alert').lstrip(' '): item for name, item in b.results.items() \
                        if name.startswith('Alert')}
                if alert_items:
                    from psmessage import Message
                    message = Message('Alert {:} Run {:}: {:}'.format(exp,ds.data_source.run, ','.join(alert_items.keys())))
                    message('')
                    for alert_type, item in alert_items.items():
                        message(alert_type)
                        message('='*len(message._message[-1]))
                        for name, tbl in item.get('table',{}).items():
                            doc = tbl.get('doc')
                            message('* '+doc[0])
                            message('   - '+doc[1])
                    
                    message('')
                    message('See report:')
                    message(b.weblink)
                    logger.info('Sending message to {:} from {:}'.format(to_name, from_name))
                    logger.info(str(message))
                    
                    if to_name:
                        message.send_mail(to_name=to_name, from_name=from_name)

            except:
                traceback.print_exc('Cannot save to {:}/{:}'.format(path,filename))

        return message

    except:
        try:
            xdrop.to_netcdf(h5file, engine=engine)
            logger.info('Saving file to {:}'.format(h5file))
        except:
            traceback.print_exc('Cannot save to {:}/{:}'.format(path,filename))
        return xdrop

def load_small_xarray(ds, path=None, filename=None, refresh=False, 
        engine='h5netcdf', **kwargs):
    """Load small xarray Dataset with PyDataSource.DataSource. 
    
    """
    import xarray as xr
    import glob
    import os
    if not path:
        path = os.path.join(ds.data_source.res_dir,'nc')
    elif path == 'home':
        path = os.path.join(os.path.expanduser('~'), 'RunSummary', 'nc')

    if not filename:
        filename = 'run{:04}_smd.nc'.format(ds.data_source.run)

    if refresh or not glob.glob(os.path.join(path, filename)):
        x = make_small_xarray(ds, path=path, filename=filename, **kwargs)
    else:
        x = xr.open_dataset(os.path.join(path,filename), engine='h5netcdf')

    return x

def make_small_xarray(self, auto_update=True,
        add_dets=True, add_counts=False, add_1d=True,
        ignore_unused_codes=True,
        drop_code='ec162', drop_attr='delta_drop', 
        path=None, filename=None, save=True, engine='h5netcdf', 
        nevents=None):
    """Make Small xarray Dataset.
    Parameters
    ----------
    ignore_unused_codes : bool
        If true drop unused eventCodes except drop_code [default=True]
    """
    from xarray_utils import set_delta_beam
    import numpy as np
    import pandas as pd
    import time
    import os
    time0 = time.time() 
    self.reload()
    if not nevents:
        nevents = self.nevents
    
    logger = logging.getLogger(__name__)
    cnames = {code: 'ec{:}'.format(code) for  code in self.configData._eventcodes}
    try:
        code = int(drop_code[2:])
        if code not in cnames:
            cnames[code] = 'ec{:}'.format(code) 
    except:
        traceback.print_exc('Cannot add drop_code = '.format(code))
    data = {cnames[code]: np.zeros(nevents, dtype=bool) for code in cnames}
    #data = {cnames[code]: np.zeros(nevents, dtype=bool) for code in self.configData._eventcodes}
    data1d = {}
    dets1d = {}
    coords = ['fiducials', 'sec', 'nsec']
    dets = {'EventId': {'names': {a:a for a in coords}}}
    coords += data.keys()
    for det, attrs in dets.items():
        for name, attr in attrs['names'].items():
            data.update({name: np.zeros(nevents, dtype=int)})
    
    ievt = 0
    if add_dets:
        evt = self.events.next()
        for det, detector in self._detectors.items(): 
            #srcstr = detector._srcstr 
            #srcname = srcstr.split('(')[1].split(')')[0]
            #devName = srcname.split(':')[1].split('.')[0]
            if det == 'EBeam':
                attrs = ['ebeamCharge', 'ebeamDumpCharge', 'ebeamEnergyBC1', 'ebeamEnergyBC2', 
                         'ebeamL3Energy', 'ebeamLTU250', 'ebeamLTU450', 'ebeamLTUAngX', 'ebeamLTUAngY', 
                         'ebeamLTUPosX', 'ebeamLTUPosY', 'ebeamPhotonEnergy', 
                         'ebeamPkCurrBC1', 'ebeamPkCurrBC2', 'ebeamUndAngX', 'ebeamUndAngY', 
                         'ebeamUndPosX', 'ebeamUndPosY', 'ebeamXTCAVAmpl', 'ebeamXTCAVPhase']
                attrs = {det+'_'+attr: attr for attr in attrs}
                #attrs = {'EBeam_'+attr.lstrip('ebeam'): attr for attr in attrs}
                dets.update({'EBeam': {'names': attrs}})
                for name, attr in attrs.items():
                    data.update({name: np.zeros(nevents)})
            elif det == 'FEEGasDetEnergy':
                attrs = ['f_11_ENRC', 'f_12_ENRC', 'f_21_ENRC', 'f_22_ENRC', 'f_63_ENRC', 'f_64_ENRC']
                attrs = {det+'_'+attr: attr for attr in attrs}
                #attrs = {'GasDet_'+''.join(attr.split('_')[0:2]): attr for attr in attrs}
                dets.update({'FEEGasDetEnergy': {'names': attrs}})
                for name, attr in attrs.items():
                    data.update({name: np.zeros(nevents)})
            elif det == 'PhaseCavity':
                attrs = ['charge1', 'charge2', 'fitTime1', 'fitTime2']
                #attrs = {'PhaseCavity_'+attr: attr for attr in attrs}
                attrs = {det+'_'+attr: attr for attr in attrs}
                dets.update({'PhaseCavity': {'names': attrs}})
                for name, attr in attrs.items():
                    data.update({name: np.zeros(nevents)})
            else:
                try:
                    #detector = self._detectors.get(det)
                    while det not in evt._attrs and ievt < min([1000,nevents]):
                        evt.next(publish=False, init=False)
                        ievt += 1
                    if auto_update and hasattr(detector, '_update_xarray_info'):
                        try:
                            detector._update_xarray_info()
                        except:
                            pass

                    info = detector._xarray_info
                    attr_info = info.get('dims',{})
                    attrs = {}
                    attrs1d = {}
                    for attr, vals in attr_info.items():
                        try:
                            b = list(vals[1])
                        except:
                            b = [vals[1]]
                        try:
                            if len(vals[0]) == 0:
                                name = det+'_'+attr
                                attrs[name] = attr
                                data[name]  = np.zeros(nevents)
                            elif add_1d and len(vals[0]) == 1 and b[0] <= 16:
                                nch = b[0]
                                name = det+'_'+attr
                                attrs1d[name] = attr
                                data1d[name]  = np.zeros((nevents, nch))
                        except:
                            logger.info('Cannot add {:}, {:} -- {:}'.format(det, attr, vals))
                    if add_counts and not attrs and 'corr' in detector._attrs:
                        detector.add.count('corr')
                        for attr, vals in attr_info.items():
                            if len(vals[0]) == 0:
                                name = det+'_'+attr
                                attrs[name] = attr
                                data[name]  = np.zeros(nevents)
                    if attrs:
                        dets[det] = {'names': attrs}
                    if attrs1d:
                        dets1d[det] = {'names': attrs1d}

                except:
                    logger.info('Cannot add {:}'.format(det))

        self.reload()
    
    nupdate = 100
    time_last = time0
    logger.info('Loading: {:}'.format(dets.keys()))
    for i, evt in enumerate(self.events):       
        if i >= nevents:
            break
        if i % nupdate == nupdate-1:
            time_next = time.time()
            dtime = time_next-time_last
            logger.info('{:8} of {:8} -- {:8.3f} sec, {:8.3f} events/sec'.format(i+1, 
                    nevents, time_next-time0, nupdate/dtime))
            time_last = time_next 

        for code in evt.Evr.eventCodes_strict:
            if code in cnames:
                data[cnames[code]][i] = code

        if add_dets:
            for det, attrs in dets.items():
                if det == 'EventId' or det in evt._attrs:
                    detector = getattr(evt, det)
                    for name, attr in attrs['names'].items():
                        try:
                            data[name][i] = getattr(detector, attr)
                        except:
                            pass
            if add_1d:
                for det, attrs in dets1d.items():
                    if det in evt._attrs:
                        detector = getattr(evt, det)
                        for name, attr in attrs['names'].items():
                            try:
                                data1d[name][i] = getattr(detector, attr)
                            except:
                                pass

    df = pd.DataFrame(data)
    x = df.to_xarray()
    x = x.rename({'index':'time'})
    x['time'] = [np.datetime64(int(sec*1e9+nsec), 'ns') for sec,nsec in zip(x.sec,x.nsec)]
    x.attrs['data_source'] = self.data_source.__str__()
    x.attrs['instrument'] = self.data_source.instrument.upper()
    x.attrs['run'] = self.data_source.run
    x.attrs['experiment'] = self.data_source.exp
    x.attrs['expNum'] = self.expNum
    # add attributes
    for det, item in dets.items():
        detector = self._detectors.get(det)
        if det == 'EventId':
            continue
        try:
            detector._update_xarray_info()
            det_info = detector._xarray_info['dims']
            for name, attr in item['names'].items():
                info = det_info.get(attr, ([],(),{},))
                if len(info) == 3:
                    attrs = info[2]
                    x[name].attrs['attr'] = attr
                    x[name].attrs['alias'] = det
                    x[name].attrs.update(attrs)
        except:
            logger.info('Error updating scalar data for {:}: {:}'.format(det, item))
    # add 1d
    if add_1d:
        for det, item in dets1d.items():
            detector = self._detectors.get(det)
            try:
                detector._update_xarray_info()
                det_info = detector._xarray_info['dims']
                for name, attr in item['names'].items():
                    info = det_info.get(attr, ([],(),{},))
                    if len(info[0]) == 1:
                        x[name] = (('time', det+'_'+info[0][0]), data1d.get(name))
                    if len(info) == 3:
                        attrs = info[2]
                        x[name].attrs['attr'] = attr
                        x[name].attrs['alias'] = det
                        x[name].attrs.update(attrs)
            except:
                logger.info('Error updating 1D data for {:}: {:}'.format(det, item))

    x = x.set_coords(coords)
    if ignore_unused_codes:
        drop_codes = []
        for code, ec in cnames.items():
            if ec != drop_code and not x[ec].any():
                drop_codes.append(code)
                x = x.drop(ec)
        if drop_codes:
            logger.info('Dropping unused eventCodes: {:}'.format(drop_codes))

    if 'ec162' in x:
        set_delta_beam(x, code=drop_code, attr=drop_attr)

    self.x = x

    if save:
        if not path:
            path = os.path.join(self.data_source.res_dir,'nc')
        elif path == 'home':
            path = os.path.join(os.path.expanduser('~'), 'RunSummary', 'nc')

        if not os.path.isdir(path):
            os.mkdir(path)
        
        if not filename:
            filename = 'run{:04}_smd.nc'.format(self.data_source.run)

        try:
            self.x.to_netcdf(os.path.join(path,filename), engine='h5netcdf')
        except:
            traceback.print_exc('Cannot save to {:}/{:}'.format(path,filename))

    return self.x

def rawsum(self, attr='raw'):
    """Return raw sum of AreaDetector
    """
    import numpy as np
    return np.sum(getattr(self, attr))

def count(self, attr='corr'):
    """Return pedestal corrected count of AreaDetector
    """
    import numpy as np
    return np.sum(getattr(self, attr))

def peak_height(self):
    """Max value of each waveform for WFDetector.
    """
    return self.waveform.max(axis=1)

def peak_time(self):
    """Time of max of each waveform for WFDetector.
    """
    import numpy as np
    return np.array([self.wftime[ch][index] for ch,index in enumerate(self.waveform.argmax(axis=1))])

def amplitudes(self):
    """
    Returns an array of max values of a set of waveforms.
    """
    return filtered(self).max(axis=1)

def filtered(self, signal_width=10):
    """
    Returns filtered waveform for WFDetector.
    """
    from scipy import signal
    import numpy as np
    waveform = self.waveform
    af = []
    hw = signal_width/2
    afilter = np.array([-np.ones(hw),np.ones(hw)]).flatten()/(hw*2)
    nch = waveform.shape[0]
    for ich in range(nch):
        wf = waveform[ich]
        f = -signal.convolve(wf, afilter)
        f[0:len(afilter)+1] = 0
        f[-len(afilter)-1:] = 0
        af.append(f[hw:wf.size+hw])
    
    return  np.array(af)



