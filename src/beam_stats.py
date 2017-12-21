"""
Beam statistics methods
"""
import logging
import traceback
from IPython.core.debugger import Tracer

def load_exp_sum(exp, instrument=None, path=None, nctype='drop_sum', save=True):
    """
    Load drop stats summary for all runs
    
    Arguments
    ---------
    exp : str
        experiment name

    Parameters
    ----------
    instrument : str
        instrument name.  default = exp[:3] 
    
    path : str
        path of run summary files

    save : bool
        Save drop_summary file 

    """
    import xarray_utils
    import os
    import glob
    import xarray as xr
    import numpy as np
    import pandas as pd
    if not instrument:
        instrument = exp[0:3]
    if not path:
        path = os.path.join('/reg/d/psdm/',instrument,exp,'results','nc')

    files = sorted(glob.glob('{:}/run*_{:}.nc'.format(path,nctype)))

    axstats = {}
    dvars = []
    for f in files:                                      
        x = xr.open_dataset(f, engine='h5netcdf')
        try:
            if x.data_vars:
                axstats[x.run] = x
                dvars += x.data_vars
                print('Loading Run {:}: {:}'.format(x.run, f))
            else:
                print('Skipping Run {:}: {:}'.format(x.run, f))
        except:
            print('cannot do ', f)

    dvars = sorted(list(set(dvars)))
    ax = []
    aattrs = {}
    for attr in dvars:
        adf = {}
        for run, x in axstats.items():
            if attr in x.data_vars:
                if attr not in aattrs:
                    aattrs[attr] = {}
                adf[run] = x[attr].to_pandas()
                for a, val in x[attr].attrs.items():
                    if val:
                        aattrs[attr][a] = val 

        ax.append(xr.Dataset(adf).to_array().to_dataset(name=attr))

    x = xr.merge(ax)
    del(ax)
    x = xarray_utils.resort(x).rename({'variable': 'run'})
    x.attrs['experiment'] = exp
    x.attrs['instrument'] = instrument
    x.attrs['expNum'] = axstats.values()[0].attrs['expNum']
    x.coords['dvar'] = dvars
    dattrs = {a.replace('_detected',''): str(a) for a in axstats.values()[0].attrs.keys() if a.endswith('detected')}
    advar = {}
    for attr in dattrs:
        advar[attr] = {}
        for dvar in dvars:
            advar[attr][dvar] = np.zeros((len(x.run)), dtype='bool')
    
    rinds = dict(zip(x.run.values, range(x.run.size)))
    for run, xo in axstats.items():
        irun = rinds[run]
        for attr in dattrs:
            for dvar in xo.attrs.get(attr+'_detected',[]):
                advar[attr][dvar][irun] = True
    
    for attr in aattrs:
        try:
            if attr in x:
                for a in ['doc', 'unit', 'alias']:
                    x[attr].attrs[a] = aattrs[attr].get(a, '')
        except:
            print('cannot add attrs for {:}'.format(attr))

    for attr in dattrs:
        data = np.array([advar[attr][a] for a in dvars]).T
        x[attr] = (('run', 'dvar'), data)
    
    if save:
        sum_file = '{:}/drop_summary.nc'.format(path)
        x.attrs['file_name'] = sum_file
        print('Saving drop_summary file: {:}'.format(sum_file))
        x.to_netcdf(sum_file, engine='h5netcdf')

    return x

def build_drop_stats(x, min_detected=2,  
        alert=True, to_name=None, from_name=None, html_path=None,
        report_name='drop_summary', path=None, engine='h5netcdf'):
    """
    """
    import os
    from requests import post
    from os import environ
    update_url = environ.get('BATCH_UPDATE_URL')

    if isinstance(x, str):
        x = load_exp_sum(x)
   
    exp = x.attrs.get('experiment')
    instrument = x.attrs.get('instrument')
    run = x.attrs.get('run')
    expNum = x.attrs.get('expNum')
    
    if not path:
        path = os.path.join('/reg/d/psdm/',instrument,exp,'results','nc')

    if not os.path.isdir(path):
        os.mkdir(path)
    
    filename = '{:}'.format(report_name)
    h5file = os.path.join(path,report_name+'.nc')
     
    report_notes = ['Report includes:']
    try:
        from build_html import Build_html
        from h5write import runlist_to_str
        b = Build_html(x, h5file=h5file, filename=report_name, path=html_path,
                title=exp+' Drop Summary', subtitle='Drop Summary')
        dattrs = [attr for attr, a in x.data_vars.items() if 'dvar' in a.dims]
        batch_counters = {}
        webattrs = [instrument.upper(), expNum, exp, report_name, 'report.html'] 
        weblink='http://pswww.slac.stanford.edu/experiment_results/{:}/{:}-{:}/{:}/{:}'.format(*webattrs)
        for attr in dattrs:
            inds = (x[attr].to_pandas().sum() >= min_detected)
            attrs = [a for a, val in inds.iteritems() if val]
            gattrs = {}
            for a in attrs:
                dattr = a.split('_')[0]
                if dattr not in gattrs:
                    gattrs[dattr] = []
                gattrs[dattr].append(a)
                
            if attr == 'timing_error':
                if attrs:
                    report_notes.append(' - ALERT:  off-by-one timing errors detected \n {:}'.format([str(a) for a in sorted(gattrs)]))
                    batch_str = ', '.join(['<a href={:}#{:}_data>{:}</a>'.format(weblink, a,a) \
                                            for a in sorted(gattrs)])
                    batch_attr = '<a href={:}>{:}</a>'.format(weblink,'Off-by-one detected')
                    #batch_attr = 'Off-by-one detected'
                    batch_counters[batch_attr] = [batch_str, 'red']
                #    for a in attrs:
                #        report_notes.append('      {:}'.format(a))
                else:
                    report_notes.append(' - No off-by-one timing errors detected')
                    batch_attr = 'Off-by-one'
                    batch_counters[batch_attr] = ['None Detected', 'green']
            else:
                if attrs:
                    report_notes.append(' - {:} detected \n {:}'.format(attr.replace('_',' '), [str(a) for a in sorted(gattrs)]))
                #    for a in attrs:
                #        report_notes.append('      {:}'.format(a))
                else:
                    report_notes.append(' - No {:} detected'.format(attr.replace('_',' ')))

            for alias, attrs in gattrs.items():
                if attr == 'timing_error':
                    tbl_type = alias+'_'+attr
                    attr_cat = 'Alert Off-by-one Error'
                else:
                    tbl_type = attr
                    attr_cat = alias
                doc = ['Runs with {:} for {:} attributes'.format(attr.replace('_',' '), alias)]
                df = x[attr].to_pandas()[attrs]
                for a in attrs:
                    if a in x:
                        da = df[a]
                        name = '_'.join(a.split('_')[1:])
                        aattrs = x[a].attrs
                        runstr = runlist_to_str(da.index[da.values])
                        doc.append(' - {:} runs with {:}: {:} [{:}] \n       [{:}]'.format(df[a].sum(), name, 
                                    aattrs.get('doc',''), aattrs.get('unit',''), runstr))
                howto = ['x["{:}"].to_pandas()[{:}]'.format(attr, attrs)]
                df = df.T.rename({a: '_'.join(a.split('_')[1:]) for a in attrs}).T
                b.add_table(df, attr_cat, tbl_type, tbl_type, doc=doc, howto=howto, hidden=True)

        b.to_html(h5file=h5file, report_notes=report_notes)

        if update_url:
            try:
                print('Setting batch job counter output: {:}'.format(batch_counters))
                post(update_url, json={'counters' : batch_counters})
            except:
                traceback.print_exc('Cannot update batch submission json counter info to {:}'.format(update_url))


        if not to_name:
            if isinstance(alert, list) or isinstance(alert, str):
                to_name = alert
            else:
                to_name = from_name
            
        if alert and to_name is not None:
            try:
                alert_items = {name.lstrip('Alert').lstrip(' '): item for name, item in b.results.items() \
                        if name.startswith('Alert')}
                if alert_items:
                    from psmessage import Message

                    message = Message('Alert {:} Off-by-one Errors'.format(exp))
                    message('')
                    for alert_type, item in alert_items.items():
                        message(alert_type)
                        message('='*len(message._message[-1]))
                        for name, tbl in item.get('table',{}).items():
                            df = tbl.get('DataFrame')
                            message('')
                            message('* '+name)
                            for a in df:
                                da = df[a]
                                message('  -{:} -- {:} Errors:'.format(a,da.sum()))
                                message('       [{:}]'.format(runlist_to_str(da.index[da.values])))
                    
                    message('')
                    message('See report:')
                    message(b.weblink)
                    print('Sending message to {:} from {:}'.format(to_name, from_name))
                    print(str(message))
                    
                    if to_name:
                        message.send_mail(to_name=to_name, from_name=from_name)

                    return message
            
            except:
                traceback.print_exc('Cannot send alerts: \n {:}'.format(str(message)))

        return b
    
    except:
        traceback.print_exc('Cannot build beam drop stats')

def get_beam_stats(exp, run, default_modules={}, 
        flatten=True, refresh=True,
        drop_code='ec162', drop_attr='delta_drop', nearest=5, drop_min=3,  
        report_name=None, path=None, engine='h5netcdf', 
        wait=None, timeout=False,
        **kwargs):
    """
    Get drop shot statistics to detected dropped shots and beam correlated detectors.

    Parameters
    ----------
    drop_code : str
        Event code that indicates a dropped shot with no X-rays
    
    drop_attr: str
        Name for number of shots off from dropped shot.
    
    nearest : int
        Number of nearest events to dropped shot to analayze

    drop_min : int
        Minumum number of dropped shots to perform analysis


    """
    from xarray_utils import set_delta_beam
    import PyDataSource
    import xarray as xr
    import numpy as np
    import pandas as pd
    import time
    import os
    from requests import post
    from os import environ
    update_url = environ.get('BATCH_UPDATE_URL')
    
    time0 = time.time() 
    logger = logging.getLogger(__name__)
    logger.info(__name__)
    
    ds = PyDataSource.DataSource(exp=exp, run=run, default_modules=default_modules, wait=wait, timeout=timeout)
#    if update_url:
#        batch_counters = {'Status': ['Loading small data...','yellow']}
#        post(update_url, json={'counters' : batch_counters})
    
    x = load_small_xarray(ds, refresh=refresh, path=path)
   
    nevents = ds.nevents
    if drop_code not in x or not x[drop_code].values.any():
        logger.info('Skipping beam stats analysis for {:}'.format(ds))
        logger.info('  -- No {:} present in data'.format(drop_code))
        try:
            batch_counters = {'Warning': ['No dropped shots present in {:} events'.format(nevents), 'red']}
            print('Setting batch job counter output: {:}'.format(batch_counters))
            if update_url:
                post(update_url, json={'counters' : batch_counters})
        except:
            traceback.print_exc('Cannot update batch submission json counter info to {:}'.format(update_url))

        return x

    set_delta_beam(x, code=drop_code, attr=drop_attr)
    xdrop = x.where(abs(x[drop_attr]) <= nearest, drop=True)
    ntimes = xdrop.time.size
    try:
        ndrop = int(np.sum(xdrop.get(drop_code)))
    except:
        ndrop = 0

    if ndrop < drop_min:
        logger.info('Skipping beam stats analysis for {:}'.format(ds))
        logger.info('  -- only {:} events with {:} in {:} events'.format(ndrop, drop_code, nevents))
        try:
            batch_counters = {'Warning': ['Only {:} dropped shots present in {:} events'.format(ndrop, nevents), 'red']}
            print('Setting batch job counter output: {:}'.format(batch_counters))
            if update_url:
                post(update_url, json={'counters' : batch_counters})
        except:
            traceback.print_exc('Cannot update batch submission json counter info to {:}'.format(update_url))
        
        return x

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
                # for not just use rawsum for 'Epix10ka' and  'Jungfrau' until full
                # development of gain switching in Detector module
                if devName.startswith('Opal') or devName in ['Epix10ka','Jungfrau']:
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
            evt_info = '{:8} of {:8} -- {:8.3f} sec, {:8.3f} events/sec'.format(itime+1, 
                    ntimes, time_next-time0, nupdate/dtime)
            logger.info(evt_info)
            time_last = time_next 
#            if update_url:
#                batch_counters = {'Status': [evt_info,'yellow']}
#                post(update_url, json={'counters' : batch_counters})

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

    h5file = os.path.join(path,report_name+'.nc')
   
    print('Build {:} {:} {:}'.format(exp,run, xdrop))
    b = build_beam_stats(exp=exp, run=run, xdrop=xdrop, 
            report_name=report_name, h5file=h5file, path=path, **kwargs)
    
    try:
        xdrop.to_netcdf(h5file, engine=engine)
        logger.info('Saving file to {:}'.format(h5file))
        print('Saving file to {:}'.format(h5file))
    except:
        traceback.print_exc('Cannot save to {:}'.format(h5file))
   
    return xdrop

def build_beam_stats(exp=None, run=None, xdrop=None, instrument=None,
        report_name=None, h5file=None, path=None, 
        alert=True, to_name=None, from_name=None, html_path=None,
        make_scatter=False,
        pulse=None, **kwargs):
    """
    """
    import os
    import numpy as np
    import pandas as pd
    import xarray as xr
    from requests import post
    from os import environ
    update_url = environ.get('BATCH_UPDATE_URL')

    if not path:
        if not exp:
            exp = str(xdrop.attrs.experiment)
        if not instrument:
            if xdrop is None:
                instrument = exp[:3]
            else:
                instrument = str(xdrop.attrs.instrument)

        path = os.path.join('/reg/d/psdm/',instrument,exp,'results','nc')

    if not run:
        if not xdrop:
            print('Error:  Need to specify exp and run or alternatively xdrop DataSet') 
            return None
        else:
            run = xdrop.attrs.run

    if not report_name:
        report_name = 'run{:04}_drop_stats'.format(run)
    
    if not h5file:
        h5file = os.path.join(path,report_name+'.nc')
 
    if not xdrop:
        import xarray as xr
        xdrop = xr.open_dataset(h5file, engine='h5netcdf')

    exp = xdrop.attrs.get('experiment')
    instrument = xdrop.attrs.get('instrument')
    run = xdrop.attrs.get('run')
    expNum = xdrop.attrs.get('expNum')

    try:
        from build_html import Build_html
   
        b = Build_html(xdrop, h5file=h5file, filename=report_name, path=html_path)
        drop_attr = xdrop.attrs.get('drop_attr', 'ec162')
        if 'XrayOff' not in xdrop and drop_attr in xdrop:
            xdrop.coords['XrayOff'] = (xdrop[drop_attr] == True)
            xdrop.coords['XrayOff'].attrs['doc'] = 'Xray Off for events with {:}'.format(drop_attr)
            xdrop.coords['XrayOn'] = (xdrop[drop_attr] == False)
            xdrop.coords['XrayOn'].attrs['doc'] = 'Xray On for events without {:}'.format(drop_attr)

        report_notes = []
        try:
            energy_mean= xdrop.EBeam_ebeamPhotonEnergy.where(xdrop.XrayOn, drop=True).values.mean()
            energy_std= xdrop.EBeam_ebeamPhotonEnergy.where(xdrop.XrayOn, drop=True).values.std()
            report_notes.append('Energy = {:5.0f}+={:4.0f} eV'.format(energy_mean, energy_std))
            charge_mean= xdrop.EBeam_ebeamCharge.where(xdrop.XrayOn, drop=True).values.mean()
            charge_std= xdrop.EBeam_ebeamCharge.where(xdrop.XrayOn, drop=True).values.std()
            report_notes.append('Charge = {:5.2f}+={:4.2f} mA'.format(charge_mean, charge_std))
        except:
            pass

        report_notes.append('Report includes:')
        
        b._xstats = b.add_delta_beam(pulse=pulse)
        corr_attrs = xdrop.attrs.get('beam_corr_detected')
        print('Beam Correlations Detected {:}'.format(corr_attrs))
        if corr_attrs:
            pulse = xdrop.attrs.get('beam_corr_attr')
            if not pulse:
                pulse = 'FEEGasDetEnergy_f_21_ENRC'
                if pulse not in xdrop or xdrop[pulse].sum() == 0:
                    pulse = 'FEEGasDetEnergy_f_11_ENRC'
                if pulse not in xdrop or xdrop[pulse].sum() == 0:
                    pulse = None
            
            if pulse:
                corr_attrs = [a for a in corr_attrs if a == pulse or not a.startswith('FEEGasDetEnergy')]
                corr_attrs.append(pulse)
                if 'PhaseCavity_charge1' in corr_attrs and 'PhaseCavity_charge2' in corr_attrs:
                    corr_attrs.remove('PhaseCavity_charge2')
                print('Adding Detector Beam Correlations {:}'.format(corr_attrs))
                try:
                    b.add_detector(attrs=corr_attrs, catagory=' Beam Correlations', confidence=0.3,
                        cut='XrayOn',
                        make_timeplot=False, make_histplot=False, make_table=False, 
                        make_scatter=make_scatter)
                except:
                    traceback.print_exc('Cannot make beam correlations {;}'.format(attrs))

        webattrs = [instrument.upper(), expNum, exp, report_name, 'report.html'] 
        weblink='http://pswww.slac.stanford.edu/experiment_results/{:}/{:}-{:}/{:}/{:}'.format(*webattrs)
        #batch_counters['report'] = ['See <a href={:}>Run{:04} Report</a>'.format(weblink,run),'blue']
        batch_counters = {}
        try:
            offbyone_detectors = list(sorted(set([str(xdrop[a].attrs.get('alias')) for a in xdrop.timing_error_detected])))
            if offbyone_detectors:
                #batch_str = ', '.join(['<a href={:}#{:}_data>{:}</a>'.format(weblink, attr,attr) \
                #                        for attr in offbyone_detectors])
                batch_str = ', '.join([attr for attr in offbyone_detectors])
                batch_attr = '<a href={:}#{:}_data>{:}</a>'.format(weblink, "%20Alert%20Timing%20Error", 'Off-by-one detected')
                batch_counters[batch_attr] = [batch_str, 'red']
                report_notes.append(' - Off-by-one detected: '+ str(offbyone_detectors))
                for a in sorted(xdrop.timing_error_detected):
                    report_notes.append('    + '+a)
        except:
            offbyone_detectors = []

        try:
            drop_detectors = list(sorted(set([str(xdrop[a].attrs.get('alias')) for a in xdrop.drop_shot_detected])))
            if drop_detectors:
                batch_str = ', '.join([attr for attr in drop_detectors])
                #batch_str = ', '.join(['<a href={:}#{:}_data>{:}</a>'.format(weblink, attr,attr) \
                #                        for attr in drop_detectors])
                batch_counters['Dropped shot detected'] = [batch_str, 'green']
                report_notes.append(' - Dropped shot detected: '+str(drop_detectors))
                for a in sorted(xdrop.drop_shot_detected):
                    report_notes.append('    + '+a)
        except:
            drop_detectors = []

        try:
            beam_detectors = list(sorted(set([str(xdrop[a].attrs.get('alias')) for a in xdrop.beam_corr_detected])))
            if beam_detectors:
                batch_str = ', '.join([attr for attr in beam_detectors])
                #batch_str = ', '.join(['<a href={:}#{:}_data>{:}</a>'.format(weblink, attr,attr) \
                #                        for attr in beam_detectors])
                batch_attr = '<a href={:}#{:}_data>{:}</a>'.format(weblink, "%20Beam%20Correlations", 'Beam Correlated detected')
                batch_counters[batch_attr] = [batch_str, 'green']
                report_notes.append(' - Beam correlated detected: '+str(beam_detectors))
                for a in sorted(xdrop.beam_corr_detected):
                    report_notes.append('    + '+a)
        except:
            beam_detectors = []

        if b.results:
            # only make reports if not empty
            b.to_html(h5file=h5file, report_notes=report_notes)


        if update_url:
            try:
                print('Setting batch job counter output: {:}'.format(batch_counters))
                post(update_url, json={'counters' : batch_counters})
            except:
                traceback.print_exc('Cannot update batch submission json counter info to {:}'.format(update_url))

        if not to_name:
            if isinstance(alert, list) or isinstance(alert, str):
                to_name = alert
            else:
                to_name = from_name
            
        alert_items = {name.lstrip(' ').lstrip('Alert').lstrip(' '): item for name, item in b.results.items() \
                if name.lstrip(' ').startswith('Alert')}
       
        if alert_items and offbyone_detectors != ['EBeam']:
            if alert and to_name is not 'None':
                message = None
                try:
                    from psmessage import Message
                    import pandas as pd
                    message = Message('Alert {:} Run {:}: {:}'.format(exp,run, ','.join(alert_items.keys())))
                    event_times = pd.to_datetime(b._xdat.time.values)
                    begin_time = event_times.min()
                    end_time = event_times.max()
                    run_time = (end_time-begin_time).seconds
                    minutes,fracseconds = divmod(run_time,60)

                    message('- Run Start:  {:}'.format(begin_time.ctime()))
                    message('- Run End:    {:}'.format(end_time.ctime()))
                    message('- Duration:   {:} seconds ({:02.0f}:{:02.0f})'.format(run_time, minutes, fracseconds))
                    message('- Total events: {:}'.format(len(event_times) ) )
                    message('')
 
                    for alert_type, item in alert_items.items():
                        message(alert_type)
                        message('='*len(message._message[-1]))
                        for name, tbl in item.get('figure',{}).items():
                            doc = tbl.get('doc')
                            message('* '+doc[0])
                            message('   - '+doc[1])
                    
                    message('')
                    message('See report:')
                    message(b.weblink)
                    
                    print(str(message))
                    try:
                        print('Sending message to {:} from {:}'.format(to_name, from_name))
                        print(str(message))
                        message.send_mail(to_name=to_name, from_name=from_name)
                    except:
                        traceback.print_exc('ERROR Sending message to {:} from {:}'.format(to_name, from_name))

                except:
                    traceback.print_exc('Cannot send alerts: \n {:}'.format(str(message)))

    except:
        traceback.print_exc('Cannot build drop report')

    return b


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
        x.coords['XrayOff'] = (x.ec162 == True)
        x.coords['XrayOff'].attrs['doc'] = 'Xray Off for events with ec162'
        x.coords['XrayOn'] = (x.ec162 == False)
        x.coords['XrayOn'].attrs['doc'] = 'Xray On for events without ec162'

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


def save_exp_stats(exp, instrument=None, path=None, find_corr=True):
    """
    Save drop stats summaries for all runs 
    """
    import os
    import glob
    import xarray as xr
    from xarray_utils import find_beam_correlations
    if not instrument:
        instrument = exp[0:3]
    if not path:
        path = os.path.join('/reg/d/psdm/',instrument,exp,'results','nc')

    files = sorted(glob.glob('{:}/run*_{:}.nc'.format(path, 'drop_stats')))

    for f in files:                                      
        x = xr.open_dataset(f, engine='h5netcdf')
        try:
            run = x.run
            save_file='{:}/run{:04}_{:}.nc'.format(path, run, 'drop_sum')
            print(save_file)
            if not find_corr and glob.glob(save_file):
                xstats = xr.open_dataset(save_file, engine='h5netcdf')
                xout = xstats.copy(deep=True) 
                xstats.close()
                del xstats
                for avar, da in x.data_vars.items():
                    try:
                        if avar in xout:
                            for a in ['doc', 'unit', 'alias']:
                                val = x[avar].attrs.get(a, '') 
                                if isinstance(val, unicode):
                                    val = str(val)
                                xout[avar].attrs[a] = val 
                    except:
                        print('Cannot add attrs for {:}'.format(avar))
                
                for attr, val in x.attrs.items():
                    try:
                        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], unicode):
                            val = [str(v) for v in val]
                        elif isinstance(val, unicode):
                            val = str(val)
                        xout.attrs[attr] = val
                    except:
                        print('Cannot add attrs for {:}'.format(attr))

                xout.to_netcdf(save_file, engine='h5netcdf')

            else:
                xout = x.copy(deep=True) 
                x.close()
                del x
                xstats = find_beam_correlations(xout, groupby='ec162', cut='XrayOn', save_file=save_file)
                xout.to_netcdf(f, engine='h5netcdf')
           
        except:
            traceback.print_exc('Cannot do {:}'.format(f))

    return load_exp_sum(exp)


