# standard python modules
import os
import operator
import time
import traceback
import numpy as np


"""
DEVELOPMENT MODULE:  Direct write to_hdf5 to be xarray compatible without using xarray.
"""
#from pylab import *

def runlist_to_str(runs, portable=False):
    """Convert list of runs to string representation.
    """
    if portable:
        strmulti='-'
        strsingle='_'
    else:
        strmulti=':'
        strsingle=','

    runs = np.array(sorted(set(runs)))
    runsteps = runs[1:] - runs[0:-1]
    runstr = '{:}'.format(runs[0])
    for i in range(len(runsteps)):    
        if i == 0:
            if runsteps[i] == 1:
                grouped_runs = True
            else:
                grouped_runs = False
            
            if i == len(runsteps)-1:
                runstr += '{:}{:}'.format(strsingle,runs[i+1])

        elif i == len(runsteps)-1:
            if grouped_runs:
                if runsteps[i] == 1:
                    runstr += '{:}{:}'.format(strmulti,runs[i+1])
                else:
                    runstr += '{:}{:}{:}{:}'.format(strmulti,runs[i],strsingle,runs[i+1])
            else:
                runstr += '{:}{:}:{:}'.format(strsingle,runs[i],runs[i+1])

        elif i > 0:
            if runsteps[i] == 1:
                if not grouped_runs:
                    runstr += '{:}{:}'.format(strsingle,runs[i])

                grouped_runs = True
            else:
                if grouped_runs:
                    runstr += '{:}{:}'.format(strmulti,runs[i])
                else:
                    runstr += '{:}{:}'.format(strsingle,runs[i])

                grouped_runs = False

    return runstr

def runstr_to_array(runstr, portable=False):
    """Convert run string to list.
    """
    if portable:
        strmulti='-'
        strsingle='_'
    else:
        strmulti=':'
        strsingle=','
    
    runs = []
    for item in runstr.split(strsingle):
        runrange = item.split(strmulti)
        if len(runrange) == 2:
            for run in range(int(runrange[0]),int(runrange[1])+1):
                runs.append(run)
        else:
            runs.append(int(runrange[0]))

    return np.array(sorted(runs))

def read_netcdfs(files, dim='time', transform_func=None, engine='h5netcdf'):
    """
    Read netcdf files and return concatenated xarray Dataset object

    Parameters
    ----------
    files : str or list
        File name(s) which may have '*' and '?' to be used for matching
    dim : str
        Diminsion along which to concatenate Dataset objects
        Default = 'time'
    transform_func : object
        Method to transform each Dataset before concatenating
    engine : str
        Engine for loading files.  default = 'h5netcdf'

    """
    import glob
    import xarray as xr
    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with xr.open_dataset(path, engine=engine) as ds:
            if 'nevents' in ds.attrs:
                try:
                    ds = ds.isel(time=slice(None, ds.nevents))
                except:
                    traceback.print_exc()
                    print 'cannot select', ds.nevents
            ds['time'] = [np.datetime64(int(sec*1e9+nsec), 'ns') for sec,nsec in zip(ds.sec.values,ds.nsec.values)]
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob.glob(files))
    datasets = []
    xattrs = None
    for p in paths:
        try:
            xo = process_one_path(p)
            datasets.append(xo)
            if p == min(paths, key=len):
                xattrs = xo.attrs
        except:
            traceback.print_exc()
            print 'Cannot open {:}'.format(p)
   
    #try:
    #    x = resort(xr.concat(datasets, dim))
    #except:
    x = xr.merge(datasets)
    if xattrs:
        x.attrs.update(**xattrs)
    x = resort(x)
    x = x.set_coords([a for a in x.data_vars if a.endswith('present')]) 
    
    if 'ichunk' in x.attrs:
        x.attrs.pop('ichunk')
    x.attrs['files'] = paths
    return x 

#ds = PyDataSource.DataSource('exp=cxij4915:run=49:smd')
def open_h5netcdf(file_name=None, path='', file_base=None, exp=None, run=None, 
        h5folder='scratch', subfolder='nc', chunk=False, combine=False, summary=False, **kwargs):
    """
    Open hdf5 file with netcdf4 convention using builtin xarray engine h5netcdf.
    """
    import xarray as xr
    if exp:
        instrument = exp[0:3]
    
    if not file_name and not path:
        path = '/reg/d/psdm/{:}/{:}/{:}/{:}'.format(instrument, exp, h5folder, subfolder)
    
    if combine:
        if not file_name:
            if file_base:
                file_name = os.path.join(path,file_base+'_*.nc')
            else:
                file_name = '{:}/run{:04}_*.nc'.format(path, int(run))

        return read_netcdfs(file_name)
 
    elif chunk and run:
        file_names = '{:}/run{:04}_c*.nc'.format(path, int(run))
        if True:
            x = read_netcdfs(file_names)
        else:
            import glob
            files = glob.glob(file_names)
            xo = xr.open_dataset(files[0], engine='h5netcdf')
            x = resort(xr.open_mfdataset(file_names, engine='h5netcdf'))
            x.attrs.update(**xo.attrs)
            x.attrs.pop('ichunk')
            x.attrs['files'] = files

        return x

    else:
        if not file_name:
            if not file_base:
                file_base = 'run{:04}'.format(int(run))
                if summary:
                    file_base += '_sum'

            file_name = os.path.join(path,file_base+'.nc')

        return xr.open_dataset(file_name, engine='h5netcdf')



def to_h5netcdf(xdat=None, ds=None, file_name=None, path=None, 
        h5folder='scratch', subfolder='nc', **kwargs):
    """Write hdf5 file with netcdf4 convention using builtin xarray engine h5netcdf.
    """
    if xdat:
        if xdat.__class__.__name__ == 'DataSource':
            # 1st arg is actually PyDataSource.DataSource
            ds = xdat
            xdat = None

    if not xdat:
        if not ds:
            import PyDataSource
            ds = PyDataSource.DataSource(**kwargs)

        xdat = to_xarray(ds, **kwargs)
    
    if not path:
        path = '/reg/d/psdm/{:}/{:}/{:}/{:}/'.format(xdat.instrument,xdat.experiment,h5folder,subfolder)

    if not os.path.isdir(path):
        os.mkdir(path)

    if not file_name:
        if 'ichunk' in xdat.attrs:
            file_name = '{:}/run{:04}_c{:03}.nc'.format(path, int(xdat.run[0]), xdat.attrs['ichunk'])

        else:
            # add in sets for mulit run
            file_base = xdat.attrs.get('file_base')
            if not file_base:
                if 'stat' in xdat.dims:
                    run = xdat.run
                    file_base = 'run{:04}_sum'.format(int(run))
                else:
                    run = sorted(set(xdat.run.values))[0]
                    file_base = 'run{:04}'.format(int(run))
                
            file_name = os.path.join(path,file_base+'.nc')
    
    xdat.to_netcdf(file_name, engine='h5netcdf')

    return xdat

def process_one_file(file_name, transform_func=None, engine='h5netcdf'):
    """Load one file
    """
    import xarray as xr
    # use a context manager, to ensure the file gets closed after use
    with xr.open_dataset(file_name, engine=engine) as ds:
        if 'nevents' in ds.attrs:
            try:
                ds = ds.isel(time=slice(None, ds.nevents))
            except:
                traceback.print_exc()
                print 'cannot select', ds.nevents
        ds['time'] = [np.datetime64(int(sec*1e9+nsec), 'ns') for sec,nsec in zip(ds.sec.values,ds.nsec.values)]
        # transform_func should do some sort of selection or
        # aggregation
        if transform_func is not None:
            ds = transform_func(ds)
        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        ds.load()
        return ds

def merge_datasets(file_names, engine='h5netcdf', 
            save_file=None, cleanup=False, quiet=False):
    """
    """
    import xarray as xr
    datasets = []
    xattrs = {}
    for file_name in file_names:
        #det = file_name.split('/run')[1].split('_')[2].split('.')[0]
        try:
            print 'processing', file_name
            xo = process_one_file(file_name, engine=engine)
            det = xo.attrs.get('alias', 'base')
            if det not in xattrs:
                xattrs[det] = xo.attrs
            datasets.append(xo)
            #if file_name == min(paths, key=len):
            #    xattrs = xo.attrs
        except:
            traceback.print_exc()
            print 'Cannot open {:}'.format(file_name)
            return xo, xattrs 
   
    x = resort(xr.merge(datasets))
    if 'base' in xattrs:
        x.attrs = xattrs['base']

    if save_file and isinstance(save_file, str):
        if not quiet:
            print 'Saving', save_file
        x.to_netcdf(save_file, engine=engine)

    return x

def read_chunked(run=None, path=None, exp=None, dim='time', 
        h5folder='scratch', subfolder='nc',
        omit_attrs=['ichunk', 'nevents'], 
        merge=False, quiet=False,
        save=True, save_path=None,
        transform_func=None, engine='h5netcdf'):
    """
    Read netcdf files and return concatenated xarray Dataset object

    Parameters
    ----------
    files : str or list
        File name(s) which may have '*' and '?' to be used for matching
    dim : str
        Diminsion along which to concatenate Dataset objects
        Default = 'time'
    transform_func : object
        Method to transform each Dataset before concatenating
    engine : str
        Engine for loading files.  default = 'h5netcdf'

    """
    import xarray as xr
    import glob
    if exp:
        instrument = exp[0:3]
    if not path:
        path = '/reg/d/psdm/{:}/{:}/{:}/{:}/Run{:04}/'.format(instrument, 
                exp,h5folder,subfolder,run)

    files = glob.glob('{:}/run{:04}_*.nc'.format(path, run)) 
    #dets = set([a.lstrip('{:}/run{:04}_'.format(path,run)).split('_')[1].split('.')[0] for a in files])
    datachunks = []
    chunks = set([int(a.lstrip('{:}/run{:04}_'.format(path,run)).lstrip('C').split('.')[0].split('_')[0]) for a in files])
    xattrs = {}
    for chunk in chunks:
        merge_chunk = False
        if not merge:
            file_name = '{:}//run{:04}_C{:02}.nc'.format(path,run,chunk)
            try:
                if not quiet:
                    print 'Loading chunk', chunk
                xchunk = process_one_file(file_name)
            except:
                merge_chunk = True

        if merge or merge_chunk:
            if not quiet:
                print 'Merging chunk', chunk
            file_names = glob.glob('{:}//run{:04}_C{:02}_*.nc'.format(path,run,chunk)) 
            save_file = '{:}//run{:04}_C{:02}.nc'.format(path,run,chunk)
            xchunk = merge_datasets(file_names, save_file=save_file, quiet=quiet)
    
        datachunks.append(xchunk)
        try:
            xattrs[chunk] = xchunk.attrs
        except:
            traceback.print_exc()
            return xchunk

    try:
        if not quiet:
            print 'Concat all chunks'
        x = resort(xr.concat(datachunks, dim))
    except:
        traceback.print_exc()
        print 'Concat Failed'
        return datachunks

    if xattrs:
        x.attrs.update(**xattrs[0])
    
    x = x.set_coords([a for a in x.data_vars if a.endswith('present')]) 
    
    nsteps = x.attrs.get('nsteps', 1)
#    steps = set([x.step.values])
#    if 'steps' not in x:
#        x.coords['steps'] = steps

#    if nsteps > 1:
#        uniq_attrs = {}
#        for det, a in xattrs.items():
#            for attr, item in a.items():
#                try:
#                    vals = set(item.values())
#                    if len(vals) > 1 and attr not in omit_attrs:
#                        if det not in uniq_attrs:
#                            uniq_attrs[det] = {}
#                        uniq_attrs[det][attr] = item.values()
#                except:
#                    pass
        
    for attr in omit_attrs:
        if attr in x.attrs:
            del x.attrs[attr]
    
    if save:
        if isinstance(save, str):
            save_file = save
        else:
            if not save_path:
                save_path = os.path.dirname(os.path.dirname(path))
            save_file = '{:}//run{:04}.nc'.format(save_path,run)
        
        if not quiet:
            print 'Saving Run {:} to {:}'.format(run, save_path) 
        x.to_netcdf(save_file, engine=engine)

    return x


def to_hdf5(self, save=True, cleanup=False, base=False, **kwargs):
    """Write PyDataSource.DataSource to hdf5 file.
    """
    path, file_base = write_hdf5(self, **kwargs)
    exp = self.data_source.exp
    run = self.data_source.run
#    x = open_h5netcdf(path=path, file_base=file_base, combine=True) 
#    file_name = os.path.join(path,file_base+'_*.nc')
    try:
        x = read_chunked(path=path, file_base=file_base)
    except:
        print 'Could not read files', path, file_base
        traceback.print_exc()

    if save:
        try:
            file_name = os.path.join(path,file_base+'.nc')
            x.to_netcdf(file_name, engine='h5netcdf')
        except:
            print 'Could not save xarray data', file_name
            traceback.print_exc()
            return x

        if cleanup:
            try:
                print 'Need to implement cleanup'
            except:
                print 'Could not cleanup individual files'
                traceback.print_exc()

    return x

def get_config_xarray(ds=None, exp=None, run=None, path=None, file_name=None, 
        h5folder='results', subfolder='nc', reload=False, summary=False,
        no_create=False, **kwargs):
    """
    Get xarray run config.
    """
    import glob
    import xarray as xr
    if ds:
        exp = self.data_source.exp
        run = self.data_source.run 
    
    instrument = exp[0:3]

    if not file_name:
        if not path:
            exp_dir = "/reg/d/psdm/{:}/{:}".format(instrument, exp)
            if h5folder is 'results' and not os.path.isdir(os.path.join(exp_dir, h5folder)): 
                h5folder0 = h5folder
                h5folder = 'res'

            if not os.path.isdir(os.path.join(exp_dir, h5folder)): 
                raise Exception(os.path.join(expdir, h5folder0)+' does not exist!')

            path = os.path.join(exp_dir,'results',subfolder)
        
        if not os.path.isdir(path):
            os.mkdir(path)
    
        file_base = 'run{:04}'.format(int(run))
        file_name = os.path.join(path,'{:}_{:}.nc'.format(file_base,'config'))

    print file_name
    if not glob.glob(file_name):
        if no_create:
            return None
        else:
            reload = True

    if reload:
        if not ds:
            import PyDataSource
            ds = PyDataSource.DataSource(exp=exp, run=run)
        write_hdf5(ds, file_base=file_base, path=path, no_events=True)
        print 'Need to create'
    
    x= process_one_file(file_name)
    if summary:
        x = to_summary(x)

    return x


# Need to add in 'chunking based on steps'
def write_hdf5(self, nevents=None, max_size=10001, 
        path='', file_base=None, 
        h5folder='scratch', subfolder='nc',
        publish=False,
        store_data=[],
        chunk_steps=False,
        ichunk=None,
        nchunks=24,
        #code_flags={'XrayOff': [162], 'XrayOn': [-162], 'LaserOn': [183, -162], 'LaserOff': [184, -162]},
        code_flags={'XrayOff': [162], 'XrayOn': [-162]},
        drop_unused_codes=True,
        pvs=[], epics_attrs=[], 
        eventCodes=None,  
        save=None, 
        mpio=False, 
        no_events=False,
        **kwargs):
    """
    DEVELOPMENT:  Write directly to hdf5 with h5netcdf package.  
    based on to_xarray method.
       
    Parameters
    ----------
    max_size : uint
        Maximum array size of data objects to build into xarray.
    ichunk: int
        chunk index (skip ahead nevents*ichunk)
    pvs: list
        List of pvs to be loaded vs time
    epics_attrs: list
        List of epics pvs to be saved as run attributes based on inital value 
        of first event.
    code_flags : dict
        Dictionary of event code flags. 
        Default = {'XrayOff': [162], 'XrayOn': [-162]}
    eventCodes : list
        List of event codes 
        Default is all event codes in DataSource
    drop_unused_codes : bool
        If true drop unused eventCodes [default=True]

    Example
    -------
    import PyDataSource
    ds = PyDataSource.DataSource(exp='xpptut15',run=200)
    evt = ds.events.next()
    evt.opal_1.add.projection('raw', axis='x', roi=((0,300),(400,1024)))
    evt.cs140_rob.add.roi('calib',sensor=1,roi=((104,184),(255,335)))
    evt.cs140_rob.add.count('roi')

    """
    time0 = time.time()
    try:
        import xarray as xr
        import h5netcdf
        from mpi4py import MPI
    except:
        raise Exception('xarray package not available. Use for example conda environment with "source conda_setup"')

           
    rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
    if not no_events:
        ichunk=rank
    
    self.reload()
    evt = self.events.next(publish=publish, init=publish)
    dtime = evt.EventId
    if not eventCodes:
        eventCodes = sorted(self.configData._eventcodes.keys())
    
    if hasattr(self.configData, 'ScanData') and self.configData.ScanData:
        nsteps = self.configData.ScanData.nsteps
    else:
        nsteps = 1
    
    ievent0 = 0
    if not nevents:
        nevents_total = self.nevents
        if ichunk is not None:
            if chunk_steps and nsteps > 1:
                istep = ichunk-1
                ievent_start = self.configData.ScanData._scanData['ievent_start'][istep]
                ievent_end = self.configData.ScanData._scanData['ievent_end'][istep]
                nevents = ievent_end-ievent_start+1
                ievent0 = ievent_start
            else:
                nevents = int(np.ceil(self.nevents/float(nchunks)))
                ievent0 = (ichunk-1)*nevents
            
            print 'Do {:} of {:} events for {:} chunk'.format(nevents, self.nevents, ichunk)
        else:
            nevents = nevents_total
    
    else:
        nevents_total = nevents
        if ichunk is not None:
            nevents = int(np.ceil(nevents_total/float(nchunks)))
            ievent0 = (ichunk)*nevents

    run = int(self.data_source.run)
    if not path:
        if no_events:
            path = '/reg/d/psdm/{:}/{:}/{:}/{:}/'.format(self.data_source.instrument, 
                self.data_source.exp,h5folder,subfolder)
        else:
            path = '/reg/d/psdm/{:}/{:}/{:}/{:}/Run{:04}/'.format(self.data_source.instrument, 
                self.data_source.exp,h5folder,subfolder,run)

    if not os.path.isdir(path):
        os.mkdir(path)

    if not file_base:
        #if ichunk is None:
        if True:
            file_base = 'run{:04}'.format(int(run))
        else:
            file_base = 'run{:04}_c{:03}'.format(run, ichunk)

    adat = {}
    axdat = {}
    atimes = {}
    btimes = []
    xcoords = {}
    axcoords = {}

    if no_events:
        file_name = os.path.join(path,'{:}_{:}.nc'.format(file_base,'config'))
        xbase = h5netcdf.File(file_name, 'w')
        ntime = nevents_total
    elif mpio: 
        file_name = os.path.join(path,'{:}_{:}.nc'.format(file_base,'base'))
        xbase = h5netcdf.File(file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        ntime = nevents_total
    else:
        file_name = os.path.join(path,'{:}_C{:02}_{:}.nc'.format(file_base,ichunk,'base'))
        xbase = h5netcdf.File(file_name, 'w')
        ntime = nevents

    xbase.dimensions['time'] = ntime
    #xbase = h5netcdf.File(file_name, 'w')

    neventCodes = len(eventCodes)
    det_funcs = {}
    epics_pvs = {}
    for pv in pvs:
        epics_pvs[pv] = {} 

    if True:
        # Experiment Attributes
        xbase.attrs['data_source'] = str(self.data_source)
        xbase.attrs['run'] = self.data_source.run
        for attr in ['instrument', 'experiment', 'expNum', 'calibDir']:
            xbase.attrs[attr] = getattr(self, attr)

        
        ttypes = {'sec': 'int32', 
                  'nsec': 'int32', 
                  'fiducials': 'int32', 
                  'ticks': 'int32', 
                  'run': 'int32'}
        
        # explicitly order EventId coords in desired order 
        if ichunk == 0:
            print 'Begin processing {:} events'.format(nevents)
     
        cattrs =  ['sec', 'nsec', 'fiducials', 'ticks', 'run', 'step']

        for attr in cattrs:
            xcoords[attr] = xbase.create_variable(attr, ('time',), int)

        coordinates = ' '.join(cattrs)

        # Event Codes -- earlier bool was not supported but now is. 
        #if not no_events:
        if True:
            for code in eventCodes:
                xbase.create_variable('ec{:}'.format(code), ('time',), bool)
                coordinates += ' ec{:}'.format(code)

            for attr, ec in code_flags.items():
                xbase.create_variable(attr, ('time',), bool)
                xbase[attr].attrs['doc'] = 'Event code flag: True if all positive and no negative "codes" are in eventCodes'
                xbase[attr].attrs['codes'] = ec

        xbase.attrs['event_flags'] = code_flags.keys()

        #xbase.create_variable('steps', ('isteps'), data=range(nsteps))
        #xbase.create_variable('codes', ('icodes'), data=eventCodes)
     
        # Scan Attributes -- cxbase.create_variable('codes', data=eventCodes)annot put None or dicts as attrs in netcdf4
        # e.g., pvAliases is a dict
        if hasattr(self.configData, 'ScanData') and self.configData.ScanData:
            if self.configData.ScanData.nsteps == 1:
                attrs = ['nsteps']
            else:
                attrs = ['nsteps', 'pvControls', 'pvMonitors', 'pvLabels']
                #xbase.coords['pvControls'] = self.configData.ScanData.pvControls
                #xbase.create_variable('pvControls'] = self.configData.ScanData.pvControls
                for attr, vals in self.configData.ScanData.control_values.items():
                    alias = self.configData.ScanData.pvAliases[attr]
                    xbase.create_variable(alias+'_steps', ('steps',), data=vals) 
                    xbase.create_variable(alias, ('time',), vals.dtype)
                    coordinates += ' '+alias+'+steps'
                    coordinates += ' '+alias

            for attr in attrs:
                val = getattr(self.configData.ScanData, attr)
                if val:
                    xbase.attrs[attr] = val 
 
    base_coordinates = coordinates
    
    for srcstr, src_info in self.configData._sources.items():
        det = src_info['alias']
        if ichunk == 0:
            print 'configuring', det
        nmaxevents = 100
        ievt = 0
        if not no_events:
            if mpio:
                file_name = os.path.join(path,'{:}_{:}.nc'.format(file_base,det))
                axdat[det] = h5netcdf.File(file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
            else:
                file_name = os.path.join(path,'{:}_C{:02}_{:}.nc'.format(file_base,ichunk,det))
                axdat[det] = h5netcdf.File(file_name, 'w')
            
            axdat[det].dimensions['time'] = ntime
            
        try:
            while det not in evt._attrs and ievt < nmaxevents:
                evt.next(publish=publish, init=publish)
                ievt += 1
            
            detector = getattr(evt,det)
            if hasattr(detector, '_update_xarray_info'):
                detector._update_xarray_info()
            
            config_info = detector._xarray_info.get('attrs')
#            if attrs:
#                axdat[det].coords[det+'_config'] = ([det+'_steps'], range(nsteps))
#                axdat[det].coords[det+'_config'].attrs.update(attrs)

            adat[det] = {}
            axcoords[det] = {}
            det_funcs[det] = {}
            xarray_dims = detector._xarray_info.get('dims')
            funcs = []
            # Add default attr information.
            src_info.update(**detector.sourceData._source)
            if 'src' in src_info:
                src_info['src'] = str(src_info['src'])
            
            if not no_events:
                axdat[det].attrs.update(**config_info)
                axdat[det].attrs.update(**src_info)
                axdat[det].attrs['funcs'] = funcs
            
            attr = 'present'
            alias = det+'_'+attr
            xbase.create_variable(alias, ('time',), bool)
            xbase[alias].attrs.update(**config_info)
            xbase[alias].attrs.update(**src_info)
            xbase[alias].attrs['funcs'] = funcs
            if not no_events:
                axdat[det].create_variable(alias, ('time',), bool)
                axdat[det][alias].attrs.update(**config_info)
                axdat[det][alias].attrs.update(**src_info)
                axdat[det][alias].attrs['funcs'] = funcs
            
            if xarray_dims is not None: 
                for attr,item in sorted(xarray_dims.items(), key=operator.itemgetter(0)):
                    # Only save data with less than max_size total elements
                    alias = det+'_'+attr
                    attr_info = src_info.copy()
                    if len(item) == 3:
                        attr_info.update(**item[2])

                    attr_info['attr'] = attr
                    if detector._tabclass == 'evtData':
                        if detector.evtData is not None:
                            infos = detector.evtData._attr_info.get(attr)
                            if infos:
                                attr_info.update({a: infos[a] for a in ['doc', 'unit']})
                        else:
                            print 'No data for {:} in {:}'.format(str(detector), attr)
                    
                    else:
                        if detector._calib_class is not None:
                            infos = detector.calibData._attr_info.get(attr)
                            if infos is not None:
                                attr_info.update(infos)
                        
                        elif detector.detector is not None:
                            infos = detector.detector._attr_info.get(attr)
                            if infos is not None:
                                attr_info.update(infos)
                    
                    # Make sure no None attrs
                    for a, aitm in attr_info.items():
                        if aitm is None:
                            attr_info.update({a, ''})
                        if hasattr(aitm, 'dtype') and aitm.dtype is np.dtype('O'):
                            attr_info.pop(a)

                    det_funcs[det][attr] = {'alias': alias, 'det': det, 'attr': attr, 'attr_info': attr_info}
                    if np.product(item[1]) <= max_size or alias in store_data:
                        a = [det+'_'+name for name in item[0]]
                        a.insert(0, 'time')
                        try:
                            b = list(item[1])
                            #maxsize = list(item[1])
                        except:
                            b = [item[1]]
                            #maxsize = [item[1]]
                        #maxsize.insert(0, None) 
                        b.insert(0, ntime)
                        if not no_events:
                            for xname, xshape in zip(a, b):
                                if xname not in axdat[det].dimensions:
                                    axdat[det].dimensions[xname] = xshape

                            adat[det][alias] = axdat[det].create_variable(alias, a, float)
                            funcs.append(attr)
                            try:
                                axdat[det][alias].attrs.update(**attr_info)
                            except:
                                print 'Cannot add attrs', attr_info
                            det_funcs[det][attr]['event'] = {'dims': a, 'shape': b}
            
            coordinates = ' '.join(['sec', 'nsec', 'fiducials', 'ticks', 'run', 'step'])
            if not no_events:
                for attr in ['sec', 'nsec', 'fiducials', 'ticks', 'run', 'step']:
                    axdat[det].create_variable(attr, ('time',), int)
        
            coords = detector._xarray_info.get('coords')
            if coords:
                for coord, item in sorted(coords.items(), key=operator.itemgetter(0)):
                    alias = det+'_'+coord
                    attr_info = detector.calibData._attr_info.get(coord)
                    try:
                        if isinstance(item, tuple):
                            dims = {det+'_'+dim for dim in item[0]}
                            vals = item[1]
                            xbase.create_variable(alias, dims, data=vals)
                            if not no_events:
                                axcoords[det][alias] = axdat[det].create_variable(alias, dims, data=vals)
                        else:
                            xbase.create_variable(alias, (alias,), data=item)
                            if not no_events:
                                axcoords[det][alias] = axdat[det].create_variable(alias, (alias,), data=item)
                        if attr_info:
                            xbase[alias].attrs.update(**attr_info)
                            if not no_events:
                                axdat[det][alias].attrs.update(**attr_info)
                        coordinates += ' '+alias
                    except:
                        if ichunk == 0:
                            print 'Missing coord', det, coord, item
            
            base_coordinates += coordinates
            if not no_events:
                axdat[det].attrs['coordinates'] = coordinates

        except:
            print 'ERROR loading', srcstr, det
            traceback.print_exc()
# Need to fix handling of detector image axis

    xbase.attrs['coordinates'] = base_coordinates
                  
    if ichunk == 0:
        print 'Dataset configured'
    #return xbase, axdat 

    self.reload()
    if True or not no_events:
        time0 = time.time()
        igood = -1
        aievt = {}
        aievents = {}

        # keep track of events for each det
        for srcstr, srcitem in self.configData._sources.items():
            det = srcitem.get('alias')
            aievt[det] = -1
            aievents[det] = []
      
        if ichunk is not None:
            #print 'Making chunk {:}'.format(ichunk)
            #print 'Starting with event {:} of {:}'.format(ievent0,self.nevents)
            #print 'Analyzing {:} events'.format(nevents)
            xbase.attrs['ichunk'] = ichunk
            # Need to update to jump to event.
            if ichunk > 1 and not chunk_steps:
                for i in range(ievent0):
                    evt = self.events.next()
            
                #print 'Previous event before current chunk:', evt

        if ichunk is not None:
            evtformat = '{:10.1f} sec, Event {:} of {:} in chunk {:} with {:} accepted'
        else:
            evtformat = '{:10.1f} sec, Event {:} of {:} with {:} accepted'
        
        #for ievent in range(ds.nevents+1):
        for ievt in range(nevents):
            ievent = ievent0+ievt
            if ievt > 0 and (ievt % 100) == 0:
                if ichunk is not None:
                    print evtformat.format(time.time()-time0, ievt, nevents, ichunk, igood+1)
                else:
                    print evtformat.format(time.time()-time0, ievt, nevents, igood+1)
            
            if ichunk > 0 and chunk_steps:
                if ievt == 0:
                    # on first event skip to the desired step
                    for i in range(ichunk):
                        step_events = self.steps.next()
                
                try:
                    evt = step_events.next()
                except:
                    ievent = -1
                    continue

            elif ievent < self.nevents:
                try:
                    evt = self.events.next(publish=publish, init=publish)
                except:
                    ievent = -1
                    continue
            else:
                ievent = -1
                continue

            if len(set(eventCodes) & set(evt.Evr.eventCodes)) == 0:
                continue
           
            dtime = evt.EventId
            if dtime is None:
                continue
            
            if igood+1 == nevents:
                break
            
            igood += 1
            iwrite = igood
            if mpio:
                iwrite += ievent0

            istep = self._istep
            xbase['step'][iwrite] = istep
            btimes.append(dtime)
            
            for attr in ['sec', 'nsec', 'fiducials', 'ticks']:
                xbase[attr][iwrite] = getattr(dtime, attr)
            
            for ec in evt.Evr.eventCodes:
                if ec in eventCodes:
                    xbase['ec{:}'.format(ec)][iwrite] = True

            for attr, codes in code_flags.items():
                if evt.Evr.present(codes):
                    xbase[attr][iwrite] = True

#            for pv, pvarray in epics_pvs.items():
#                try:
#                    val = float(self.epicsData.getPV(pv).data()) 
#                    pvarray.update({dtime: val})
#                except:
#                    print 'cannot update pv', pv, dtime

            for det in evt._attrs:
                xbase[det+'_present'][iwrite] = True
            
            if not no_events:
                for det in evt._attrs:
                    detector = evt._dets.get(det)
                    aievt[det] += 1 
                    iwrite = aievt[det]
                    if mpio:
                        iwrite += ievent0
                    aievents[det].append(ievent)
                    axdat[det][det+'_present'][iwrite] = True
                    for attr in ['sec', 'nsec', 'fiducials', 'ticks']:
                        axdat[det][attr][iwrite] = getattr(dtime, attr)
                    
                    for attr in  axdat[det].attrs['funcs']:
                        vals = getattr(detector, attr)
                        alias = det+'_'+attr
                        if vals is not None:
                            try:
                                axdat[det][alias][iwrite] = vals
                            except:
                                print 'Event Error', alias, det, attr, ievent, vals
                                vals = None
        
        xbase.attrs['nevents'] = igood+1
        for det in axdat:
            axdat[det].attrs['nevents'] = aievt[det]+1
            axdat[det].close()

    elif False:
        try:
            for iwrite, atup in enumerate(self._idx_times_tuple):
                for iattr, attr in enumerate(['sec', 'nsec', 'fiducials']):
                    xbase[attr][iwrite] = atup[iattr]
        except:
            traceback.print_exc()
            return self, xbase 
        
        if True:
            time0 = time.time()
            for ievent, evt in enumerate(self.events):
                if ievent % 100 == 0:
                    print ievent, evt

            print 'total time', time.time()-time0

    xbase.close()

    return path, file_base

#        xbase = xbase.isel(time=range(len(btimes)))
#        xbase['time'] =  [e.datetime64 for e in btimes]
#        for attr, dtyp in ttypes.items():
#            xbase.coords[attr] = (['time'], np.array([getattr(e, attr) for e in btimes],dtype=dtyp))
#            
#        # fill each control PV with current step value
#        scan_variables = []
#        if self.configData.ScanData and self.configData.ScanData.nsteps > 1:
#            for attr, vals in self.configData.ScanData.control_values.items():
#                alias = self.configData.ScanData.pvAliases[attr]
#                scan_variables.append(alias)
#                xbase.coords[alias] = (['time'], xbase.coords[alias+'_steps'][xbase.step]) 
#
#        xbase.attrs['scan_variables'] = scan_variables
#        xbase.attrs['correlation_variables'] = []
#       
#        # add in epics_attrs (assumed fixed over run)
#        for pv in epics_attrs:
#            try:
#                xbase.attrs.update({pv: self.epicsData.getPV(pv).data()[0]})
#            except:
#                print 'cannot att epics_attr', pv
#                traceback.print_exc()

#        if drop_unused_codes:
#            for ec in eventCodes:
#                print 'Dropping unused eventCode', ec
#                if not xbase['ec{:}'.format(ec)].any():
#                    xbase = xbase.drop('ec{:}'.format(ec))
#
#        # cut down size of xdat
#        det_list = [det for det in axdat]
#        for det in np.sort(det_list):
#            nevents = len(atimes[det])
#            if nevents > 0 and det in axdat:
#                try:
#                    print 'merging', det, nevents
#                    xdat = axdat.pop(det)
#                    if 'time' in xdat.dims:
#                        xdat = xdat.isel(time=range(nevents))
#                        xdat['time'] = [e.datetime64 for e in atimes[det]]
#                        xdat = xdat.reindex_like(xbase)
#            
#                    xbase = xbase.merge(xdat)
#                
#                except:
#                    print 'Could not merge', det
#                    return xbase, xdat, axdat, atimes, btimes
#
#        attrs = [attr for attr,item in xbase.data_vars.items()] 
#        for attr in attrs:
#            for a in ['unit', 'doc']:
#                if a in xbase[attr].attrs and xbase[attr].attrs[a] is None:
#                    xbase[attr].attrs[a] = ''
#        
#        for pv, pvdata in epics_pvs.items():
#            xdat = xr.Dataset({pv: (['time'], np.array(pvdata.values()).squeeze())}, 
#                                  coords={'time': [e.datetime64 for e in pvdata.keys()]} )
#            xbase = xbase.merge(xdat)
#
#        xbase = resort(xbase)
#
#        if save:
#            try:
#                to_h5netcdf(xbase)
#            except:
#                print 'Could not save to_h5netcdf'
#
#        return xbase


def to_summary(x, dim='time', groupby='step', 
        save_summary=False,
        normby=None,
        omit_list=['run', 'sec', 'nsec', 'fiducials', 'ticks'],
        stats=['mean', 'std', 'min', 'max', 'count'],
        **kwargs):
    """
    Summarize a run.
    
    Parameters
    ---------
    x : xarray.Dataset
        input xarray Dataset
    dim : str
        dimension to summarize over [default = 'time']
    groupby : str
        coordinate to groupby [default = 'step']
    save_summary : bool
        Save resulting xarray Dataset object
    normby : list or dict
        Normalize data by attributes
    omit_list : list
        List of Dataset attributes to omit
    stats : list
        List of statistical operations to be performed for summary.
        Default = ['mean', 'std', 'min', 'max', 'count']
    """
    import xarray as xr
    xattrs = x.attrs
    data_attrs = {attr: x[attr].attrs for attr in x}
    if 'Damage_cut' in x:
        x = x.where(x.Damage_cut).dropna(dim)
   
    coords = [c for c in x.coords if c != dim and c not in omit_list and dim in x.coords[c].dims] 
    x = x.reset_coords(coords)
    if isinstance(normby, dict):
        for norm_attr, attrs in normby.items():
            x = normalize_data(x, attrs, norm_attr=norm_attr)
    elif isinstance(normby, list):
        x = normalize_data(x, normby)

    # With new xarray 0.9.1 need to make sure loaded otherwise h5py error
    x.load()
    xgroups = {}
    if groupby:
        group_vars = [attr for attr in x.keys() if groupby+'s' in x[attr].dims]
        if group_vars:
            xgroups = {attr: x[attr].rename({groupby+'s': groupby}) for attr in group_vars}
            for attr in group_vars:
                del x[attr]
        x = x.groupby(groupby)

    dsets = [getattr(x, func)(dim=dim) for func in stats]
    x = xr.concat(dsets, stats).rename({'concat_dim': 'stat'})
    for attr,val in xattrs.items():
        x.attrs[attr] = val
    for attr,item in data_attrs.items():
        if attr in x:
            x[attr].attrs.update(item)

    for c in coords:                                                       
        x = x.set_coords(c)
    
    for attr in group_vars:
        x[attr] = xgroups[attr]
    x = resort(x)
    if save_summary:
        to_h5netcdf(x)
    return x

def add_steps(x, attr, name=None):
    vals = getattr(x, attr).values
    steps = np.sort(list(set(vals)))
    asteps = np.digitize(vals, steps)
    if not name:
        name = attr+'_step'
 
    x.coords[name] = (['time'], asteps)


def add_index(x, attr, name=None, nbins=8, bins=None, percentiles=None):

    if not bins:
        if not percentiles:
            percentiles = (arange(nbins+1))/float(nbins)*100.

        bins = np.percentile(x[attr].to_pandas().dropna(), percentiles)

    if not name:
        name = attr+'_index'

    #per = [percentiles[i-1] for i in np.digitize(x[attr].values, bins)]
    x[name] = (['time'], np.digitize(x[attr].values, bins))


def normalize_data(x, variables=[], norm_attr='PulseEnergy', name='norm', quiet=True):
    """
    Normalize a list of variables with norm_attr [default = 'PulseEnergy']
    """
    if not variables:
        variables = [a for a in get_correlations(x) if not a.endswith('_'+name)]    

    for attr in variables:
        aname = attr+'_'+name
        try:
            x[aname] = x[attr]/x[norm_attr]
            x[aname].attrs = x[attr].attrs
            try:
                x[aname].attrs['doc'] = x[aname].attrs.get('doc','')+' -- normalized to '+norm_attr
                units = x[attr].attrs.get('unit')
                norm_units = x[norm_attr].attrs.get('unit')
                if units and norm_units:
                    x[aname].attrs['unit'] = '/'.join([units, norm_units])
            except:
                if not quiet:
                    print 'cannot add attrs for', aname
        except:
            print 'Cannot normalize {:} with {:}'.format(attr, norm_attr)

    return  resort(x)

def resort(x):
    """
    Resort alphabitically xarray Dataset
    """
    coords = sorted([c for c in x.coords.keys() if c not in x.coords.dims])
    x = x.reset_coords()
    x = x[sorted(x.data_vars)]

    for c in coords:                                                       
        x = x.set_coords(c)

    return x

def map_indexes(xx, yy, ww):                                                                      
    """
    Simplified map method from PSCalib.GeometryAccess.img_from_pixel_arrays
    
    Parameters
    ----------
    xx : array-like
        Array of x coordinates
    yy : array-like
        Array of y coordinates
    ww : array-like
        Array of weights

    Returns
    -------
    2D image array

    """
    a = np.zeros([xx.max()+1,yy.max()+1])
    a[xx,yy] = ww
    return a

def xy_ploterr(a, attr=None, xaxis=None, title='', desc=None, fmt='o', **kwargs):
    """Plot summary data with error bars, e.g.,
        xy_ploterr(x, 'MnScatter','Sample_z',logy=True)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if not attr:
        print 'Must supply plot attribute'
        return

    if 'groupby' in kwargs:
        groupby=kwargs['groupby']
    elif 'step' in a.dims:
        groupby='step'
    else:
        groupby='run'

    run = a.attrs.get('run')
    experiment = a.attrs.get('experiment', '')
    runstr = '{:} Run {:}'.format(experiment, run)
    name = a.attrs.get('name', runstr)
    if not title:
        title = '{:}: {:}'.format(name, attr)

    if not xaxis:
        xaxis = a.attrs.get('scan_variables')
        if xaxis:
            xaxis = xaxis[0]

    ylabel = kwargs.get('ylabel', '')
    if not ylabel:
        ylabel = a[attr].name
        unit = a[attr].attrs.get('unit')
        if unit:
            ylabel = '{:} [{:}]'.format(ylabel, unit)

    xlabel = kwargs.get('xlabel', '')
    if not xlabel:
        xlabel = a[xaxis].name
        unit = a[xaxis].attrs.get('unit')
        if unit:
            xlabel = '{:} [{:}]'.format(xlabel, unit)
    
    if xaxis:
        if 'stat' in a[xaxis].dims:
            xerr = a[xaxis].sel(stat='std').values
            a[xaxis+'_axis'] = ([groupby], a[xaxis].sel(stat='mean').values)
            xaxis = xaxis+'_axis'
        else:
            xerr = None

        a = a.swap_dims({groupby:xaxis})
    
    else:
        xerr = None

    if desc is None:
        desc = a[attr].attrs.get('doc', '')

    ndims = len(a[attr].dims)
    if ndims == 2:
        c = a[attr].to_pandas().T
        if xerr is not None:
            c['xerr'] = xerr
        c = c.sort_index()
        
        plt.figure()
        plt.gca().set_position((.1,.2,.8,.7))
        p = c['mean'].plot(yerr=c['std'],xerr=c.get('xerr'), title=title, fmt=fmt, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if desc:
            plt.text(-.1,-.2, desc, transform=p.transAxes, wrap=True)   
 
        return p 
    elif ndims == 3:
        plt.figure()
        plt.gca().set_position((.1,.2,.8,.7))
        pdim = [d for d in a[attr].dims if d not in ['stat', groupby, xaxis]][0]
        for i in range(len(a[attr].coords[pdim])):
            c = a[attr].sel(**{pdim:i}).drop(pdim).to_pandas().T.sort_index()
            p = c['mean'].plot(yerr=c['std'], fmt=fmt, **kwargs)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        p.set_title(title)
        if desc:
            plt.text(-.1,-.2, desc, transform=p.transAxes, wrap=True)   

        return p 
    else:
        print 'Too many dims to plot'

def make_image(self, pixel=.11, ix0=None, iy0=None):
    """Return image from 3-dim detector DataArray."""
    import numpy as np
    base = self.name.split('_')[0]
    xx = self[base+'_indexes_x']
    yy = self[base+'_indexes_y']
    a = np.zeros([xx.max()+1,yy.max()+1])

    x = self.coords.get(base+'_ximage')
    if not x:
        if not ix0:
            ix0 = a.shape[1]/2.
        x = (np.arange(a.shape[1])-ix0)*pixel
    y = self.coords.get(base+'_yimage')
    if not y:
        if not iy0:
            iy0 = a.shape[0]/2.
        y = (np.arange(a.shape[0])-iy0)*pixel
    a[xx,yy] = self.data
    if x is not None and y is not None:
        try:
            return xr.DataArray(a, coords=[(base+'_yimage', y), (base+'_ximage', x)],
                                   attrs=self.attrs)
        except:
            pass

    return xr.DataArray(a)
    
    #return xr.DataArray(a, coords=[(base+'_yimage', y.mean(axis=1)), (base+'_ximage', x.mean(axis=0))])


