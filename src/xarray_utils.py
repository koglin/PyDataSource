def to_summary(x, dim='time', groupby='step', 
        save_summary=False,
        normby=None,
        omit_list=['run', 'sec', 'nsec', 'fiducials', 'ticks'],
        #stats=['mean', 'std', 'min', 'max',],
        stats=['mean', 'std', 'var', 'min', 'max', 'count'],
        cut='Damage_cut',
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
        Default = ['mean', 'std', 'var', 'min', 'max', 'count']
    """
    import xarray as xr
    xattrs = x.attrs
    data_attrs = {attr: x[attr].attrs for attr in x}
    if cut in x:
        x = x.where(x[cut]).dropna(dim)
   
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
        sattrs = [a for a in x if 'stat' in x[a].dims or groupby+'s' in x[a].dims]
        if sattrs:
            xstat = x[sattrs].rename({groupby+'s': groupby})
            x = x.drop(sattrs)
        x = x.groupby(groupby)
#        group_vars = [attr for attr in x.keys() if groupby+'s' in x[attr].dims]
#        if group_vars:
#            xgroups = {attr: x[attr].rename({groupby+'s': groupby}) for attr in group_vars}
#            for attr in group_vars:
#                del x[attr]
        #x = x.groupby(groupby)

    dsets = [getattr(x, func)(dim=dim) for func in stats]
    x = xr.concat(dsets, stats).rename({'concat_dim': 'stat'})
#    try:
#        x = xr.concat([dsets, stats], dim='stat')
#    except:
#        return dsets
    
    for attr,val in xattrs.items():
        x.attrs[attr] = val
    for attr,item in data_attrs.items():
        if attr in x:
            x[attr].attrs.update(item)

    for c in coords:                                                       
        x = x.set_coords(c)

    if groupby and sattrs:
        x = x.merge(xstat)
#    try:
#        for attr in group_vars:
#            x[attr] = xgroups[attr]
#    except:
#        return x, xgroups

    x = resort(x)
    if save_summary:
        to_h5netcdf(x)
    return x

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

def add_index(x, attr, name=None, nbins=8, bins=None, percentiles=None):
    import numpy as np
    if not bins:
        if not percentiles:
            percentiles = (np.arange(nbins+1))/float(nbins)*100.

        bins = np.percentile(x[attr].to_pandas().dropna(), percentiles)

    if not name:
        name = attr+'_index'

    #per = [percentiles[i-1] for i in np.digitize(x[attr].values, bins)]
    x[name] = (['time'], np.digitize(x[attr].values, bins))

def add_steps(x, attr, name=None):
    vals = getattr(x, attr).values
    steps = np.sort(list(set(vals)))
    asteps = np.digitize(vals, steps)
    if not name:
        name = attr+'_step'
 
    x.coords[name] = (['time'], asteps)

def get_correlations(y, attr, confidence=0.33, method='pearson',
        omit_list=['sec', 'nsec', 'fiducials', 'ticks', 'Damage_cut', 'EBeam_damageMask']):
    """Find variables that correlate with given attr.
    """
    x = y.reset_coords()
    attrs = [a for a, item in x.data_vars.items() if item.dims == ('time',)]
    cmatrix = x[attrs].to_dataframe().corr(method=method)
    cattrs = {a: item for a, item in cmatrix[attr].iteritems() if item > confidence \
            and a != attr and a not in omit_list}    
    return cattrs

def get_cov(y, attr, attrs=[], confidence=0.33,
        omit_list=['sec', 'nsec', 'fiducials', 'ticks', 'Damage_cut', 'EBeam_damageMask']):
    """Find variables that correlate with given attr.
    """
    x = y.reset_coords()    
    if not attrs:
        #attrs = [a for a, item in x.data_vars.items() if item.dims == ('time',)]
        attrs = [a for a in get_correlations(x, attr, confidence=confidence).keys() if a not in omit_list \
                and not a.startswith('FEEGasDetEnergy') and not a.startswith('Gasdet')]
        attrs.append(attr)

    df = x[attrs].to_dataframe()
    cmatrix = df.cov()
    #cattrs = {a: item for a, item in cmatrix[attr].iteritems() if a != attr and a not in omit_list}    
    return cmatrix

def heatmap(df, attrs=[], method='pearson', confidence=0.33, position=(0.3,0.35,0.45,0.6), show=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig = plt.figure() 
    ax1 = fig.add_subplot(111)
    #plt.gca().set_position(position)
    if attrs:
        df = df[attrs]
    
    corr = df.corr(method=method)
    if confidence:
        cattrs = df.keys()[((abs(corr)>=confidence).sum()>1).values]
        corr = df[cattrs].corr(method=method)
    
    sns.heatmap(corr,annot=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    plt.gca().set_position(position)
    if show:
        plt.show()   

def xy_ploterr(a, attr=None, xaxis=None, title='', desc=None, 
        fmt='o', position=(.1,.2,.8,.7), **kwargs):
    """Plot summary data with error bars, e.g.,
        xy_ploterr(x, 'MnScatter','Sample_z',logy=True)
    """
    import matplotlib.pyplot as plt
    if not attr:
        print 'Must supply plot attribute'
        return

    if 'groupby' in kwargs:
        groupby=kwargs['groupby']
    elif 'step' in a.dims:
        groupby='step'
    elif 'steps' in a.dims:
        groupby='steps'
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

    if xaxis:
        if 'stat' in a[xaxis].dims:
            xerr = a[xaxis].sel(stat='std').values
            a[xaxis+'_axis'] = ([groupby], a[xaxis].sel(stat='mean').values)
            xaxis = xaxis+'_axis'
        else:
            xerr = None

        a = a.swap_dims({groupby:xaxis})
    
    else:
        xaxis = groupby
        xerr = None

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
    
    if desc is None:
        desc = a[attr].attrs.get('doc', '')

    ndims = len(a[attr].dims)
    if ndims == 2:
        c = a[attr].to_pandas().T
        if xerr is not None:
            c['xerr'] = xerr
        c = c.sort_index()
        
        plt.figure()
        plt.gca().set_position(position)
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
            c = a[attr].sel(**{pdim:i})
            if pdim in c:
                c = c.drop(pdim)
            c = c.to_pandas().T.sort_index()
            p = c['mean'].plot(yerr=c['std'], fmt=fmt, **kwargs)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        p.set_title(title)
        if desc:
            plt.text(-.1,-.2, desc, transform=p.transAxes, wrap=True)   

        return p 
    else:
        print 'Too many dims to plot'





