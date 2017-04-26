def to_summary(x, dim='time', groupby='step', 
        save_summary=False,
        normby=None,
        omit_list=['run', 'sec', 'nsec', 'fiducials', 'ticks'],
        #stats=['mean', 'std', 'min', 'max',],
        stats=['mean', 'std', 'var', 'min', 'max', 'count'],
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
        sattrs = [a for a in x if 'stat' in x[a].dims or groupby+'s' in x[a].dims]
        xstat = x[sattrs].rename({groupby+'s': groupby})
        x = x.drop(sattrs).groupby(groupby)
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

    if groupby:
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



