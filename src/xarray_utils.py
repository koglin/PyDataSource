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

    return corr

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

def ttest_groupby(xo, attr, groupby='ec162', ishot=0, nearest=None, verbose=False):
    """
    nearest neighbors compare with
    """
    from scipy import stats
    try:
        # Select DataArray for attr with groupby as coord
        da = xo.reset_coords()[[attr,groupby]].set_coords(groupby)[attr].load().dropna(dim='time')
    except:
        return None
    ntime = len(da.time)
    g = da.groupby(groupby).groups
    if len(g) > 1:
        if nearest is not None:
            k0 = []
            for inear in range(-nearest,0)+range(1,nearest+1):
                for itest in g[1]:
                    a = itest+inear
                    if a>0 and a<ntime:
                        k0.append(a)
        else:
            k0 = [a+ishot for a in g[0] if (a+ishot)>0 and (a+ishot)<ntime]
        k1 = [a+ishot for a in g[1] if (a+ishot)>0 and (a+ishot)<ntime]
        df1 = da[k1].to_pandas().dropna()
        df0 = da[k0].to_pandas().dropna()
        ttest = stats.ttest_ind(df1,df0)
        return ttest
    else:
        if verbose:
            print '{:} has only one group -- cannot compare'.format(attr)
        return None

def test_correlation(x, attr0, attr1='Gasdet_post_atten', cut=None, shift=None, dim='time'):
    from scipy import stats
    import numpy as np
    xds = x[[attr0,attr1]]
    if cut and cut in xds:
        xds = xds.where(x[cut]).where(x[cut].shift(**{dim:-shift})).dropna(dim) 

    xds = xds.reset_coords()
    
    df0 = xds[attr0].to_pandas()
    if shift:
        df0 = df0.shift(-shift)
    df1 = xds[attr1].to_pandas()
    kind = np.isfinite(df1) & np.isfinite(df0)
    result = stats.pearsonr(df0[kind],df1[kind])
    return result

def set_delta_beam(x, code='ec162', attr='delta_drop'):
    """Find the number of beam codes to nearest code
       using time stamps
    """
    import traceback#
    import pandas as pd
    import numpy as np
    if code not in x:
        return None
    else:
        try:
            df0 = x.reset_coords()[code].to_pandas()
            if df0.any():
                df_beam = x.reset_coords().fiducials.to_pandas()/3
                df_drop = df_beam[df0]
                x.coords[attr] = df_beam
                i = 0
                for a, b in df_beam.iteritems():
                    x.coords[attr][i] = b-df_drop.values[np.argmin(abs(df_drop.index-a))]
                    i += 1
                # Not robust way to get rid of outliers 
                #dbeam_max = df_beam.diff().max()
                #x.coords[attr][abs(x.coords[attr]) > dbeam_max] = dbeam_max
                x.coords[attr].attrs['doc'] = "number of beam codes to nearest {:}".format(code) 
                return x.coords[attr]
            else:
                print('No event code {:} to drop'.format(code))
                return None
        except:
            traceback.print_exc('Cannot set drop_code = '.format(code))


def find_beam_correlations(xo, pvalue=1e-20, pvalue0_ratio=0.1, corr_pvalue=0.0001,
            groupby='ec162', nearest=5, corr_coord='delta_drop',
            pulse=None, confidence=0.1, sigma0=5,
            percentiles=[0.5], 
            save_file=None,
            cut=None, verbose=False, **kwargs):
    """
    """
    import traceback
    xcoor_coord = set_delta_beam(xo, code=groupby, attr=corr_coord)
    import xarray as xr
    import pandas as pd
    xstats = xr.Dataset()
    attrs = [a for a in xo if xo[a].dims == ('time',)]
    xds = xo[attrs].load()
    xo.attrs['drop_shot_detected'] = []
    xo.attrs['timing_error_detected'] = []
    xo.attrs['beam_warning_detected'] = []
    xo.attrs['beam_corr_detected'] = []
    if 'EBeam_damageMask' in xds:
        xds = xds.drop('EBeam_damageMask')
    if pulse not in xds:
        if 'FEEGasDetEnergy_f_21_ENRC' in xds:
            pulse = 'FEEGasDetEnergy_f_21_ENRC'
            if xds[pulse].sum() == 0:
                pulse = 'FEEGasDetEnergy_f_11_ENRC'

        else:
            pulse = 'GasDet_f21'
    xo.attrs['beam_corr_attr'] = pulse
    xo.attrs['beam_corr_confidence'] = confidence 
    xo.attrs['drop_attr'] = groupby

    print('Analyzing beam correlations for {:} Run {:}'.format(xo.experiment, xo.run))
    for attr in [a for a in xds.data_vars if xds[a].dims == ('time',)]:
        #if verbose:
        print '*****', attr, '*******'
        attrs = [attr, groupby, pulse]
        if cut:
            if cut in xds:
                attrs.append(cut)
            else:
                print('Ommitting cut: {:} not valid cut'.format(cut))
                cut = None

        x = xds[attrs]
        alias = x[attr].attrs.get('alias')
        attest = {}
        actest = {}

        for ishot in range(-nearest,nearest+1):
            if x[pulse].attrs.get('alias') == x[attr].attrs.get('alias'):
                # if FEEGasDetEnergy detector then test agains all other with shifted drops
                tnearest = None 
            else:
                tnearest = nearest
            ttest = ttest_groupby(x, attr, groupby=groupby, ishot=ishot, 
                                    nearest=tnearest)
            attest[ishot] = ttest
            if ttest is None:
                if verbose:
                    print attr, groupby, 'Not valid test'
                continue
         
            # Do not test correlation of same attr
            ctest = test_correlation(x, attr, pulse, cut=cut, shift=ishot)
            actest[ishot] = ctest

        xstd = x[attr].groupby(groupby).std()
        xmean = x[attr].groupby(groupby).mean()
        try:
            # Statistics for delta_drop
            df = xo.reset_coords()[[attr,corr_coord]].to_dataframe()
            df_nearest = df.where(abs(df[corr_coord]) <= nearest).dropna()
            df_table = df_nearest[[attr,corr_coord]].groupby(corr_coord).describe(percentiles=percentiles)[attr]

            # T-test for the means of *two independent* samples of scores.
            # see scipy.stats.ttest_ind
            # The calculated t-statistic, The two-tailed p-value
            df_ttest = pd.DataFrame(attest,index=['t_stat','t_pvalue']).T
            # Pearson correlation coefficient and the p-value for testing non-correlation.
            # see scipy.stats.pearsonr
            # Pearson's correlation coefficient, 2-tailed p-value
            df_ctest = pd.DataFrame(actest,index=['beam_corr','c_pvalue']).T        
            df_stats = df_table.join(df_ctest).join(df_ttest)
            
            tag_shot_corr = False
            t_pvalue0 = df_stats['t_pvalue'][0]
            if abs(float((xmean[1]-xmean[0])/xstd[0])) > sigma0 or xstd[1] > xstd[0]*sigma0:
                # First check if dropped shot and regular shot mean values differ by > sigma0 
                ishot = 0
                tag_shot_corr = True
            else:
                # Otherwise choose shot with greatest ttest significance
                ishot = df_stats['t_stat'].abs().idxmax()
            
            try:
                # ishot can be nan when t_stat are nan
                t_pvalue = df_stats['t_pvalue'][ishot]
                sig_significance = (df_stats['mean']/df_stats['std'])[ishot]
                # Check pvalue valid and if not timed with drop_code then
                # check ratio of found ishot pvalue is less than pvalue on drop code
                # Ignore if mean/std for time detected is too big to be reasonable
                if sig_significance < 1.e10 and t_pvalue0 != 1 and t_pvalue <= pvalue \
                            and (t_pvalue/t_pvalue0 < pvalue0_ratio or ishot == 0):
                    tag_shot_corr = True

            except:
                t_pvalue = None

           
            try:
                shot_corr_detected = False
                shot_corr = df_stats['beam_corr'].abs().idxmax()
                c_pvalue = df_stats['c_pvalue'][shot_corr]
                beam_corr = df_stats['beam_corr'][shot_corr]
                if shot_corr != 0:
                    c_pvalue0 = df_stats['c_pvalue'][0]
                    beam_corr0 = df_stats['beam_corr'][0]
                    # default to correlated with X-rays if not clear correlation on timing error
                    if c_pvalue < corr_pvalue and abs(beam_corr) > 0.2:
                        shot_corr_detected = True
                        # Make sure shot correlation is not also OK on drop shot
                        if (beam_corr0 > beam_corr/2. or (c_pvalue>0 and c_pvalue0/c_pvalue >= pvalue0_ratio)):
                            shot_corr = 0
                            beam_corr = df_stats['beam_corr'][shot_corr]
                            c_pvalue = df_stats['c_pvalue'][shot_corr]
                else:
                    dfs = df_stats['c_pvalue'].copy()
                    dfs.pop(0)
                    c_pvalue0 = dfs.min()
                    dfs = df_stats['beam_corr'].copy()
                    dfs.pop(0)
                    beam_corr0 = dfs.max()
                    #beam_corr = df_stats['beam_corr'][shot_corr]
                    # If off-by-one make sure not beam correlated on ec162
                    if ishot != 0 and c_pvalue < corr_pvalue and \
                                (abs(beam_corr) > 0.2 or abs(beam_corr) > abs(beam_corr0)*2.):
                        tag_shot_corr = False
                    elif ishot != 0 and (c_pvalue == 1 or c_pvalue0 == 1):
                        tag_shot_corr = False

                    # Make checks to be sure beam_corr is valid 
                    if c_pvalue0 == 0 or c_pvalue/c_pvalue0 > 1./pvalue0_ratio:
                        shot_corr_detected = False
                    elif c_pvalue == 0 and c_pvalue0 < pvalue**2 and beam_corr0>beam_corr/2.:
                        shot_corr_detected = False
                    elif c_pvalue < corr_pvalue and (abs(beam_corr) > 0.2 or abs(beam_corr) > abs(beam_corr0)*2.):
                        shot_corr_detected = True

                # Make sure beam correlation is consistent with drop shot detection
                # If beam_corr is on ec162 then do not tag
                if shot_corr_detected:
                    if tag_shot_corr and ishot != shot_corr:
                        if shot_corr == 0:
                            tag_shot_corr = False
                            ishot = shot_corr
                            t_pvalue = df_stats['t_pvalue'][ishot]
                    
                    xo[attr].attrs['beam_corr'] = beam_corr
                    xo[attr].attrs['shot_corr'] = shot_corr
                    if attr not in xo.attrs['beam_corr_detected']:
                        xo.attrs['beam_corr_detected'].append(attr)
                    note = 'Beam-corr  of {:5.3f}'.format(beam_corr)
                    corr_note = '{:} on {:2} shot for {:} detector {:} (corr = {:}, c_pvalue = {:})'.format(note, shot_corr, 
                                alias, attr, beam_corr, c_pvalue)
                    print corr_note

            except:
                shot_corr = 0
                c_pvalue = None
                beam_corr = None
                shot_corr_detected = False

            # Tag shot correlation after checking beam correlation
            if tag_shot_corr: 
                xo[attr].attrs['delta_beam'] = ishot 
                xo[attr].attrs['delta_beam_pvalue'] = t_pvalue
                if ishot == 0:
                    note = 'Drop-shot detected '
                    xo[attr].attrs['drop_shot_detected'] = True
                    if attr not in xo.attrs['drop_shot_detected']:
                        xo.attrs['drop_shot_detected'].append(attr)
                elif alias in ['EBeam','PhaseCavity'] or attr.endswith('xpos') or attr.endswith('ypos'):
                    note = 'Beam-warning detected'
                    xo[attr].attrs['beam_warning_detected'] = True
                    if attr not in xo.attrs['beam_warning_detected']:
                        xo.attrs['beam_warning_detected'].append(attr)
                else:
                    note = 'Time-error detected'
                    xo[attr].attrs['timing_error_detected'] = True
                    if attr not in xo.attrs['timing_error_detected']:
                        xo.attrs['timing_error_detected'].append(attr)
                
                shot_note = '{:} on {:2} shot for {:} detector {:} (t_pvalue={:})'.format(note, ishot, 
                        alias, attr, t_pvalue)
                print shot_note


            #xstats[attr] = ((corr_coord,'drop_stats'), df_stats)
            xstats[attr] = df_stats

        except:
            traceback.print_exc()
            print 'Cannot calc stats for', attr, ishot
        
    for avar, da in xo.data_vars.items():
        try:
            if avar in xstats:
                for a in ['doc', 'unit', 'alias']:
                    val = xo[avar].attrs.get(a, '') 
                    if isinstance(val, unicode):
                        val = str(val)
                    xstats[avar].attrs[a] = val 
        except:
            print('Cannot add attrs for {:}'.format(avar))
    
    for attr, val in xo.attrs.items():
        try:
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], unicode):
                val = [str(v) for v in val]
            elif isinstance(val, unicode):
                val = str(val)
            xstats.attrs[attr] = val
        except:
            print('Cannot add attrs for {:}'.format(attr))

    if 'dim_1' in xstats:
        xstats = xstats.rename({'dim_1':'drop_stats'})
    else:
        print('No Drop stats')
        return None

    if save_file:
        try:
            xstats.to_netcdf(save_file, engine='h5netcdf')
            print('Save drop summary file: {:}'.format(save_file))
        except:
            print('Cannot save file: {:}'.format(save_file))

    return xstats
