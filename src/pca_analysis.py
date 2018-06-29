def pca(x):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import scale
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from scipy import stats
    from IPython.display import display, HTML

    # figures inline in notebook
    #%matplotlib inline

    np.set_printoptions(suppress=True)

    DISPLAY_MAX_ROWS = 20  # number of max rows to print for a DataFrame
    pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)


    plt.rcParams['axes.labelsize'] = 20
    attrs = ['Timetool_amplitude', 
             'Timetool_nxt_amplitude', 
             'Timetool_position_fwhm', 
             'Timetool_position_pixel', 
             #'Timetool_ref_amplitude', 
             'PhaseCavity_charge1', 
             #'laser_fs_corr', 'laser_fs_delay',
             'PhaseCavity_fitTime1', 
             #'PhaseCavity_fitTime2',
            ]
    itimes = x[attrs].groupby('Timetool_valid').groups[1]
    xselect = x[attrs].isel_points(time=itimes)
    df = xselect.to_array().to_pandas().T
    attr_names={'PhaseCavity_fitTime1': 'PhaseCavity_fitTime1', 'PhaseCavity_fitTime2': 'PhaseCavity_fitTime2', 
                'Timetool_ref_amplitude': 'ref_amplitude', 'Timetool_amplitude': 'amplitude', 
                'laser_fs_delay': 'laser_fs_delay', 'PhaseCavity_charge1': 'PhaseCavity_charge1', 
                'Timetool_position_fwhm': 'position_fwhm', 'Timetool_position_pixel': 'position_pixel', 
                'laser_fs_corr': 'laser_fs_corr', 'Timetool_nxt_amplitude': 'nxt_amplitude'}
    df.rename(inplace=True, columns=attr_names)
    dfs = df.apply(scale)
    pca = PCA(n_components=3).fit(dfs)


