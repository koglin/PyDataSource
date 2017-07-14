import PyDataSource

class Cspad(PyDataSource.Detector):
    """Cspad Detector Class.
    """

    def __init__(self,*args,**kwargs):

        PyDataSource.Detector.__init__(self,*args,**kwargs)

        self.add.count('corr', name='corr_count')
        self.add.projection('corr', 'r', name='corr_r')

    def add_max_plot(self):
        """
        Add max count plot of CsPad
        """
        if not self._det_config['stats'].get('corr_stats'):
            self.add.stats('corr')
            self.next()
        self.add.property(img_max)
        self.add.psplot('img_max')

def img_max(self):
    """
    Method to make image from corr_stats
    """
    try:
        import numpy as np
        stat = 'max'
        code = self.configData.eventCode
        attr = 'corr_stats'
        dat = getattr(self, attr).sel(stat=stat).sel(codes=code)[-1]
        xx = self.calibData.indexes_x
        yy = self.calibData.indexes_y
        a = np.zeros([int(xx.max())+1,int(yy.max())+1])
        a[xx,yy] = dat.data
        return a
    except:
        return None

