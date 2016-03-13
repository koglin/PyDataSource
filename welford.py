import numpy as np

class Welford(object):

    def __init__(self, x=None):
        self._K = 0 
        self.n = 0
        self._Ex = 0
        self._Ex2 = 0
        self.shape = None
        self._min = None
        self._max = None
        self._init = False
        self.__call__(x)
         
    def add_data(self, x):
        if x is None:
            return
        
        x = np.array(x)
        if not self._init:
            self._init = True
            self._K = x
            self._min = x
            self._max = x
            self.shape = x.shape
        else:
            self._min = np.minimum(self._min, x)
            self._max = np.maximum(self._max, x)
        
        self.n += 1
        self._Ex += x - self._K
        self._Ex2 += (x - self._K) * (x - self._K)
    
    def __call__(self, x):
        self.add_data(x)

    def max(self):
        return self._max

    def min(self):
        return self._min

    def mean(self, axis=None):
        """Compute the mean of accumulated data.
           
           Parameters
           ----------
           axis: None or int or tuple of ints, optional
                Axis or axes along which the means are computed. The default is to
                compute the mean of the flattened array.
        """
        if self.n < 1:
            return None

        val = np.array(self._K + self._Ex / np.float(self.n))
        if axis:
            return val.mean(axis=axis)
        else:
            return val

    def var(self):
        """Compute the variance of accumulated data.
        """
        if self.n <= 1:
            return  np.zeros(self.shape)
            
        val = np.array((self._Ex2 - (self._Ex*self._Ex)/self.n) / np.float(self.n-1))

        return val

    def std(self):
        """Compute the standard deviation of accumulated data.
        """
        return np.sqrt(self.var())

    def __str__(self):
        if self._init:
            return "{} +- {}".format(self.mean(), self.std())
        else:
            return "{}".format(self.shape)

    def __repr__(self):
        return "< Welford: {:} >".format(str(self))
