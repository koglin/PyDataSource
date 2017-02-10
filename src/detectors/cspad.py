import PyDataSource
import os

from pylab import *

class Cspad(PyDataSource.Detector):
    """Cspad Detector Class.
    """

    def __init__(self,*args,**kwargs):

        PyDataSource.Detector.__init__(self,*args,**kwargs)
 

