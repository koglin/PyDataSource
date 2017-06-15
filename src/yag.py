import PyDataSource
import os

from scipy import ndimage

class Yag(PyDataSource.Detector):
    """Yag Detector.
    """

    def __init__(self,*args,**kwargs):

        PyDataSource.Detector.__init__(self,*args,**kwargs)

        self.add.property(center_xpos, doc='Yag X center_of_mass')
        self.add.property(center_ypos, doc='Yag Y center_of_mass')
        self.add.projection('calib', axis='x', name='img_y', axis_name='yimg')
        self.add.projection('calib', axis='y', name='img_x', axis_name='ximg')
        self.add.count('calib', name='img_count')


def center_xpos(self): 
    return ndimage.measurements.center_of_mass(self.calib)[0]

def center_ypos(self):
    return ndimage.measurements.center_of_mass(self.calib)[1]

