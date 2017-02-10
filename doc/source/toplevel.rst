.. _data_access

.. currentmodule:: data_access

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import os
    np.random.seed(123456)
    np.set_printoptions(precision=4, suppress=True)
    import matplotlib
    matplotlib.style.use('ggplot')
    pd.options.display.max_rows = 15

    #### portions of this were borrowed from the
    #### Pandas cheatsheet
    #### created during the PyData Workshop-Sprint 2012
    #### Hannah Chen, Henry Chow, Eric Cox, Robert Mauriello



Data Access
*****************
The **PyDataSource** package provides access to LCLS event data by wrapping the psana "Detector" interface.  

Use run and exp keywords to access your data.

.. sourcecode:: ipython
   
     [1]: a = [1,1]
     [2]: b = ['a','b']
     b

Writiing Python
---------------

write like this

.. ipython:: python
   
     a = [1,1]
     b = ['a','b']

next
----

ploting


.. plot:: pyplots/ellipses.py
   :include-source:


.. sourcecode:: ipython

   In [1]: import PyDataSource

   In [2]: ds = PyDataSource.DataSource(exp='xpptut15',run=54)
    
   In[3]: ds.configData
         < ConfigData: exp=mfx12616:run=45:smd >
        *Detectors in group 0 are "BLD" data recorded at 120 Hz on event code 40
        *Detectors listed as Controls are controls devices with unknown event code (but likely 40).

        Alias                     Group          Rate  Code  Pol. Delay [s]    Width [s]    Source                    
        ------------------------------------------------------------------------------------------------------------------------
        Acqiris                       1                 198   Pos 0.000147630  0.000010000  DetInfo(MfxEndstation.0:Acqiris.0)      
        BeamMonitor                   1                 198                                 DetInfo(MfxEndstation.0:Wave8.0)        
        EBeam                         1                                                     BldInfo(EBeam)                          
        FEEGasDetEnergy               1                                                     BldInfo(FEEGasDetEnergy)                
        FEE_Spec                      1                                                     BldInfo(FEE-SPEC0)                      
        PhaseCavity                   1                                                     BldInfo(PhaseCavity)                    
        Rayonix                       1                 198   Neg 0.002998336  0.000500000  DetInfo(MfxEndstation.0:Rayonix.0)      
        < ConfigData: exp=mfx12616:run=45:smd >

            

Event Iteration
***************

Compatability with psana
------------------------

.. code-block:: python

   ds = PyDataSource.DataSource('exp=xpptut15:run=54:smd')




