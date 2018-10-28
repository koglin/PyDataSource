.. _config_data:

.. currentmodule:: PyDataSource

Configuration Data
******************

Configuration data for each detector as well as for example daq control information is available
as configData.  

.. sourcecode:: ipython

    In [14]: ds.configData.ControlData.show_info()
    Out[14]: 
    duration                 seconds: 0 nano:0         Maximum duration of the scan.
    events                              0         Maximum number of events per scan.
    npvControls                         0         Number of PVControl objects in this configuration.
    npvLabels                           0         Number of PVLabel objects in this configuration.
    npvMonitors                         0         Number of PVMonitor objects in this configuration.
    pvControls                         []         PVControl configuration objects
    pvLabels                           []         PVLabel configuration objects
    pvMonitors                         []         PVMonitor configuration objects
    uses_duration                       0         returns true if the configuration uses duration control.
    uses_events                         1         returns true if the configuration uses events limit.
    uses_l3t_events                     0         returns true if the configuration uses l3trigger events limit.

    In [15]: ds.configData.ScanData.show_info()
    Out[15]: 
    xpptut15  : Run 54
    ----------------------------------------------------------------------
    Number of steps                   1 nsteps          
    Number of monitor PVs             0 npvMonitors     
    Number of control PVs             0 npvControls     

    Alias                    PV                                      
    ----------------------------------------------------------------------

    Step Events   Time [s]
    ---------------------
       0   1218   10.154

    In [16]: ds.configData.Sources
    Out[16]: < ConfigSources: exp=xpptut15:run=54:smd >

    In [17]: ds.configData.Sources.show_info()
    Out[17]: 
    *Detectors in group 0 are "BLD" data recorded at 120 Hz on event code 40
    *Detectors listed as Controls are controls devices with unknown event code (but likely 40).

    Alias                     Group          Rate  Code  Pol. Delay [s]    Width [s]    Source                    
    ------------------------------------------------------------------------------------------------------------------------
    EBeam                         0        120 Hz    40                                 BldInfo(EBeam)                          
    FEEGasDetEnergy               0        120 Hz    40                                 BldInfo(FEEGasDetEnergy)                
    PhaseCavity                   0        120 Hz    40                                 BldInfo(PhaseCavity)                    
    XppEnds_Ipm0                  0        120 Hz    40                                 BldInfo(XppEnds_Ipm0)                   
    XppSb2_Ipm                    0        120 Hz    40                                 BldInfo(XppSb2_Ipm)                     
    XppSb3_Ipm                    0        120 Hz    40                                 BldInfo(XppSb3_Ipm)                     
    cspad                         1        120 Hz    40   Pos 0.000549832  0.000010000  DetInfo(XppGon.0:Cspad.0)               
    yag2                          1        120 Hz    40   Pos 0.000690739  0.000300000  DetInfo(XppSb3Pim.1:Tm6740.1)           
    yag_lom                       1        120 Hz    40   Pos 0.000690739  0.000300000  DetInfo(XppMonPim.1:Tm6740.1)           

    In [18]: ds.configData.Sources.cspad
    Out[18]: < SourceData: cspad = DetInfo(XppGon.0:Cspad.0) >

    In [19]: ds.configData.Sources.cspad.show_info()
    Out[19]: 
    evr_width               0.000010000 s   Evr trigger width                       
    group                             1     Evr group                               
    eventCode                        40     Evr event code                          
    src                    DetInfo(XppGon.0:Cspad.0)                                             
    evr_delay               0.000549832 s   Evr trigger delay                       
    map_key                      (0, 3)     Evr configuation map key (card,channel) 
    alias                         cspad                                             
    evr_polarity                      0     Evr trigger polarity                    

    In [20]: ds.configData.                         
                            ds.configData.ControlData  ds.configData.keys         ds.configData.ScanData     ds.configData.Sources      ds.configData.XppSb3_Ipm   
                            ds.configData.cspad        ds.configData.Partition    ds.configData.show_all     ds.configData.XppEnds_Ipm0 ds.configData.yag2         
                            ds.configData.get          ds.configData.put          ds.configData.show_info    ds.configData.XppSb2_Ipm   ds.configData.yag_lom      


This information is gathered from the psana.env() configStore objection.  As with evt data, the configData object
makes the configStore keys, get and put methods avilable from the configStore object.  


Detector Timing and Config Check
--------------------------------

The detector timing and configuration can be checked against recommended operating values with 
the checkConfig methods in configData object.

e.g., the recommended evr width for the 2.3M CsPad detectors is 0.1 ms.  
In this example the width was set to 3 ms, and only a warning was issued 
since this is generally not expected to cause an issue with the detector
readout.

.. sourcecode:: ipython

    In [6]: ds = PyDataSource.DataSource(exp='cxix34517', run=42)

    In [7]: ds.configData.configCheck.show_info()
    Out[7]: 
    detector        parameter      level    error                 
    --------------------------------------------------------------
    DsdCsPad        evr_width      warning  0.003 > 0.0001        


The recommended detector timing settings from confluence:

- https://confluence.slac.stanford.edu/x/kIWhBg

There are two files in json format.

One for the detector configdb information:

- /reg/g/psdm/utils/arp/config/default/config_alert.json

And one for the evr trigger information:

- /reg/g/psdm/utils/arp/config/default/trigger_alert.json


The structure is:

- devName (.e.g., Cspad, Opal1000)
  - alert/warning level (still tbd if emails or log posts are generated with alerts)
    - attribute (e.g., evr_width, evr_polarity, evr_delay)
      - value:  either a single value or a list of the valid range [low, high]

For example for the Cspad2x2 (i.e., 140k) detectors, a common readout issue
is setting the wrong polarity or delay.  
The line in the default trigger_alert file is specified as

.. code-block:: bash

    "Cspad2x2":{
        "alert":{"evr_polarity":"Pos","evr_delay":[0.00048,0.00052]},
        "warning":{"evr_width":[0.00001,0.0001]}
        },


To make things more flexibile for each instrument to put in specific alerts 
that may conflict with other instruments, there will be folders for each instrument.  
The intent is that they generally will point to the default 
unless instrument specific values are desired.  i.e.,

.. code-block:: bash

    ls /reg/g/psdm/utils/arp/config -l
    total 4
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 19  2018 amo -> default
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 20  2018 cxi -> default
    drwxrwsr-x 2 koglin ps-pcds 5 May 18 12:36 default
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 19  2018 mec -> default
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 19  2018 mfx -> default
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 19  2018 sxr -> default
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 19  2018 xcs -> default
    lrwxrwxrwx 1 koglin ps-pcds 7 Apr 19  2018 xpp -> default


.. autosummary::
    :toctree: generated/

    config_check.ConfigCheck


ConfigData Class API
--------------------

.. autosummary::
    :toctree: generated/

    ConfigData


Attributes
----------

.. autosummary::
    :toctree: generated/

    ConfigData.Sources
    ConfigData.ScanData

