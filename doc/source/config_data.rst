.. _config_data:

.. currentmodule:: PyDataSource

Configuration Data
******************

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

