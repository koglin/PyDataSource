.. _data_access:

.. currentmodule:: PyDataSource

Data Access Overview
********************

The **PyDataSource** package provides access to LCLS event data by wrapping the psana "Detector" interface.  This page provides a quick introduction to basic event data access.  See other pages and the API section for futher details and options.

Analysis Environment
--------------------

Use the following environment Setup from psana machine.  Start ipython with the --pylab option (or if youy prefer import numpy and matplotlib as needed).

.. code-block:: bash 

    .  /reg/g/psdm/etc/ana_env.sh
    source conda_setup
    ipython --pylab
 
Data Source
-----------

Use run and exp keywords to access your data using **PyDataSource.DataSource**.

.. sourcecode:: ipython

    In [1]: import PyDataSource

    In [2]: ds = PyDataSource.DataSource(exp='xpptut15',run=54)

    In [3]: ds.configData
    Out[3]: < ConfigData: exp=mfx12616:run=45:smd >
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

    In [4]: evt  = ds.events.next()

Press tab to see detectors in event (according to aliases defined in daq).

.. sourcecode:: ipython
  
    In [5]: evt.
             evt.cspad           evt.Evr             evt.keys            evt.PhaseCavity     evt.XppSb3_Ipm      
             evt.EBeam           evt.FEEGasDetEnergy evt.L3T             evt.XppEnds_Ipm0    evt.yag2            
             evt.EventId         evt.get             evt.next            evt.XppSb2_Ipm      evt.yag_lom         

Calibrated Area Detector
------------------------

For complex detectors such as the cspad, the detector object gives access to the raw, calibrated and reconstructed images.
 - raw:    raw uncorrected data (3D array)
 - calib:  calibrated "unassembled data" (3D array, pedestal and common mode corrected, no geometry applied) 
 - image:  "assembled images" (2D array, geometry applied).

Tab on the evt.cspad object to see the attributes available and use the show_info method to print a summary table.  

.. sourcecode:: ipython

    In [36]: evt.cspad.
                evt.cspad.add        evt.cspad.epicsData  evt.cspad.L3T        evt.cspad.set_cmpars  
                evt.cspad.calib      evt.cspad.EventId    evt.cspad.monitor    evt.cspad.shape       
                evt.cspad.calibData  evt.cspad.Evr        evt.cspad.next       evt.cspad.show_all   >
                evt.cspad.configData evt.cspad.evtData    evt.cspad.psplots    evt.cspad.show_info   
                evt.cspad.detector   evt.cspad.image      evt.cspad.raw        evt.cspad.size        

    In [35]: evt.cspad.show_info()
    Out[35]: 
    --------------------------------------------------------------------------------
    cspad xpptut15, Run 54, Step 0, Event 0, 11:37:12.4517, [140, 141, 41, 40]
    --------------------------------------------------------------------------------
    calib                 <0.01065> ADU     Calibrated data
    image                <0.008139> ADU     Reconstruced 2D image from calibStore geometry
    raw                  <1.57e+03> ADU     Raw data
    shape              (32, 185, 388)         Shape of raw data array
    size                    2296960         Total size of raw data

See example below of plotting an image with the matplotlib plotting library

.. plot:: examples/cspad.py
   :include-source:

Event Iteration
---------------

One can iterate events from the ds.events object, the evt object, or a detector object.  Each of these are python iterators.

The representation of each of these provides details of the event including the event codes present.

.. sourcecode:: ipython

    In [6]: ds.events.next()
    Out[6]: < EvtDetectors: xpptut15, Run 54, Step 0, Event 1, 11:37:12.4599, [140, 40] >

    In [7]: evt.next()
    Out[7]: < EvtDetectors: xpptut15, Run 54, Step 0, Event 2, 11:37:12.4683, [140, 141, 142, 143, 144, 41, 42, 43, 44, 40] >

    In [8]: evt.cspad.next()
    Out[8]: < Detector: cspad xpptut15, Run 54, Step 0, Event 3, 11:37:12.4768, [140, 40] >

    In [9]: for evt in ds.events:
        ...:     print evt
        ...:     if ds._ievent == 6:
        ...:         break
        ...:     
    xpptut15, Run 54, Step 0, Event 4, 11:37:12.4849, [140, 141, 41, 40]
    xpptut15, Run 54, Step 0, Event 5, 11:37:12.4934, [140, 40]
    xpptut15, Run 54, Step 0, Event 6, 11:37:12.5019, [140, 141, 142, 41, 42, 40]


Event Codes
-----------

To check if an event codes is present:

.. sourcecode:: ipython

    In [11]: evt.Evr.present(40)
    Out[11]: True

    In [12]: evt.Evr.present(41)
    Out[12]: False

Beamline Data
-------------

The following Electron and Photon Beam data are generally recorded for all experiments.

.. sourcecode:: ipython

    In [17]: evt.EBeam.show_info()
    Out[17]: 
    --------------------------------------------------------------------------------
    EBeam xpptut15, Run 54, Step 0, Event 3, 11:37:12.4768, [140, 40]
    --------------------------------------------------------------------------------
    damageMask                    1048574         Damage mask.
    ebeamCharge                0.00088572 nC      Beam charge in nC.
    ebeamDumpCharge                     0 e-      Bunch charge at Dump in num. electrons
    ebeamEnergyBC1                -13.772 mm      Beam position in mm (related to beam energy).
    ebeamEnergyBC2               -0.38553 mm      Beam position in mm (related to beam energy).
    ebeamL3Energy                       0 MeV     Beam energy in MeV.
    ebeamLTU250                         0 mm      LTU250 BPM value in mm, used to compute photon energy. from BPMS:LTU1:250:X
    ebeamLTU450                         0 mm      LTU450 BPM value in mm, used to compute photon energy. from BPMS:LTU1:450:X
    ebeamLTUAngX                        0 mrad    LTU beam angle in mrad.
    ebeamLTUAngY                        0 mrad    LTU beam angle in mrad.
    ebeamLTUPosX                        0 mm      LTU beam position (BPMS:LTU1:720 through 750) in mm.
    ebeamLTUPosY                        0 mm      LTU beam position in mm.
    ebeamPhotonEnergy                   0 eV      computed photon energy, in eV
    ebeamPkCurrBC1                 33.661 Amps    Beam current in Amps.
    ebeamPkCurrBC2             7.1578e+08 Amps    Beam current in Amps.
    ebeamUndAngX                        0 mrad    Undulator launch feedback beam x-angle in mrad.
    ebeamUndAngY                        0 mrad    Undulator launch feedback beam y-angle in mrad.
    ebeamUndPosX                        0 mm      Undulator launch feedback (BPMs U4 through U10) beam x-position in mm.
    ebeamUndPosY                        0 mm      Undulator launch feedback beam y-position in mm.
    ebeamXTCAVAmpl                      0 MVolt   XTCAV Amplitude in MVolt.
    ebeamXTCAVPhase                     0 degrees XTCAV Phase in degrees.

    In [18]: evt.FEEGasDetEnergy.show_info()
    Out[18]: 
    --------------------------------------------------------------------------------
    FEEGasDetEnergy xpptut15, Run 54, Step 0, Event 3, 11:37:12.4768, [140, 40]
    --------------------------------------------------------------------------------
    f_11_ENRC                      3.3573 mJ      First energy measurement (mJ) before attenuation. (pv name GDET:FEE1:241:ENRC)
    f_12_ENRC                  -0.0015399 mJ      Second (duplicate!) energy measurement (mJ) after attenuation. (pv name GDET:FEE1:242:ENRC)
    f_21_ENRC                  -0.0026742 mJ      First energy measurement (mJ) after attenuation. (pv name  GDET:FEE1:361:ENRC)
    f_22_ENRC                   0.0009475 mJ      Second (duplicate!) energy measurement (mJ) after attenuation. (pv name GDET:FEE1:362:ENRC)
    f_63_ENRC                   -0.012089 mJ      First energy measurement (mJ) for small signals (<0.5 mJ), after attenuation. (pv name GDET:FEE1:363:ENRC)
    f_64_ENRC                  -0.0032068 mJ      Second (duplicate!) energy measurement (mJ) for small signals (<0.5mJ), after attenutation. (pv name GDET:FEE1:364:ENRC)

    In [19]: evt.PhaseCavity.show_info()
    Out[19]: 
    --------------------------------------------------------------------------------
    PhaseCavity xpptut15, Run 54, Step 0, Event 3, 11:37:12.4768, [140, 40]
    --------------------------------------------------------------------------------
    charge1                       0.11027 pico-columbs UND:R02:IOC:16:BAT:Charge1 value in pico-columbs.
    charge2                       0.13702 pico-columbs UND:R02:IOC:16:BAT:Charge2 value in pico-columbs.
    fitTime1                       70.333 pico-seconds UND:R02:IOC:16:BAT:FitTime1 value in pico-seconds.
    fitTime2                     -0.31647 pico-seconds UND:R02:IOC:16:BAT:FitTime2 value in pico-seconds.

Epics Data
----------

The LCLS DAQ includes many slowly changing quantities (e.g. voltages, temperatures, motor positions) that are recorded with software called EPICS.  These can either be accessed by aliases or PV name.  

.. sourcecode:: ipython

    In [26]: ds = PyDataSource.DataSource(exp='xpptut15',run=59)

    In [26]: evt = ds.events.next() 

    In [27]: ds.epicsData. 
                  ds.epicsData.alias        ds.epicsData.ccm          ds.epicsData.CSPAD        ds.epicsData.filt         ds.epicsData.hrm1         ds.epicsData.ipm1          
                  ds.epicsData.aliases      ds.epicsData.ccmE         ds.epicsData.drift        ds.epicsData.FS3          ds.epicsData.hrm2         ds.epicsData.ipm1b         
                  ds.epicsData.alio         ds.epicsData.ccmTheta0    ds.epicsData.epicsConfig  ds.epicsData.getPV        ds.epicsData.HX2          ds.epicsData.ipm2         >
                  ds.epicsData.analogOut    ds.epicsData.ChipAddress  ds.epicsData.EVR          ds.epicsData.gon          ds.epicsData.HX3          ds.epicsData.ipm3          
                  ds.epicsData.Be           ds.epicsData.ChipName     ds.epicsData.FeeAtt       ds.epicsData.grid         ds.epicsData.ipm          ds.epicsData.ire           

    In [27]: ds.epicsData.SampleTemp.show_info()
    Out[27]: 
    SampleTemp_GetA                         480 -- XPP:USR:TCT:01:GET_TEMP_A                
    SampleTemp_GetB                         480 -- XPP:USR:TCT:01:GET_TEMP_B                
    SampleTemp_GetC                       189.5 -- XPP:USR:TCT:01:GET_TEMP_C                
    SampleTemp_GetD                        3.15 -- XPP:USR:TCT:01:GET_TEMP_D                
    SampleTemp_Set                           20 -- XPP:USR:TCT:01:PUT_SOLL_1                
    SampleTemp_SetRB                         20 -- XPP:USR:TCT:01:GET_SOLL_1                

    In [28]: ds.epicsData.SampleTemp.GetA.value
    Out[28]: 480.0

Epics data associated with detectors (using the convention of using an alias starting with the detector name) can be accessed either from the ds.epicsData object or more conveniently from the epicsData object in the detector.

.. sourcecode:: ipython

    In [28]: ds.epicsData.yag2.show_info()
    Out[28]: 
    yag2_focus                           -1.151 -- XPP:SB3:CLF:01.RBV                       
    yag2_zoom                                65 -- XPP:SB3:CLZ:01.RBV                       

    In [29]: evt.yag2.epicsData.show_info()
    Out[29]: 
    yag2_focus                           -1.151 -- XPP:SB3:CLF:01.RBV                       
    yag2_zoom                                65 -- XPP:SB3:CLZ:01.RBV                       

Waveform Detectors
------------------

There are several types of voltage-versus-time ("waveform") detectors supported: 
 - Acqiris (now "Agilent U1065A") 
 - 'Imp' detectors (SLAC).
 - 'Wave8' detectors (SLAC).

.. plot:: examples/acqiris.py
   :include-source:

NOTE: For Acqiris detectors the user can often find between 0-7 "zeros" at the end of the arrays.  This is because the acqiris will read out 40,000 samples, for example, but the trigger is not necessarily on sample 0: from shot-to-shot it can vary between 0 and 7. 

In the above user interface we move the data arrays so the trigger happens on sample 0 (so users can more easily add results from different events, for example). But after we do this, to keep the array sizes constant, we pad the end with zeros. This is not ideal, but it makes the user-interface simpler, which is important for LCLS users.

In this example we use older data that does not have aliases, so the detectors names are generated from their 'daq' identifiers.


Anlyzing Scans
--------------
Information about daq scans (where things like motor positions are changed during a run in "steps" or "calibcycles") is accessible from the ds.configData.  The ScanData object may take dozens of seconds to load the first time depending on the length of the scan.  Once loaded you have access to the scan information as lists the length of the number of steps in the scan. 

.. sourcecode:: ipython

    In [2]: import PyDataSource

    In [3]: ds  = PyDataSource.DataSource(exp='xpptut15',run=200)

    In [4]: %time ds.configData.ScanData.show_info()
    CPU times: user 51.7 s, sys: 2.4 s, total: 54.1 s
    Wall time: 50.5 s
    Out[4]: 
    xpptut15  : Run 200
    ----------------------------------------------------------------------
    Number of steps                  45 nsteps          
    Number of monitor PVs             0 npvMonitors     
    Number of control PVs             1 npvControls     

    Alias                    PV                                      
    ----------------------------------------------------------------------
    lxt_vitara_ttc           lxt_vitara_ttc                          

    Step Events   Time [s] lxt_vitara_ttc
    -----------------------------------
       0   1306   19.179     -1.000e-12
       1    625    5.207     -7.500e-13
       2    629    5.239     -5.000e-13
       3    635    5.289     -2.500e-13
       4    635    5.289     -8.470e-22
       5   3413   37.788      2.500e-13
       6    623    5.190      5.000e-13
       7    629    5.240      7.500e-13
       8    629    5.240      1.000e-12
       9    629    5.241      1.250e-12
      10    635    5.290      1.500e-12
      11    635    5.291      1.750e-12
      12    635    5.291      2.000e-12
      13    629    5.241      2.250e-12
      14   1487   17.988      2.500e-12
      15    623    5.190      2.750e-12
      16    623    5.189      3.000e-12
      17    623    5.190      3.250e-12
      18    629    5.240      3.500e-12
      19    629    5.240      3.750e-12
      20    637    5.306      4.000e-12
      21    629    5.240      4.250e-12
      22    777    6.473      4.500e-12
      23    629    5.240      4.750e-12
      24    629    5.240      5.000e-12
      25    629    5.240      5.250e-12
      26    629    5.240      5.500e-12
      27    629    5.239      5.750e-12
      28    629    5.239      6.000e-12
      29    629    5.240      6.250e-12
      30    629    5.241      6.500e-12
      31    629    5.241      6.750e-12
      32    623    5.191      7.000e-12
      33    629    5.240      7.250e-12
      34    623    5.191      7.500e-12
      35   1361   11.340      7.750e-12
      36    767    6.388      8.000e-12
      37    899    7.488      8.250e-12
      38   1037    8.640      8.500e-12
      39    629    5.240      8.750e-12
      40    629    5.240      9.000e-12
      41    623    5.190      9.250e-12
      42    623    5.190      9.500e-12
      43    629    5.240      9.750e-12
      44    629    5.241      1.000e-11

    In [5]:  ds.configData.ScanData.control_values
                     ds.configData.ScanData.control_values   ds.configData.ScanData.nevents           
                     ds.configData.ScanData.end_times        ds.configData.ScanData.npvControls       
                     ds.configData.ScanData.lxt_vitara_ttc   ds.configData.ScanData.npvMonitors      >
                     ds.configData.ScanData.monitor_hivalues ds.configData.ScanData.nsteps            
                     ds.configData.ScanData.monitor_lovalues ds.configData.ScanData.pvAliases         

  
Here is a quick way to iterate over steps:

.. sourcecode:: ipython

    In [6]:  for istep, stepevts in enumerate(ds.steps):
        ...:     evt = stepevts.next()
        ...:     print istep, evt
        ...:     
    0 xpptut15, Run 200, Step 0, Event 0, 07:17:55.8309, [141, 90, 40, 41, 140]
    1 xpptut15, Run 200, Step 1, Event 0, 07:18:17.5429, [141, 142, 91, 40, 41, 42, 140]
    2 xpptut15, Run 200, Step 2, Event 0, 07:18:25.0651, [90, 40, 140]
    3 xpptut15, Run 200, Step 3, Event 0, 07:18:32.9365, [60, 141, 142, 91, 40, 41, 42, 140]
    4 xpptut15, Run 200, Step 4, Event 0, 07:18:40.8244, [90, 40, 140]
    ...


Reloading DataSource
--------------------

Use the reload method to start back at the beginning of a run.

.. sourcecode:: ipython

    In [64]: ds.reload()

    In [65]: evt = ds.events.next()

    In [66]: evt
    Out[66]: < EvtDetectors: xpptut15, Run 200, Step 0, Event 0, 07:17:55.8309, [141, 90, 40, 41, 140] >

    In [67]: evt.next()
    Out[67]: < EvtDetectors: xpptut15, Run 200, Step 0, Event 1, 07:17:55.8392, [50, 51, 60, 61, 90, 40, 140] >


Jump Quickly to Specific Event
------------------------------

Time stamps can be used to jump to specific event in the next method.  Alternatively an interger event number can be supplied to goto event number in DataSource (although it may not be exactly same event depending on how the data_source string is corresponding keywords to define the data_source is defined and also may differ for fast feedback and offline analysis environments).

This example shows jumping to event 200, saving the EventTime, going back to event 0, and then using the saved EventTime to jump back to event 200.  

.. sourcecode:: ipython

    In [67]: evt.next()
    Out[67]: < EvtDetectors: xpptut15, Run 200, Step 0, Event 1, 07:17:55.8392, [50, 51, 60, 61, 90, 40, 140] >

    In [68]: evt.next(200)
    Out[68]: < EvtDetectors: xpptut15, Run 200, Step -1, Event 200, 07:17:57.4975, [50, 60, 141, 90, 40, 41, 140] >

    In [69]: et200 = evt.EventId.EventTime

    In [70]: evt.next(0)
    Out[70]: < EvtDetectors: xpptut15, Run 200, Step -1, Event 0, 07:17:55.8309, [141, 90, 40, 41, 140] >

    In [71]: evt.next(et200)
    Out[71]: < EvtDetectors: xpptut15, Run 200, Step -1, Event 200, 07:17:57.4975, [50, 60, 141, 90, 40, 41, 140] >


If nothing is passed in the next method, then the next event goes to the next event before we started jumping to specific events.

.. sourcecode:: ipython

    In [72]: evt.next()
    Out[72]: < EvtDetectors: xpptut15, Run 200, Step 0, Event 2, 07:17:55.8476, [50, 60, 141, 142, 90, 40, 41, 42, 140] >

   
    
    
    
    
    
    


