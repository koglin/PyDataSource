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

.. ipython:: python 

    import PyDataSource

    ds = PyDataSource.DataSource(exp='xpptut15',run=54)

    ds.configData
    
Get the first event and press tab to see detectors in event (according to aliases defined in daq).

.. ipython:: 
 
    evt  = ds.events.next()

    @verbatim
    In [5]: evt.<TAB>
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

.. ipython:: python


    @verbatim
    evt.cspad.<TAB>
                evt.cspad.add        evt.cspad.epicsData  evt.cspad.L3T        evt.cspad.set_cmpars  
                evt.cspad.calib      evt.cspad.EventId    evt.cspad.monitor    evt.cspad.shape       
                evt.cspad.calibData  evt.cspad.Evr        evt.cspad.next       evt.cspad.show_all   >
                evt.cspad.configData evt.cspad.evtData    evt.cspad.psplots    evt.cspad.show_info   
                evt.cspad.detector   evt.cspad.image      evt.cspad.raw        evt.cspad.size        

    evt.cspad.show_info()

See example below of plotting an image with the matplotlib plotting library

.. plot:: examples/cspad.py
   :include-source:

Event Iteration
---------------

One can iterate events from the ds.events object, the evt object, or a detector object.  Each of these are python iterators.

The representation of each of these provides details of the event including the event codes present.

.. ipython:: python

    evt = ds.events.next()

    evt.next()

    evt.cspad.next()

    for evt in ds.events:
        print evt
        if ds._ievent == 6:
            break


Event Codes
-----------

To check if an event codes is present:

.. ipython:: python

    evt.Evr.present(40)

    evt.Evr.present(41)

Note that the default is to strictly check if code has same timestamp as the EventId.
Use strict=False keyword to check if code occurred with any timestamp since last event.

Beamline Data
-------------

The following Electron and Photon Beam data are generally recorded for all experiments.

.. ipython:: python

    evt.EBeam.show_info()
    
    evt.FEEGasDetEnergy.show_info()
    
    evt.PhaseCavity.show_info()

Epics Data
----------

The LCLS DAQ includes many slowly changing quantities (e.g. voltages, temperatures, motor positions) that are recorded with software called EPICS.  These can either be accessed by aliases or PV name.  

.. ipython:: python

    ds = PyDataSource.DataSource(exp='xpptut15',run=59)

    evt = ds.events.next() 

    @verbatim
    ds.epicsData.<TAB>
                  ds.epicsData.alias        ds.epicsData.ccm          ds.epicsData.CSPAD        ds.epicsData.filt         ds.epicsData.hrm1         ds.epicsData.ipm1          
                  ds.epicsData.aliases      ds.epicsData.ccmE         ds.epicsData.drift        ds.epicsData.FS3          ds.epicsData.hrm2         ds.epicsData.ipm1b         
                  ds.epicsData.alio         ds.epicsData.ccmTheta0    ds.epicsData.epicsConfig  ds.epicsData.getPV        ds.epicsData.HX2          ds.epicsData.ipm2         >
                  ds.epicsData.analogOut    ds.epicsData.ChipAddress  ds.epicsData.EVR          ds.epicsData.gon          ds.epicsData.HX3          ds.epicsData.ipm3          
                  ds.epicsData.Be           ds.epicsData.ChipName     ds.epicsData.FeeAtt       ds.epicsData.grid         ds.epicsData.ipm          ds.epicsData.ire           


    ds.epicsData.SampleTemp.show_info()

    ds.epicsData.SampleTemp.GetA.value

Epics data associated with detectors (using the convention of using an alias starting with the detector name) can be accessed either from the ds.epicsData object or more conveniently from the epicsData object in the detector.

.. ipython:: python

    ds.epicsData.yag2.show_info()

    evt.yag2.epicsData.show_info()

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

    In [21]: import PyDataSource

    In [22]: ds  = PyDataSource.DataSource(exp='xpptut15',run=200)

    In [23]: %time ds.configData.ScanData.show_info()
    CPU times: user 51.7 s, sys: 2.4 s, total: 54.1 s
    Wall time: 50.5 s
    Out[23]: 
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

  
Here is a quick way to iterate over steps:

.. sourcecode:: ipython

    In [24]:  for istep, stepevts in enumerate(ds.steps):
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

.. ipython:: python

    ds.reload()

    evt = ds.events.next()

    evt

    evt.next()


Jump Quickly to Specific Event
------------------------------

Time stamps can be used to jump to specific event in the next method.  Alternatively an interger event number can be supplied to goto event number in DataSource (although it may not be exactly same event depending on how the data_source string is corresponding keywords to define the data_source is defined and also may differ for fast feedback and offline analysis environments).

This example shows jumping to event 200, saving the EventTime, going back to event 0, and then using the saved EventTime to jump back to event 200.  

.. ipython:: python

    evt.next()

    evt.next(200)

    et200 = evt.EventId.EventTime

    evt.next(0)

    evt.next(et200)


If nothing is passed in the next method, then the next event goes to the next event before we started jumping to specific events.

.. ipython:: python

    evt.next()



