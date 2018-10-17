.. _apps:

.. currentmodule:: PyDataSource

Applications
************

idatasource
-----------

An ipython session can be started that automatically loads the specified experiement and run. 

e.g., 

.. code-block:: bash
    
    .  /reg/g/psdm/etc/psconda.sh
    idatasource --exp=xpptut15 --run=54


starts an ipython session with the DataSource automatically loaded...


.. sourcecode:: ipython

    ********************************************************************************

    Data loaded for the psana data_source =  exp=xpptut15:run=54:smd
    Total Events =  1219
    Load time =   1.0 sec


    ********************************************************************************
    ds is a python friendly (i.e, tab accessible) form of the psana.DataSource.
    Event data is accessible through aliases (psana get and keys methods are also preserved),  e.g.,

    In [1]: evt = ds.events.next()
    In [2]: evt.Evr.eventCodes
    Out[2]: [140, 141, 41, 40]

    The current event is also available in the ds.events object, e.g.,

    In [3]: ds.events.current.Evr.eventCodes
    Out[3]: [140, 141, 41, 40]

    For offline data (using smd small data access), one can also iterate through "calib cycle" steps.  e.g.,
    In [4]: for events in ds.steps:
                for evt in events:
                    # Do something with events...

    ********************************************************************************
    < DataSource: exp=xpptut15:run=54:smd 1219 events >
    < DataSource: exp=xpptut15:run=54:smd 1219 events >

    In [1]: ds.configData
    Out[1]: < ConfigData: exp=xpptut15:run=54:smd >
    ***Detectors in group 0 are "BLD" data recorded at 120 Hz on event code 40
    ***Detectors listed as Controls are controls devices with unknown event code (but likely 40).

    Alias                Grp   Description Code Pol   Delay [s]   Width [s] Source                        
    ------------------------------------------------------------------------------------------------------
    EBeam                  0        120 Hz   40                             BldInfo(EBeam)                
    FEEGasDetEnergy        0        120 Hz   40                             BldInfo(FEEGasDetEnergy)      
    PhaseCavity            0        120 Hz   40                             BldInfo(PhaseCavity)          
    XppEnds_Ipm0           0        120 Hz   40                             BldInfo(XppEnds_Ipm0)         
    XppSb2_Ipm             0        120 Hz   40                             BldInfo(XppSb2_Ipm)           
    XppSb3_Ipm             0        120 Hz   40                             BldInfo(XppSb3_Ipm)           
    cspad                  1        120 Hz   40 Pos 0.000549832 0.000010000 DetInfo(XppGon.0:Cspad.0)     
    yag2                   1        120 Hz   40 Pos 0.000690739 0.000300000 DetInfo(XppSb3Pim.1:Tm6740.1) 
    yag_lom                1        120 Hz   40 Pos 0.000690739 0.000300000 DetInfo(XppMonPim.1:Tm6740.1) 
    < ConfigData: exp=xpptut15:run=54:smd >


On a daq monitoring node (e.g., daq-cxi-mon01) that is hosting a psana shared memory process, idatasource can be executed
without exp and run keywords.

idatasource help
----------------

.. code-block:: bash

    datasource --help
    usage: idatasource [-h] [-e EXP] [-r RUN] [-i INSTRUMENT] [-s STATION] [--idx]
                       [--smd] [--h5] [--xtc_dir XTC_DIR] [--ffb] [--show_errors]
                       [--indexed] [--base BASE] [--shmem]
                       [data_source]

    positional arguments:
      data_source           psana data_source

    optional arguments:
      -h, --help            show this help message and exit
      -e EXP, --exp EXP     Experiment number
      -r RUN, --run RUN     Run number
      -i INSTRUMENT, --instrument INSTRUMENT
                            Instrument
      -s STATION, --station STATION
                            Station
      --idx                 Load indexed XTC data
      --smd                 Load smd small XTC data
      --h5                  Use hdf5 data instead of xtc
      --xtc_dir XTC_DIR     xtc file directory
      --ffb                 Use FFB data
      --show_errors         Show Errors in cases that might not be explicit due to
                            try/except statements
      --indexed             Use indexing, see: https://confluence.slac.stanford.ed
                            u/display/PSDM/psana+-+Python+Script+Analysis+Manual
                            #psana-PythonScriptAnalysisManual-
                            RandomAccesstoXTCFiles("Indexing")
      --base BASE           Base into which DataSource object is initiated.
      --shmem               Use shmem data stream



liveplot with shared memory
---------------------------

To open all area detector plots on a shared memory node use the liveplot application,
which loads the DataSource and opens area detector plots with the add.psplot method.

e.g.,

.. code-block:: bash
    
    .  /reg/g/psdm/etc/psconda.sh

    liveplot
    
    No exp provided -- cxilu1817 is the active experiment
    No data source specified, so assume this is shared memory.
    setting calibDir cxilu1817 /reg/d/psdm/cxi/cxilu1817/calib
    Opened queue /PdsFromMonitorDiscovery_psana (17)
    Opening shared memory /PdsMonitorSharedMemory_psana of size 0x4c800000 (0x22 * 0x2400000)
    Shared memory at 0x7f9b53800000
    Opened queue /PdsToMonitorEvQueue_psana_0 (18)
    Opened queue /PdsToMonitorEvQueue_psana_1 (19)
    ...
    *** Misc warnings may appear here that can be ignored ***
    ...
    --------------------------------------------------------------------------
    psmon plot added -- use the following to view: 
    --> psplot -s daq-cxi-mon05 -p 12301 Timetool_image
    WARNING -- see notice when adding for -p PORT specification
               if default PORT=12301 not available


Note that you may see MPI and shared object file warnings that can be ignored


liveplot offline
----------------

Liveplot can also be used offliine data on a psana machine.  

In this mode, one can interactively step through events as well as reload run when it is finished.


liveplot help
-------------

.. code-block:: bash

    liveplot --help
    usage: liveplot [-h] [-e EXP] [-r RUN] [--det DET] [-n NEVENTS]
                    [-i INSTRUMENT] [-s STATION] [--idx] [--smd] [--h5]
                    [--xtc_dir XTC_DIR] [--ffb] [--indexed] [--base BASE]
                    [--shmem]

    optional arguments:
      -h, --help            show this help message and exit
      -e EXP, --exp EXP     Experiment number
      -r RUN, --run RUN     Run number
      --det DET             Cmma separated detector alias names used for plot
                            Input method
      -n NEVENTS, --nevents NEVENTS
                            Number of Events to analyze
      -i INSTRUMENT, --instrument INSTRUMENT
                            Instrument
      -s STATION, --station STATION
                            Station
      --idx                 Load indexed XTC data
      --smd                 Load smd small XTC data
      --h5                  Use hdf5 data instead of xtc
      --xtc_dir XTC_DIR     xtc file directory
      --ffb                 Use FFB data
      --indexed             Use indexing, see: https://confluence.slac.stanford.ed
                            u/display/PSDM/psana+-+Python+Script+Analysis+Manual
                            #psana-PythonScriptAnalysisManual-
                            RandomAccesstoXTCFiles("Indexing")
      --base BASE           Base into which DataSource object is initiated.
      --shmem               Use shmem data stream




