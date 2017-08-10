.. _api:

.. currentmodule:: PyDataSource

PyDatSource API
===============

Data Source
-----------

.. autosummary::
    :toctree: generated/

    DataSource
    SourceData
    ScanData
    ConfigSources
    ConfigData
    EpicsConfig

Data Iterators
--------------

.. autosummary::
    :toctree: generated/
    
    Runs
    Steps
    Run
    RunEvents
    SmdEvents
    StepEvents
    Events

Detector Access
---------------

.. autosummary::
    :toctree: generated/

    EvtDetectors
    Detector
    AddOn

Event Info
----------

.. autosummary::
    :toctree: generated/

    EvrData
    EvrNullData
    EventId
    L3Ttrue
    L3Tdata

psana.Detector Access
---------------------

.. autosummary::
    :toctree: generated/
    
    ImageData
    ImageCalibData
    WaveformData
    WaveformCalibData
    IpimbData

Epics Data
----------

.. autosummary::
    :toctree: generated/

    EpicsData
    PvData
    EpicsStorePV
    TimeStamp

Low-level Psana Data Interface
------------------------------

.. autosummary::
    :toctree: generated/
    
    PsanaTypeList
    PsanaTypeData
    PsanaSrcData

Data Structures
---------------

.. autosummary::
    :toctree: generated/
    
    DataSource.to_hdf5
    to_summary
    to_h5netcdf 
    open_h5netcdf 

DataSource Configuration
------------------------

.. autosummary::
    :toctree: generated/
    
    DataSource.xarray_kwargs
    DataSource.save_config
    DataSource.load_config


