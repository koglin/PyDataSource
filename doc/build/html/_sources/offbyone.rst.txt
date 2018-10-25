.. _offbyone:

.. currentmodule:: PyDataSource

.. _ARP: https://confluence.slac.stanford.edu/display/PSDM/Automatic+Run+Processing
.. _Off-by-one: https://confluence.slac.stanford.edu/display/PSDM/Off-by-one+Timing+Error+Detection

Off-by-one_ Timing Error Analysis
*********************************

A script for automatically detecting timing errors has been created in the psana PyDataSource package.
The batch job submission script for use in the Automatic Run Processor (or ARP_) web service is located at:

/reg/g/psdm/utils/arp/submit_offbyone

The ARP script is automatically configured for each new experiment in the experiment 
Workflow / Batch Processing / Definitions tab of the Experiment Data Manager.  e.g.,

.. figure::  images/offbyone_arp_config.png
   :align:   center

Optional parameters are the batch queue followed by a list of people to be alerted by e-mail if an off-by-one timing error is detected.

With the ARP Autorun enabled, this script will be automatically submitted for every run.  When the script is finished it updates the Report column in the Workflow / Batch Processing / Control tab

.. figure::  images/offbyone_arp_config.png
   :align:   center

In the report, there are links to more detailed html reports (open with middle-click / Open link in new tab), which are also available in the Workflow / Data Summary tab.


Off-by-one Methods
------------------

The submit_offbyone application loads the run 'beam stats' information into an xarray with the get_beam_stats method

.. autosummary::
    :toctree: generated/

    beam_stats.get_beam_stats

This method first loads all the small data for the run.  
The following highlights the procedure and methods used.

.. sourcecode:: ipython

    In [1]: import PyDataSource
    
    In [2]: from PyDataSource.beam_stats import load_small_xarray

    In [3]: ds = PyDataSource.DataSource(exp='sxrx22615', run=7)

    In [4]: xsmd = load_small_xarray(ds)

    In [5]: from PyDataSource.xarray_utils import set_delta_beam

    In [6]: set_delta_beam(xsmd, code='ec162', attr='delta_drop')


where load_small_xarray simply calls the make_small_xarray method unless a file already exists
(and the keyword refresh is not set to False), 
in which case it is loaded from an xarray compatible hdf5 file.

.. autosummary::
    :toctree: generated/

    beam_stats.load_small_xarray
    beam_stats.make_small_xarray

Instrument specific transmission information is then added according to the
beam_stats._transmission_pvs dictionary.

Dropped Shots
-------------

Then the xarray_utils.set_delta_beam method is used to 
determint the number of beam codes to the nearest drop_code, which by default is the BYKIK eventCode 162.
i.e., delta_drop = -1 for the event preceeding the BYKIK dropped shot event code 162 and
delta_drop = 1 for the event just after the dropped shot.

.. autosummary::
    :toctree: generated/

    xarray_utils.set_delta_beam

Then there is some code to auto set number of nearest shots to include at least two on 
each side for rates as low as 10 Hz.
A gas detector cut, Gasdet_cut, is created in the xsmd xarray Dataset 
to tag low-beam events with the the appropriate gas detector
threshold automatically chosen to be 1-sigma above the 95% range of the gas detector values
when there are dropped shots.  
The BLD 'FEEGasDetEnergy_f_11_ENRC' value for the gas detector 
upstream of the gas attenuator is used.

From this combined with the dropped shot event code, XrayOn and XrayOff events are tagged. 

If a sufficient number of dropped shots are in the run (default drop_min=3),
then the larger Area Detector and Waveform data will be processed for events 
nearest the dropped shot 
(at least one and up to five on each side of the dropped shot as previously
auto determined depending on rep rate being used).
Device specific methods are used to represent the Area detector and waveforms as scalars.


Area Detector Processing
------------------------

The summed ADU (analog to digital units) of pedestal corrected data is used 
for X-ray detectors like the CsPad.  
Note that mixed gain modes for the CsPad will do not yet have the gain factor applied.

Each of the two sectors in a Jungfrau 1M detector is processed separately
using the beam_stats.sectors method on the 'raw' Jungfrau data because they are independently
read out and each sector could independently become off in timing (something that occurred
in MFX in Run 16).  
Note that the Jungfrau auto gain switching (3 ranges) correction is not yet used here.

.. autosummary::
    :toctree: generated/

    beam_stats.rawsum
    beam_stats.count
    beam_stats.sector
    beam_stats.sectors


Waveform Detector Processing
----------------------------

The Acqiris waveforms processed independently with the peak_height and peak_time methods
to find the height and time of the most prominant peak in the time spectrum 
(either positive or negative).

.. autosummary::
    :toctree: generated/

    beam_stats.wave8_height
    beam_stats.peak_height
    beam_stats.peak_time
    beam_stats.amplitudes
    beam_stats.filtered


Beam Correlation Detection
--------------------------

.. autosummary::
    :toctree: generated/

    xarray_utils.find_beam_correlations
 

Beam Correlations
-----------------

Currently the pulse energy after the gas attenuator, 'FEEGasDetEnergy_f_21_ENRC',
is used for beam correlation analysis for all instruments, 
but in the future this could be updated to use an instrument specific combination of
beam monitors (i.e., wave8 detectors installed in Run 17 for all hard X-ray instruments)
and instrument attenuators.


Example
-------

The above is all executed as in the following example:

.. sourcecode:: ipython

    In [1]: from PyDataSource.beam_stats import get_beam_stats

    In [2]: xdrop = get_beam_stats(exp='sxrx22615', run=7)
    2018-10-10 15:36:41,737 - INFO - PyDataSource.beam_stats - PyDataSource.beam_stats
    [INFO    ] PyDataSource.beam_stats
    2018-10-10 15:36:56,467 - INFO - PyDataSource.beam_stats - Loading Scalar: ['EventId', 'FEEGasDetEnergy', 'PhaseCavity', 'EBeam', 'GMD']
    [INFO    ] Loading Scalar: ['EventId', 'FEEGasDetEnergy', 'PhaseCavity', 'EBeam', 'GMD']
    2018-10-10 15:36:56,467 - INFO - PyDataSource.beam_stats - Loading 1D: []
    [INFO    ] Loading 1D: []
    2018-10-10 15:37:00,289 - INFO - PyDataSource.beam_stats -      100 of   132481 --    9.082 sec,   11.011 events/sec
    [INFO    ]      100 of   132481 --    9.082 sec,   11.011 events/sec
    2018-10-10 15:37:00,461 - INFO - PyDataSource.beam_stats -      200 of   132481 --    9.254 sec,  579.999 events/sec
    [INFO    ]      200 of   132481 --    9.254 sec,  579.999 events/sec
    ...
    2018-10-10 16:05:41,327 - INFO - PyDataSource.beam_stats - Dropping unused eventCodes: [76, 75, 77, 80, 81]
    [INFO    ] Dropping unused eventCodes: [76, 75, 77, 80, 81]


.. autosummary::
    :toctree: generated/

    beam_stats.build_beam_stats


API
---

.. autosummary::
    :toctree: generated/

    beam_stats.load_small_xarray
    beam_stats.make_small_xarray



