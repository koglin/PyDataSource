.. _exp_summary:

.. currentmodule:: PyDataSource 

Experiment Summary
******************

The ExperimentSummary class is used to load and manage archived epics information.  
It is available from the DataSource object and used in making hdf5 summary files
to automatically add epics PVs that were moved/scanned during run and epics PVs that were set
before the start of a run.
A report can be made with the to-html method.

.. sourcecode:: ipython

    In [1]: ds.exp_summary.
                         ds.exp_summary.add_user_run_table     ds.exp_summary.dfset                  ds.exp_summary.get_moved               
                         ds.exp_summary.calibration_runs       ds.exp_summary.exp                    ds.exp_summary.get_run_sets            
                         ds.exp_summary.detectors              ds.exp_summary.exp_dir                ds.exp_summary.get_scan_data          >
                         ds.exp_summary.dfruns                 ds.exp_summary.exper_id               ds.exp_summary.get_scan_series         
                         ds.exp_summary.dfscan                 ds.exp_summary.get_epics              ds.exp_summary.get_scans                

Note that experiment summary information in not available for the tutorial data since the runs come from many different experiments. 


Experiment Summary
------------------
.. autosummary::
    :toctree: generated/

    exp_summary


Experiment Summary API
----------------------
.. autosummary::
    :toctree: generated/

    ExperimentSummary
    ExperimentSummary.to_html


