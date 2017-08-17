.. _batch:

.. currentmodule:: PyDataSource 

.. _conda: https://confluence.slac.stanford.edu/display/PSDMInternal/Conda+Release+System
.. _ARP: https://confluence.slac.stanford.edu/display/PSDM/Automatic+Run+Processing 

Submitting Batch Jobs
*********************

The run summary hdf5 files and html reports can be generated from batch jobs with the submit_summary app.

For details on setting up analysis environment see conda_ secton. 

The submit_summary bash script takes the experiment name and run number as arguments
followed by the batch queue and type of processing.  

For mpi processing (12 nodes):

.. code-block:: bash 

    .  /reg/g/psdm/etc/psconda.sh
    submit_summary cxitut13 20 psanaq mpi

For single core batch processing:

.. code-block:: bash 

    .  /reg/g/psdm/etc/psconda.sh
    submit_summary cxitut13 20 psanaq batch

Change the last parameter to 'epics' in order to create an updated experiment summary, 
which will be updated to include the most recent run independent of the specified run parameter.

.. code-block:: bash 

    submit_summary cxitut13 20 psanaq epics

The resulting reports are available under the Workflow top tab, and then the Data Summary tab to the lef.

Automatic Run Processing
------------------------

The arguments for submit_summary app are defined to be compatibility with Automatic Run Processing (see ARP_), 
a web service that allows for the easier submission of batch jobs through the experiment data manager (i.e., elog). 

To use the ARP_ for executing this command, follow the ARP_ instructions for setting up
the Batch Job Definition.  
Under the Experiment tab, the Batch defs tab enter the following for submitting default run summaries.

+-------------+------------------------------------------------------------------------------------+
| Hash:       | #submit_summary                                                                |
+-------------+------------------------------------------------------------------------------------+
| Executable: | /reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.3.12/bin/submit_summary |
+-------------+------------------------------------------------------------------------------------+
| Parameters: | psanaq                                                                             |
+-------------+------------------------------------------------------------------------------------+

For updating the experiment summary add 'epics' as a second parameter

+-------------+------------------------------------------------------------------------------------+
| Hash:       | #experiment_summary                                                                |
+-------------+------------------------------------------------------------------------------------+
| Executable: | /reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.3.12/bin/submit_summary |
+-------------+------------------------------------------------------------------------------------+
| Parameters: | psanaq epics                                                                       |
+-------------+------------------------------------------------------------------------------------+

Over on Run Tables tab, the Batch control tab is where this hash may be applied to experiment runs. 
In the drop-down menu of the Action column, any hash defined in Batch defs may be selected. 

For ARP submission of custom run summaries create a python file based on the following example and 
place it in the experiment results/src subfolder and add the name of the file as the second parameter.

e.g., /reg/d/psdm/cxi/cxitut13/results/src/cxitut13.py

+-------------+------------------------------------------------------------------------------------+
| Hash:       | #custom_summary                                                                    |
+-------------+------------------------------------------------------------------------------------+
| Executable: | /reg/g/psdm/sw/conda/inst/miniconda2-prod-rhel7/envs/ana-1.3.12/bin/submit_summary |
+-------------+------------------------------------------------------------------------------------+
| Parameters: | psanaq cxitut13                                                                    |
+-------------+------------------------------------------------------------------------------------+

Note for older experiments like cxitut13, the results folder in named 'res' instead of 'results'. 
The submit_summary summary will use the res folder when the results folder is not present.

:download:`Open cxitut13.py example <examples/cxitut13.py>`.

.. literalinclude:: examples/cxitut13.py

