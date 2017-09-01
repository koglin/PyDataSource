.. _conda:

.. currentmodule:: PyDataSource

.. _conda_release_system: https://confluence.slac.stanford.edu/display/PSDMInternal/Conda+Release+System

Conda Release System
********************

For more details on the LCLS conda release system for psana, see:

https://confluence.slac.stanford.edu/display/PSDMInternal/Conda+Release+System

Analysis Environment
--------------------

Below is an example of setting of a psana conda release where sepcific versions of packages can be chosen.  In this case below, you can work with the latest PyDataSource version, which may not yet be in the current conda ana release.

.. code-block:: bash 

    ssh psbuild-rhel7

    source /reg/g/psdm/bin/conda_setup
   
    kinit

    cd /reg/d/psdm/cxi/cxitut13/res/

    condarel --newrel --name conda

    cd conda

    condarel --addpkg --name PyDataSource --tag HEAD

    scons




