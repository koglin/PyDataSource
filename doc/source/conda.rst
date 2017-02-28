.. _conda:

.. currentmodule:: PyDataSource

Conda Release System
********************

For more details on the LCLS conda release system for psana, see:
https://confluence.slac.stanford.edu/display/PSDMInternal/Conda+Release+System

Analysis Environment
--------------------

Below is an example of setting of a psana conda release where sepcific versions of packages can be chosen.  In this case below, you can work with the latest PyDataSource version, which may not yet be in the current conda ana release.

.. code-block:: bash 

    .  /reg/g/psdm/etc/ana_env.sh
    source eonda_setup
    
    cd reg/d/psdm/xpp/xpptut15/results/

    condarel --newrel --name conda

    cd conda

    condarel --addpkg --name PyDataSource --tag HEAD

    source conda_setup

    scons




