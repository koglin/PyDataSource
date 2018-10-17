.. _conda:

.. currentmodule:: PyDataSource

.. _conda_release_system: https://confluence.slac.stanford.edu/display/PSDMInternal/Conda+Release+System

Updating Releases
*****************

For more details on the LCLS conda release system for psana, see:

https://confluence.slac.stanford.edu/display/PSDMInternal/Conda+Release+System

Analysis Environment
--------------------

Below is an example of setting of a psana conda release where sepcific versions of packages can be chosen.  In this case below, you can work with the latest PyDataSource version, which may not yet be in the current conda ana release.

.. code-block:: bash 

    ssh psbuild-rhel7

    unset PYTHONPATH
    unset LD_LIBRARY_PATH
    .  /reg/g/psdm/etc/psconda.sh
   
    kinit

    cd /reg/d/psdm/cxi/cxitut13/res/

    condarel --newrel --name conda

    cd conda

    condarel --addpkg --name PyDataSource --https --tag HEAD

    source conda_setup

    scons


To use this environment you need to be on a psana machine, which has access to data.

.. code-block:: bash 

    ssh psana
    
    # first cd to base path
    cd conda

    unset PYTHONPATH
    unset LD_LIBRARY_PATH
    .  /reg/g/psdm/etc/psconda.sh

    source conda_setup


If you chose to modify any ot the PyDataSource code then execute scons again.


Upgrade ana release
-------------------

To update the analysis release.

.. code-block:: bash

    conda activate ana-1.3.63
    condarel --chenv
    source conda_setup
    scons -c
    scons


Push package updates to github and add tag so that latest PyDataSource version will go in next ana release

.. code-block:: bash
    
    git push -u origin master

    git tag -a V00-06-04 -m 'Version V00-06-04'
    git push origin V00-06-04


Sphinx Documentation
--------------------

Within conda env, build sphinx documentation in package doc folder.

After updating any rst files in the source folder, use the 'make html' command to rebuild
the html sphinx documenation

.. code-block:: bash
    
    (ana-1.3.70) *tr* [koglin@psanaphi101 doc 10:47:54] pwd
    /reg/neh/home/koglin/conda/PyDataSource/doc
    
    (ana-1.3.70) *tr* [koglin@psanaphi101 doc 10:48:01] ls
    build  ChangeLog  make.bat  Makefile  source
    
    (ana-1.3.70) *tr* [koglin@psanaphi101 doc 10:48:03] ls source
    api.rst         conda.rst        data_processing.rst  exp_summary.rst  offbyone.rst  _static
    apps.rst        config_data.rst  data_summary.rst     generated        pyplots       _templates
    batch.rst       conf.py          examples             images           savefig       xarray.rst
    build_html.rst  data_access.rst  expert.rst           index.rst        sphinxext

    (ana-1.3.70) *tr* [koglin@psanaphi101 doc 10:48:05] make html


To make html documenation availible on through the pswww web service, 
copy  PyDataSource/doc/source/build/html folder to the swdoc releases folder 
for PyDataSource (requires sudo as psreldev). e.g.,

.. code-block:: bash

    cd /reg/g/psdm/sw/conda/web/PyDataSource-tags/ 

    cp -r ~koglin/conda/PyDataSource/doc/build/html PyDataSource-V00-06-04

    ls -l
    total 14
    drwxr-xr-x 10 psreldev xs 29 Aug 16  2017 PyDataSource-V00-02-03
    drwxr-xr-x 10 psreldev xs 27 Aug 17  2017 PyDataSource-V00-02-04
    drwxr-xr-x  9 psreldev xs 25 Sep  1  2017 PyDataSource-V00-02-06
    drwxr-xr-x  9 psreldev xs 30 Oct 17 12:02 PyDataSource-V00-06-04



Then make soft link to web/ana/PyDataSource.

.. code-block:: bash
    
    cd /reg/g/psdm/sw/conda/web/ana

    rm PyDataSource

    ln -s ../PyDataSource-tags/PyDataSource-V00-06-04 PyDataSource

    ls /reg/g/psdm/sw/conda/web/ana/PyDataSource -l
    lrwxrwxrwx 1 psreldev xs 43 Oct 17 12:05 /reg/g/psdm/sw/conda/web/ana/PyDataSource -> ../PyDataSource-tags/PyDataSource-V00-06-04
    
    ls /reg/g/psdm/sw/conda/web/ana/PyDataSource 
    api.html         config_data.html      examples          _images        py-modindex.html  xarray.html
    apps.html        data_access.html      expert.html       index.html     search.html
    batch.html       data_processing.html  exp_summary.html  _modules       searchindex.js
    build_html.html  data_summary.html     generated         objects.inv    _sources
    conda.html       _downloads            genindex.html     offbyone.html  _static


Then the html documentation will show up in:
    
    https://pswww.slac.stanford.edu/swdoc/ana/PyDataSource




