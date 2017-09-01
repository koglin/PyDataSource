.. _build_html:

.. currentmodule:: PyDataSource 

.. _practice: https://confluence.slac.stanford.edu/display/PSDM/Publicly+Available+Practice+Data

Run Summary Reports
*******************

HTML web page reports can be built from the xarray data summaries using the Build_html class in the build_html module.  An auto summary option is available to summarize the data in the xarray DataSet, grouping data according to Detector aliases.  To create a summary in the the RunSummary folder of your home directory organized in subfolders by experiment and run use the path='home' keyword (in this example ~/RunSummary/cxitut13/run0020/).  By default, the path is the 'stats/summary' subfolder in the experiment directory (in this example /reg/d/psdm/cxi/cxitut13/stats/summary/run0020/).  If setting a custom path, use the full path (i.e., /reg/neh/home/user_name/RunSummary, do not use ~user_name/RunSummary)    

.. sourcecode:: ipython

    In [1]: import PyDataSource

    In [2]: x = PyDataSource.open_h5netcdf(exp='cxitut13',run=20)

    In [3]: from PyDataSource.build_html import Build_html

    In [4]: b = Build_html(x, auto=True, path='home')

This run is an example of 8keV t-field data on DsaCsPad as noted in the practice_ data confluence page.

See link for the html report generated in this example for `cxitut13/run0020 <_static/reports/cxitut13/run0020/report.html>`_ 

Basic Report Structure
----------------------
The reports are organized into data catagories that can be contracted and revealed 
(through basic use of javascript).  
Each catagory contains one or more plot, table or text object, most of which will start off as 
contraceted boxes with title text by default.

Several layers of helper methods are used to build the html report. 

Some or all of the plots and tables generated with auto=True can be added as desired.  

Auto add default report
-----------------------
The auto=True keyword is a shortcut for the following:

.. sourcecode:: ipython

    In [5]: b = Build_html(x, auto=False, path='home')

    In [6]: b.add_all()

    In [7]: b.to_html()

Where the add_all calls the following methods for each detector base alias in the xarray DataSet.

.. autosummary::
    :toctree: generated/

    Build_html
    Build_html.add_all

Customizing reports
-------------------
Custom reports can be generated using the following high level methods 
for makine sets of plots based on detector alias and data attribute names.
Later the lower level methods for adding plots and tables will be described.

.. sourcecode:: ipython

    In [8]: b = Build_html(x, auto=False, path='home')

Add Detector
------------
By convention the psana detector alias is passed to each data element in the xarray object.
If no keywords are provided, the add_detector method will attempt to add all relevant plots for  
data elements with the detector alias, i.e., 'DsaCsPad' in this example.

.. sourcecode:: ipython

    In [9]: b.add_detector('DsaCsPad')

.. autosummary::
    :toctree: generated/

    Build_html.add_detector


Add Summary 
-----------
Add summary plots for data variables

.. sourcecode:: ipython

    In [10]: variables = ['DsaCsPad_photon_hist']
    In [11]:  b.add_summary(variables)

.. autosummary::
    :toctree: generated/

    Build_html.add_summary

Add Statistics
--------------
Add detector statistics plots for data variables that have 'stats' dims.

.. sourcecode:: ipython

    In [12]: b.add_stats('DsaCsPad_corr_stats')

.. autosummary::
    :toctree: generated/

    Build_html.add_stats

Other common plots
------------------

.. autosummary::
    :toctree: generated/

    Build_html.add_setup
    Build_html.add_config
    Build_html.add_detstats
    Build_html.add_correlations
    Build_html.add_counts
    Build_html.add_axis_plots


write html report
-----------------

.. sourcecode:: ipython

    In [13]: b.to_html()

.. autosummary::
    :toctree: generated/

    Build_html.to_html

Base level report methods
-------------------------

.. autosummary::
    :toctree: generated/

    Build_html.add_plot
    Build_html.add_table
    Build_html.add_xy_ploterr





