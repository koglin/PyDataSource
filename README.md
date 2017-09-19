# PyDataSource
The PyDataSource package provides access to LCLS event data by wrapping the psana “Detector” interface. 

## Documentation for current psana release
http://pswww.slac.stanford.edu/swdoc/ana/PyDataSource

## Overview
- PyDataSource python interface to all xtc data, including image and calibrated event data using psana "Detector" interface.
-- Detector specific plug-ins available for common detectors — also serve as template for custom detector processing
-- AddOn's available for making projections, roi, histogram, peak finding, plotting (i.e., wraps many of the examples from the psana confluence page).
-- Automatically associates epics info with relevant Detectors through aliases and keeps track of meta-data including detector config

- Structure H5 summary files created using netcdf4 convention for xarray compatibility (http://xarray.pydata.org)
-- Data Arrays with axes (i.e., coordinates) and meta-data (attributes like units, doc strings, detector config info) 
-- Based on pandas and conveniently wrapping statistical and plotting tools.

- Run summaries accessed from elog web portal
-- Plots and tables with instructions to make each plot and tables in html files 
-- Currently some smart data selection to make correlation plots, but desire to automatically pick out most relevant plots and identify systematic issues with data (e.g., ‘off-by-one’ events) 
-- Provides an overview of the data and jump start anyone on accessing the data in more depth.



