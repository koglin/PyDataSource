.. _interactive_plotting:

Interactive Plotting
********************

Previous plotting examles have used the standard python matplotlib.  

A more interactive form of plotting that can also be used for real-time monitoring is available using the psana psmon package (based on PyQtGraph).  These plots can be viewed on multiple other machines as well as locally (using the zmq protocal).

This example shows how to start plotting a CsPad140k image.  

.. sourcecode:: ipython

    In [1]: import PyDataSource

    In [2]: ds  = PyDataSource.DataSource(exp='xpptut15',run=200)

    In [3]: evt = ds.events.next()

    In [4]: evt.cs140_rob.add.psplot('image')
      psmon plot added -- use the following to view: 
      --> psplot -s psanaphi108 -p 12301 cs140_rob_image
      WARNING -- see notice when adding for -p PORT specification
                 if default PORT=12301 not available

.. figure::  images/xpptut15_run200_cs140_rob.jpg
   :align:   center

In the separate window that pops up with the image, you can make adjustments including:  
 - adjusted the size by dragging a window corner.  
 - adjust color bar scale and range on the right.  
 - move and zoom in/out axes (reset by clicking the small boxed A at the bottom left of the plot).

.. figure::  images/xpptut15_run200_cs140_rob_zoom.jpg
   :align:   center

The plot will update on each next event where the detector is present.  The monitor method can be used to update events continuously until Ctrl-C is entered.

.. sourcecode:: ipython

    In [5]: evt.cs140_rob.next()
    Out[5]: < EvtDetectors: xpptut15, Run 200, Step 0, Event 1, 07:17:55.8392, [50, 51, 60, 61, 90, 40, 140] >

    In [6]: evt.cs140_rob.monitor()
    --------------------------------------------------------------------------------
    cs140_rob xpptut15, Run 200, Step 0, Event 2, 07:17:55.8476, [50, 60, 141, 142, 90, 40, 41, 42, 140]
    --------------------------------------------------------------------------------
    calib                  <0.1046> ADU     Calibrated data
    image                 <0.09625> ADU     Reconstruced 2D image from calibStore geometry
    raw                 <1.678e+03> ADU     Raw data
    shape              [  2 185 388]         Shape of raw data array
    size                     143560         Total size of raw data
    --------------------------------------------------------------------------------
    cs140_rob xpptut15, Run 200, Step 0, Event 3, 07:17:55.8559, [50, 51, 52, 60, 61, 62, 90, 40, 140]
    --------------------------------------------------------------------------------
    calib                <-0.01223> ADU     Calibrated data
    image                <-0.01132> ADU     Reconstruced 2D image from calibStore geometry
    raw                 <1.677e+03> ADU     Raw data
    shape              [  2 185 388]         Shape of raw data array
    size                     143560         Total size of raw data

Note:  currently sometimes Ctrl-C interupts in the remote plot and throws an error.  If this happens simply close the plot window and it will automatically reopen on the next event.  A fix to the underlying psmon code is being worked on to better handle the Ctrl-C. 


