.. _tracking-label:

Tracking particles with diff_classifier
====================================

The "track" function in diff_classifier operates by creating a temporary script
files in the TEMP directory and generating a command send through the shell
to ImageJ.  User inputs include tracking parameters and spot and track filter
cutoffs.

This implementation is fairly constrained from the complete number of parameters
that Trackmate offers, but will be expanded in the future.  The parameters that
are currently implemented are:

radius: estimated radius of spots in videos.  In general, should be slightly
  larger than the average particle size in the videos.  Note that the GUI
  interface uses diameter rather than radius.
threshold:
do_median_filtering: If set to True, filters the image before performing
  tracking to minimize noise.
quality: Lower threshold on spot quality filter. Usually varies anywhere between
  1 to 300.
x: Upper threshold on x coordinate spot filter.
y: Upper threshold on y coordinate spot filter.
ylo: Lower threshold on y coordinate spot filter.
linking_max_distance: max distance in pixels that a particle can travel between
  frames.
gap_closing_max_distance: max distance in pixels that a particle can travel when
  it skips a frame.
track_displacement: meant to be duration.  Minimum number of frames a trajectory
  must have to be included in the final dataset.

.. code-block:: python

  track(target, out_file, template=None, fiji_bin=None, radius=2.5, threshold=5.,
        do_median_filtering=False, quality=30.0, x=511, y=511, ylo=1, median_intensity=55000.0, snr=0.0,
        linking_max_distance=10.0, gap_closing_max_distance=10.0, max_frame_gap=3,
        track_displacement=0.0)

Selecting trajectory parameters
-------------------------------

One difficulty with particle tracking technology is that there is no easy way to
determine the "correct" tracking parameters for different videos.  Even videos
collected in the same conditions using the same particles can have different
optimal tracking parameters.  Parameters can also vary from user to user.  How
does one select optimal parameters, especially where tracking large numbers of
videos?

I am still working on this problem, but I have tried a few different things.
First, if you have a relatively homogeneous selection of videos with similar
particle sizes and uniform illumination, you can get away with using a single
set of parameters for all videos.

A second solution I have used in the situation is a set of two quality values
depending on whether the image is high-intensity or low-intensity.  I found a
fairly strong correlation between quality and mean intensity of the image, and
this has worked pretty well.  Using a simple if statement after calculating the
mean intensity of the first image can improve tracking significantly. I have
found that in general, the most sensitive parameters is the quality cutoff.  All
other parameters can remain constant.

Using regression to predict quality
-----------------------------------

The final solution that I have been toying with is using a regression technique
to predict tracking parameters based on image minimum, average, maximum, and
standard deviation intensities.  In order to implement this method, the user must
first build a training dataset (~20 videos) by using the Trackmate GUI to manually
find the best quality cutoffs for a random sample of videos from the entire set
of videos to analyze.  Then the scikit-learn toolbox can be used to select a
regression technique to predict the quality.

An example of such an implementation can be found
`here <https://github.com/ccurtis7/diff_classifier/blob/master/notebooks/03_07_18_knn_implementation.ipynb>`_.
