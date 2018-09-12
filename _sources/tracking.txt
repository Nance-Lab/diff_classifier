.. _tracking-label:

Tracking particles
==================

The "track" function in diff_classifier operates by creating a temporary script
files in the TEMP directory and generating a command sent through the shell
to ImageJ.  Tracking is implemented using the TrackMate ImageJ plugin. User
inputs include tracking parameters and spot and track filter cutoffs.

This implementation is fairly constrained from the complete number of parameters
that Trackmate offers, but will be expanded in the future.  The parameters that
are currently implemented are:

* radius: estimated radius of spots in videos.  In general, should be slightly
  larger than the average particle size in the videos.  Note that the GUI
  interface uses diameter rather than radius.
* threshold: not currently in use.
* do_median_filtering: If set to True, filters the image before performing
  tracking to minimize noise.
* quality: Lower threshold on spot quality filter. Usually varies anywhere between
  1 to 300.
* x: Upper threshold on x coordinate spot filter.
* y: Upper threshold on y coordinate spot filter.
* ylo: Lower threshold on y coordinate spot filter.
* linking_max_distance: max distance in pixels that a particle can travel between
  frames.
* gap_closing_max_distance: max distance in pixels that a particle can travel when
  it skips a frame.
* track_displacement: meant to be duration.  Minimum number of frames a trajectory
  must have to be included in the final dataset.

The algorithm is set to only implement the difference of Gaussians detection
algorithm and the simple LAP tracker algorithm.

.. code-block:: python

  track(target, out_file, template=None, fiji_bin=None, radius=2.5, threshold=5.,
        do_median_filtering=False, quality=30.0, x=511, y=511, ylo=1,
        median_intensity=55000.0, snr=0.0, linking_max_distance=10.0,
        gap_closing_max_distance=10.0, max_frame_gap=3,
        track_displacement=0.0)

Selecting trajectory parameters
-------------------------------

One difficulty with particle tracking technology is that there is no easy way to
determine the "correct" tracking parameters for different videos.  Even videos
collected in the same conditions using the same particles can have different
optimal tracking parameters.  Parameters can also vary from user to user.  How
does one select optimal parameters, especially where tracking large numbers of
videos?

If users choose to use a single set of parameters for all videos, users risk
outputting poor tracking results, especially if there is uneven illumination
or different particle distributions with the video frame. This works well for
relatively homogeneous videos with similar particle sizes and uniform
illumination.

I have implemented an intermediate approach in diff_classifier that uses
intensity data from input videos to predict quality cutoff parameters. This
assumes that particles in all videos have similar radii and are moving at
similar velocities, while adjusting the quality threshold filter. The function
imagej.regress_sys allows users to create a regression object that predicts
quality parameters from an input training dataset. Users must manually track
a small number of randomly selected videos from their entire video collection
and use the quality values as inputs to regress_sys e.g.

.. code-block:: python

  prefix = 'test'
  nvideos = 20
  all_videos = []
  for num in range(nvideos):
      all_videos.append('{}_{}'.format(prefix, num))
  yfit = []
  training_size = 4

  ij.regress_sys('.', all_videos, yfit, training_size, have_output=False)

When have_output is set to False, regress_sys will output a list of randomly
selected videos contained in all_videos which the user must manually track.
The quality values are loaded into the variable yfit to produce the regression
object using scikit-learn:

.. code-block:: python

  yfit = [20.4, 17.9, 10.3, 30.4]
  regress = ij.regress_sys('.', all_videos, yfit, training_size,
                           have_output=False)

An example of such an implementation can be found
`here <https://github.com/ccurtis7/diff_classifier/blob/master/notebooks/03_07_18_knn_implementation.ipynb>`_.
