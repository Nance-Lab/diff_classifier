Welcome to diff_classifier
===========================================

.. automodule:: diff_classifier

The diff_classifier package is complete particle tracking package implemented
using the ImageJ plugin `Trackmate <http://imagej.net/Getting_started_with_TrackMate>`_.

Usage
-----

.. code-block:: python

  import diff_classifier.utils as ut
  import diff_classifier.msd as msd
  import diff_classifier.features as ft
  import diff_classifier.imagej as ij
  import diff_classifier.heatmaps as hm

  prefix = 'test_video'
  frames = 651
  local_im = prefix + '.tif' # Name of image file
  outfile = 'Traj' + local_im.split('.')[0] + '.csv'
  msd_file = 'msd_{}.csv'.format(prefix)
  ft_file = 'features_{}.csv'.format(prefix)

  ij.track(local_im, outfile, template=None, fiji_bin=None, radius=4.5, threshold=0.,
            do_median_filtering=True, quality=4.5, x=511, y=y, ylo=1, median_intensity=300.0, snr=0.0,
            linking_max_distance=8.0, gap_closing_max_distance=10.0, max_frame_gap=2,
            track_displacement=10.0)

  df = ut.csv_to_pd(outfile)
  msds = msd.all_msds2(df, frames=frames)
  features = ft.calculate_features(msds)

  hm.plot_trajectories(prefix)

Motivation
----------

Multi-particle tracking software packages abound (see for example this `Nature
methods paper <http://www.nature.com/articles/nmeth.2808>`_ comparing the
results of 14 different teams).  But researchers are often on their own when
it comes to scale up, analysis, and visualization.  Diff_classifier seeks to
provide these tools in a centralized package, including MSD and trajectory
feature analysis tools, MSD and heatmap plots of output data, and
parallelization tools implemented using Amazon Web Services.  This package
is the primary tool for tracking analysis of nanoparticles in the brain in the
`Nance research group <https://www.nancelab.com/>`_ at the University of
Washington.

.. figure:: _static/summary.png
  :align: center

  Sample output from diff_classifier visualization tools.

Installation and getting started
--------------------------------

To install diff_classifier and begin analyzing your data, visit :ref:
`getting-started-label`.

Particle tracking
-----------------

For instructions scripting Trackmate for particle tracking analysis, visit
`Scripting Trackmate <https://imagej.net/Scripting_TrackMate>`_ as well as the
instructions using the diff_classifier pre-built functions
(:ref: `tracking-label`).

Features analysis
-----------------

Trajectory features calculations are based off the
`TrajClassifier <https://imagej.net/TraJClassifier>`_ package developed by
Thorsten Wagner.  The calculations can be found at the
`Traj wiki <https://github.com/thorstenwagner/TraJ/wiki#features>`_.
Instructions using the diff_classifier implementation can be found at
:ref: `features-analysis-label`.

Interacting with s3
-------------------

Diff_classifier provides functions for interacting with buckets on AWS S3.
Instructions on implementing uploading to/downloading from s3 can be found at
:ref: `interacting-with-s3-label`.

Cloudknot parallelization
-------------------------

Diff_classifier includes `Cloudknot <https://github.com/richford/cloudknot>`_
parallelization functions for complete tracking, analysis, and visualization
of large tracking experiments.  In general, these are only templates, and will
have to be modified by the user for their own experimental implementations.
Instructions can be found at :ref: 'cloudknot-parallelization-label'.

Bugs and issues
---------------

If you are having issues, please let us know by
`opening a new issue <https://github.com/ccurtis7/diff_classifier/issues>`_.
Please tag your issues with the "bug" or "question" label.

License
-------

This project is licensed under the
`MIT License <https://github.com/ccurtis7/diff_classifier/blob/master/LICENSE>`_.

Acknowledgements
----------------
Diff_classifier development is supported by ....

.. toctree::
  :maxdepth: 2
  :caption: User Documentation

  getting_started
  tracking
  features_analysis
  interacting_with_s3
  cloudknot_parallelization
  api/index
  examples <https://github.com/ccurtis7/diff_classifier/tree/master/notebooks>
  code <https://github.com/ccurtis7/diff_classifier>
  bugs <https://github.com/ccurtis7/diff_classifier/issues>
