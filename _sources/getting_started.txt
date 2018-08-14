.. _getting-started-label:

Getting started with diff_classifier
====================================

Pip install
-----------
Currently, diff_classifier is not on PyPi, but the next version of diff_classifer
will be uploaded to PyPi.

Git clone
---------
Users can clone a copy of diff_classifier with the command

.. code-block:: python

  git clone https://github.com/ccurtis7/diff_classifier.git
  
Running the setup file will install needed dependencies, including Fiji:

.. code-block:: python

  python3 setup.py develop
  

You can install diff_classifier from the `Github repository
<https://github.com/ccurtis7/diff_classifier>`_.  This will install
diff_classifier and its Python dependencies. Users will also need to have
downloaded `Fiji <https://imagej.net/Fiji/Downloads>`_ in order to implement the
Trackmate functionality.

Checking Fiji installation
--------------------------
The tracking portion of diff_classifier relies on the Fiji plugin `Trackmate
<https://imagej.net/TrackMate>`_. The setup file should install Fiji automatically
with the Python plugin `fijibin <https://pypi.org/project/fijibin/>`_. By default,
this is usually installed in the directory

.. code-block::

  /name/of/home/directory/.bin
  
If needed, you can install Fiji manually `here <https://fiji.sc/#download>`_.

Note: Python2/Python3 compatability
-----------------------------------

The current diff_classifier release behaves a little funky. Most of the code
is optimized for Python 3 and isn't guaranteed to work in Python 2. However, due
to some undesirable behavior of the Cloudknot dependency in Python3, any jobs
submitted to AWS with a cloudknot.knot object must be done using Python3. In
essence, diff_classifier should be run locally using Python3, ec2 instances
called upon in AWS Batch should use Python3, but Cloudknot submissions through
an iPython notebook should be run using Python2. See example notebooks
below for more guidance.

Examples
--------
You can find example implementations of diff_classifier in the `notebooks
directory <https://github.com/ccurtis7/diff_classifier/tree/master/notebooks>`_
on Github.
