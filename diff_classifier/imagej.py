"""Tools to perform particle tracking with Trackmate ImageJ plugin.

This module includes functions for prepping images and performing tracking
using the TrackMate ImageJ plugin [1]_.

References
----------
.. [1] Tinevez, JY,; Perry, N. & Schindelin, J. et al. (2016), "TrackMate: an
   open and extensible platoform for single-particle tracking.", Methods 115:
   80-90, PMID 27713081 (on Google Scholar).

"""

import sys
import subprocess
import tempfile
import random

import os.path as op
import numpy as np
import skimage.io as sio

import diff_classifier as dc
import diff_classifier.aws as aws

from sklearn import linear_model
from sklearn import svm


def partition_im(tiffname, irows=4, icols=4, ores=(2048, 2048),
                 ires=(512, 512)):
    """Partitions image into smaller images.

    Partitions a large image into irows x icols images of size ires and saves
    them. Also forces image to be square. Default input image sizes are from
    a Nikon/Hamamatsu camera setup (2048 x 2044 pixels).

    Parameters
    ----------
    tiffname : string
        Location of input image to be partitioned.
    irows : int
        Number of rows of size ires pixels to be partitioned from source image.
    icols : int
        Number of columns of size ires pixels to be partitioned from source image.
    ores : tuple of int
        Input images are scaled to size ores pixels prior to splitting.
    ires : tuple of int
        Output images are of size ires pixels.

    Examples
    --------
    >>> partition_im('your/sample/image.tif', irows=8, icols=8, ires=(256, 256))

    """
    test = sio.imread(tiffname)
    oshape = test.shape
    test2 = np.zeros((oshape[0], ores[0], ores[1]), dtype=test.dtype)
    test2[0:oshape[0], 0:oshape[1], :] = test

    new_image = np.zeros((oshape[0], ires[0], ires[1]), dtype=test.dtype)
    names = []

    for row in range(irows):
        for col in range(icols):
            new_image = test2[:, row*ires[0]:(row+1)*ires[0],
                              col*ires[1]:(col+1)*ires[1]]
            current = tiffname.split('.tif')[0] + '_%s_%s.tif' % (row, col)
            sio.imsave(current, new_image)
            names.append(current)

    return names


def mean_intensity(local_im, frame=0):
    """Calculates mean intensity of first frame of input image.

    Parameters
    ----------
    local_im : string
        Location of input image.
    frame : int
        Frame at which to perform mean.

    Returns
    -------
    test_intensity : float
        Mean intensity of input image.

    Examples
    --------
    >>> mean_intensity('your/sample/image')

    """
    test_image = sio.imread(local_im)
    test_intensity = np.mean(test_image[frame, :, :])

    return test_intensity


def track(target, out_file, template=None, fiji_bin=None,
          tparams={'radius': 3.0, 'threshold': 0.0, 'do_median_filtering': False,
           'quality': 15.0, 'xdims': (0, 511), 'ydims': (1, 511),
           'median_intensity': 300.0, 'snr': 0.0, 'linking_max_distance': 6.0,
           'gap_closing_max_distance': 10.0, 'max_frame_gap': 3,
           'track_duration': 20.0}):
    """Performs particle tracking on input video.

    Particle tracking is performed with the ImageJ plugin Trackmate. Outputs
    a csv file containing analysis settings and particle trajectories.

    Parameters
    ----------
    target : str
        Full path to a tif file to do tracking on. Can also be a URL
        (e.g., 'http://fiji.sc/samples/FakeTracks.tif')
    out_file : str
        Full path to a csv file to store the results.
    template : str, optional
        The full path of a template for tracking. Defaults to use
        `data/trackmate_template.py` stored in the diff_classifier source-code.
    fiji_bin : str
        The full path to ImageJ executable file. Includes default search
        locations for Mac and Linux systems.
    radius : float
        Estimated radius of particles in image.
    threshold : float
        Threshold value for particle detection step.
    do_median_filtering : bool
        If True, performs a median filter on video prior to tracking.
    quality : float
        Lower quality cutoff value for particle filtering.
    xdims : tuple of int
        Upper and lower x limits for particle filtering.
    ydims : tuple of int
        Upper and lower y limits for particle filtering.
    median_intensity : float
        Lower median intensity cutoff value for particle filtering.
    snr : float
        Lower signal to noise ratio cutoff value for particle filtering.
    limking_max_distance : float
        Maximum allowable distance in pixels between two frames to join
        particles in track.
    gap_closing_max_distance : float
        Maximum allowable distance in pixels between more than two frames to
        join particles in track.
    max_frame_gap : int
        Maximum allowable number of frames a particle is allowed to leave video
        and be counted as same trajectory.
    track_duration : float
        Lower duration cutoff in frames for trajectory filtering.

    """
    if template is None:
        template = op.join(op.split(dc.__file__)[0],
                           'data',
                           'trackmate_template3.py')

    if fiji_bin is None:
        if sys.platform == "darwin":
            fiji_bin = op.join(
                '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx')
        elif sys.platform.startswith("linux"):
            fiji_bin = op.join(op.expanduser('~'), 'Fiji.app/ImageJ-linux64')

    script = ''.join(open(template).readlines())
    tpfile = tempfile.NamedTemporaryFile(suffix=".py")
    fid = open(tpfile.name, 'w')
    fid.write(script.format(target_file=target, radius=str(tparams['radius']),
                            threshold=str(tparams['threshold']),
                            do_median_filtering=str(tparams['do_median_filtering']),
                            quality=str(tparams['quality']),
                            xd=str(tparams['xdims'][1]), yd=str(tparams['ydims'][1]), ylo=str(tparams['ydims'][0]),
                            median_intensity=str(tparams['median_intensity']), snr=str(tparams['snr']),
                            linking_max_distance=str(tparams['linking_max_distance']),
                            gap_closing_max_distance=str(tparams['gap_closing_max_distance']),
                            max_frame_gap=str(tparams['max_frame_gap']),
                            track_displacement=str(tparams['track_duration']),
                            track_duration=str(tparams['track_duration'])))
    fid.close()
    
    cmd = "%s --ij2 --headless --run %s" % (fiji_bin, tpfile.name)
    print(cmd)
    subp = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    fid = open(out_file, 'w')
    fid.write(subp.stdout.decode())
    fid.close()


def regress_sys(folder, all_videos, yfit, training_size, randselect=True,
                trainingdata=[], frame=0, have_output=True, download=True,
                bucket_name='ccurtis.data'):
    """Uses regression based on image intensities to select tracking parameters.

    This function uses regression methods from the scikit-learn module to
    predict the lower quality cutoff values for particle filtering in TrackMate
    based on the intensity distributions of input images. Currently only uses
    the first frame of videos for analysis, and is limited to predicting
    quality values.

    In practice, users will run regress_sys twice in different modes to build
    a regression system. First, set have_output to False. Function will return
    list of randomly selected videos to include in the training dataset. The
    user should then manually track particles using the Trackmate GUI, and enter
    these values in during the next round as the input yfit variable.

    Parameters
    ----------
    folder : str
        S3 directory containing video files specified in all_videos.
    all_videos: list of str
        Contains prefixes of video filenames of entire video set to be
        tracked.  Training dataset will be some subset of these videos.
    yfit: numpy.ndarray
        Contains manually acquired quality levels using Trackmate for the
        files contained in the training dataset.
    training_size : int
        Number of files in training dataset.
    randselect : bool
        If True, will randomly select training videos from all_videos.
        If False, will use trainingdata as input training dataset.
    trainingdata : list of str
        Optional manually selected prefixes of video filenames to be
        used as training dataset.
    have_output: bool
        If you have already acquired the quality values (yfit) for the
        training dataset, set to True.  If False, it will output the files
        the user will need to acquire quality values for.
    bucket_name : str
        S3 bucket containing videos to be downloaded for regression
        calculations.

    Returns
    -------
    regress_object : list of sklearn.svm.classes.
        Contains list of regression objects assembled from the training
        datasets.  Uses the mean, 10th percentile, 90th percentile, and
        standard deviation intensities to predict the quality parameter
        in Trackmate.
    tprefix : list of str
        Contains randomly selected images from all_videos to be included in
        training dataset.

    """

    if randselect:
        tprefix = []
        for i in range(0, training_size):
            random.seed(i+1)
            tprefix.append(all_videos[random.randint(0, len(all_videos))])
            if have_output is False:
                print("Get parameters for: {}".format(tprefix[i]))
    else:
        tprefix = trainingdata

    if have_output is True:
        # Define descriptors
        descriptors = np.zeros((training_size, 4))
        counter = 0
        for name in tprefix:
            local_im = name + '.tif'
            remote_im = "{}/{}".format(folder, local_im)
            if download:
                aws.download_s3(remote_im, local_im, bucket_name=bucket_name)
            test_image = sio.imread(local_im)
            descriptors[counter, 0] = np.mean(test_image[frame, :, :])
            descriptors[counter, 1] = np.std(test_image[frame, :, :])
            descriptors[counter, 2] = np.percentile(test_image[frame, :, :], 10)
            descriptors[counter, 3] = np.percentile(test_image[frame, :, :], 90)
            counter = counter + 1

        # Define regression techniques
        xfit = descriptors
        classifiers = [
            svm.SVR(),
            linear_model.SGDRegressor(),
            linear_model.BayesianRidge(),
            linear_model.LassoLars(),
            linear_model.ARDRegression(),
            linear_model.PassiveAggressiveRegressor(),
            linear_model.TheilSenRegressor(),
            linear_model.LinearRegression()]

        regress_object = []
        for item in classifiers:
            clf = item
            regress_object.append(clf.fit(xfit, yfit))

        return regress_object

    else:
        return tprefix


def regress_tracking_params(regress_object, to_track,
                            regmethod='LinearRegression', frame=0):
    """Predicts quality values to be used in particle tracking.

    Uses the regress object from regress_sys to predict tracking
    parameters for TrackMate analysis.

    Parameters
    ----------
    regress_object: list of sklearn.svm.classes.
        Obtained from regress_sys
    to_track: string
        Prefix of video files to be tracked.
    regmethod: {'LinearRegression', 'SVR', 'SGDRegressor', 'BayesianRidge',
        'LassoLars', 'ARDRegression', 'PassiveAggressiveRegressor',
        'TheilSenRegressor'}
        Desired regression method.

    Returns
    -------
    fqual: float
        Predicted quality factor used in TrackMate analysis.

    """

    local_im = to_track + '.tif'
    pX = np.zeros((1, 4))
    test_image = sio.imread(local_im)
    pX[0, 0] = np.mean(test_image[frame, :, :])
    pX[0, 1] = np.std(test_image[frame, :, :])
    pX[0, 2] = np.percentile(test_image[frame, :, :], 10)
    pX[0, 3] = np.percentile(test_image[frame:, :, :], 90)

    quality = []
    for item in regress_object:
        quality.append(item.predict(pX)[0])

    if regmethod == 'SVR':
        fqual = quality[0]
    elif regmethod == 'SGDRegressor':
        fqual = quality[1]
    elif regmethod == 'BayesianRidge':
        fqual = quality[2]
    elif regmethod == 'LassoLars':
        fqual = quality[3]
    elif regmethod == 'ARDRegression':
        fqual = quality[4]
    elif regmethod == 'PassiveAggressiveRegressor':
        fqual = quality[5]
    elif regmethod == 'TheilSenRegressor':
        fqual = quality[6]
    elif regmethod == 'LinearRegression':
        fqual = quality[7]
    else:
        fqual = 3.0

    return fqual
