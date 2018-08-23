"""Functions to calculate mean squared displacements from trajectory data

This module includes functions to calculate mean squared displacements and
additional measures from input trajectory datasets as calculated by the
Trackmate ImageJ plugin.

"""
import warnings
import random as rand

import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy.stats as stats
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import diff_classifier.aws as aws


def nth_diff(dataframe, n=1, axis=0):
    """Calculates the nth difference between vector elements

    Returns a new vector of size N - n containing the nth difference between
    vector elements.

    Parameters
    ----------
    dataframe : pandas.core.series.Series of int or float
        Input data on which differences are to be calculated.
    n : int
        Function calculated xpos(i) - xpos(i - n) for all values in pandas
        series.
    axis : {0, 1}
        Axis along which differences are to be calculated.  Default is 0.  If 0,
        input must be a pandas series.  If 1, input must be a numpy array.

    Returns
    -------
    diff : pandas.core.series.Series of int or float
        Pandas series of size N - n, where N is the original size of dataframe.

    Examples
    --------
    >>> df = np.ones((5, 10))
    >>> nth_diff(df)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    >>> df = np.ones((5, 10))
    >>> nth_diff (df)
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    """

    assert isinstance(n, int), "n must be an integer."

    if dataframe.ndim == 1:
        length = dataframe.shape[0]
        if n <= length:
            test1 = dataframe[:-n].reset_index(drop=True)
            test2 = dataframe[n:].reset_index(drop=True)
            diff = test2 - test1
        else:
            diff = np.array([np.nan, np.nan])
    else:
        length = dataframe.shape[0]
        if n <= length:
            if axis == 0:
                test1 = dataframe[:-n, :]
                test2 = dataframe[n:, :]
            else:
                test1 = dataframe[:, :-n]
                test2 = dataframe[:, n:]
            diff = test2 - test1
        else:
            diff = np.array([np.nan, np.nan])

    return diff


def msd_calc(track, length=10):
    """Calculates mean squared displacement of input track.

    Returns numpy array containing MSD data calculated from an individual track.

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'X', and 'Y' column

    Returns
    -------
    new_track : pandas.core.frame.DataFrame
        Similar to input track.  All missing frames of individual trajectories
        are filled in with NaNs, and two new columns, MSDs and Gauss are added:
        MSDs, calculated mean squared displacements using the formula
        MSD = <(xpos-x0)**2>
        Gauss, calculated Gaussianity

    Examples
    --------
    >>> data1 = {'Frame': [1, 2, 3, 4, 5],
    ...          'X': [5, 6, 7, 8, 9],
    ...          'Y': [6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data=data1)
    >>> new_track = msd.msd_calc(df, 5)

    >>> data1 = {'Frame': [1, 2, 3, 4, 5],
    ...          'X': [5, 6, 7, 8, 9],
    ...          'Y': [6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data=data1)
    >>> new_track = msd.msd_calc(df)

    """

    meansd = np.zeros(length)
    gauss = np.zeros(length)
    new_frame = np.linspace(1, length, length)
    old_frame = track['Frame']
    oldxy = [track['X'], track['Y']]
    fxy = [interpolate.interp1d(old_frame, oldxy[0], bounds_error=False,
                                fill_value=np.nan),
           interpolate.interp1d(old_frame, oldxy[1], bounds_error=False,
                                fill_value=np.nan)]

    intxy = [ma.masked_equal(fxy[0](new_frame), np.nan),
             ma.masked_equal(fxy[1](new_frame), np.nan)]
    data1 = {'Frame': new_frame,
             'X': intxy[0],
             'Y': intxy[1]
             }
    new_track = pd.DataFrame(data=data1)

    for frame in range(0, length-1):
        xy = [np.square(nth_diff(new_track['X'], n=frame+1)),
              np.square(nth_diff(new_track['Y'], n=frame+1))]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meansd[frame+1] = np.nanmean(xy[0] + xy[1])
            gauss[frame+1] = np.nanmean(xy[0]**2 + xy[1]**2
                                        )/(2*(meansd[frame+1]**2))

    new_track['MSDs'] = pd.Series(meansd, index=new_track.index)
    new_track['Gauss'] = pd.Series(gauss, index=new_track.index)

    return new_track


def all_msds(data):
    """Calculates mean squared displacements of a trajectory dataset

    Returns numpy array containing MSD data of all tracks in a trajectory
    pandas dataframe.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'Track_ID', 'X', and
        'Y' column. Note: it is assumed that frames begins at 1, not 0 with this
        function. Adjust before feeding into function.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        Similar to input data.  All missing frames of individual trajectories
        are filled in with NaNs, and two new columns, MSDs and Gauss are added:
        MSDs, calculated mean squared displacements using the formula
        MSD = <(xpos-x0)**2>
        Gauss, calculated Gaussianity

    Examples
    --------
    >>> data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    ...          'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    ...          'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
    ...          'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    >>> df = pd.DataFrame(data=data1)
    >>> all_msds(df)

     """

    trackids = data.Track_ID.unique()
    partcount = trackids.shape[0]
    length = int(max(data['Frame']))
    new = {}
    new['length'] = partcount*length
    new['frame'] = np.zeros(new['length'])
    new['ID'] = np.zeros(new['length'])
    new['xy'] = [np.zeros(new['length']),
                 np.zeros(new['length'])]
    meansd = np.zeros(new['length'])
    gauss = np.zeros(new['length'])

    for particle in range(0, partcount):
        single_track = data.loc[data['Track_ID'] ==
                                trackids[particle]
                                ].sort_values(['Track_ID', 'Frame'],
                                              ascending=[1, 1]
                                              ).reset_index(drop=True)
        if particle == 0:
            index1 = 0
            index2 = length
        else:
            index1 = index2
            index2 = index2 + length
        new['single_track'] = msd_calc(single_track, length=length)
        new['frame'][index1:index2] = np.linspace(1, length, length)
        new['ID'][index1:index2] = particle+1
        new['xy'][0][index1:index2] = new['single_track']['X']
        new['xy'][1][index1:index2] = new['single_track']['Y']
        meansd[index1:index2] = new['single_track']['MSDs']
        gauss[index1:index2] = new['single_track']['Gauss']

    data1 = {'Frame': new['frame'],
             'Track_ID': new['ID'],
             'X': new['xy'][0],
             'Y': new['xy'][1],
             'MSDs': meansd,
             'Gauss': gauss}
    new_data = pd.DataFrame(data=data1)

    return new_data


def make_xyarray(data, length=651):
    """Rearranges xy position data into 2d arrays

    Rearranges xy data from input pandas dataframe into 2D numpy array.

    Parameters
    ----------
    data : pd.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'Track_ID', 'X', and
        'Y' column.
    length : int
        Desired length or number of frames to which to extend trajectories.
        Any trajectories shorter than the input length will have the extra space
        filled in with NaNs.

    Returns
    -------
    xyft : dict of np.ndarray
        Dictionary containing xy position data, frame data, and trajectory ID
        data. Contains the following keys:
        farray, frames data (length x particles)
        tarray, trajectory ID data (length x particles)
        xarray, x position data (length x particles)
        yarray, y position data (length x particles)

    Examples
    --------
    >>> data1 = {'Frame': [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
    ...          'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    ...          'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
    ...          'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    >>> df = pd.DataFrame(data=data1)
    >>> length = max(df['Frame']) + 1
    >>> xyft = msd.make_xyarray(df, length=length)
    {'farray': array([[0., 0.],
               [1., 1.],
               [2., 2.],
               [3., 3.],
               [4., 4.],
               [5., 5.],
               [6., 6.]]),
     'tarray': array([[1., 2.],
               [1., 2.],
               [1., 2.],
               [1., 2.],
               [1., 2.],
               [1., 2.],
               [1., 2.]]),
     'xarray': array([[ 5., nan],
               [ 6., nan],
               [ 7.,  1.],
               [ 8.,  2.],
               [ 9.,  3.],
               [nan,  4.],
     'yarray': [nan,  5.]]),
               array([[ 6., nan],
               [ 7., nan],
               [ 8.,  2.],
               [ 9.,  3.],
               [10.,  4.],
               [nan,  5.],
               [nan,  6.]])}

    """

    # Initial values
    first_p = int(min(data['Track_ID']))
    particles = int(max(data['Track_ID'])) - first_p + 1
    xyft = {}
    xyft['xarray'] = np.zeros((length, particles))
    xyft['yarray'] = np.zeros((length, particles))
    xyft['farray'] = np.zeros((length, particles))
    xyft['tarray'] = np.zeros((length, particles))

    track = data[data['Track_ID'] == first_p
                 ].sort_values(['Track_ID', 'Frame'],
                               ascending=[1, 1]).reset_index(drop=True)
    new_frame = np.linspace(0, length-1, length)

    old_frame = track['Frame'].values.astype(float)
    oldxy = [track['X'].values,
             track['Y'].values]
    fxy = [interpolate.interp1d(old_frame, oldxy[0], bounds_error=False,
                                fill_value=np.nan),
           interpolate.interp1d(old_frame, oldxy[1], bounds_error=False,
                                fill_value=np.nan)]
    intxy = [fxy[0](new_frame), fxy[1](new_frame)]

    # Fill in entire array
    xyft['xarray'][:, 0] = intxy[0]
    xyft['yarray'][:, 0] = intxy[1]
    xyft['farray'][:, 0] = new_frame
    xyft['tarray'][:, 0] = first_p

    for part in range(first_p+1, first_p+particles):
        track = data[data['Track_ID'] == part
                     ].sort_values(['Track_ID', 'Frame'],
                                   ascending=[1, 1]).reset_index(drop=True)

        old_frame = track['Frame']
        oldxy = [track['X'].values,
                 track['Y'].values]
        fxy = [interpolate.interp1d(old_frame, oldxy[0], bounds_error=False,
                                    fill_value=np.nan),
               interpolate.interp1d(old_frame, oldxy[1], bounds_error=False,
                                    fill_value=np.nan)]

        intxy = [fxy[0](new_frame),
                 fxy[1](new_frame)]

        xyft['xarray'][:, part-first_p] = intxy[0]
        xyft['yarray'][:, part-first_p] = intxy[1]
        xyft['farray'][:, part-first_p] = new_frame
        xyft['tarray'][:, part-first_p] = part

    return xyft


def all_msds2(data, frames=651):
    """Calculates mean squared displacements of input trajectory dataset

    Returns numpy array containing MSD data of all tracks in a trajectory pandas
    dataframe.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        Contains, at a minimum a 'Frame', 'Track_ID', 'X', and
        'Y' column. Note: it is assumed that frames begins at 0.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame
        Similar to input data.  All missing frames of individual trajectories
        are filled in with NaNs, and two new columns, MSDs and Gauss are added:
        MSDs, calculated mean squared displacements using the formula
        MSD = <(xpos-x0)**2>
        Gauss, calculated Gaussianity

    Examples
    --------
    >>> data1 = {'Frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
    ...          'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    ...          'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
    ...          'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    >>> df = pd.DataFrame(data=data1)
    >>> cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss']
    >>> om flength = max(df['Frame']) + 1
    >>> msd.all_msds2(df, frames=length)[cols]

    """
    if data.shape[0] > 2:
        try:
            xyft = make_xyarray(data, length=frames)
            length = xyft['xarray'].shape[0]
            particles = xyft['xarray'].shape[1]

            meansd = np.zeros((length, particles))
            gauss = np.zeros((length, particles))

            for frame in range(0, length-1):
                xpos = np.square(nth_diff(xyft['xarray'], n=frame+1))
                ypos = np.square(nth_diff(xyft['yarray'], n=frame+1))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    meansd[frame+1, :] = np.nanmean(xpos + ypos, axis=0)
                    gauss[frame+1, :] = np.nanmean(xpos**2 + ypos**2, axis=0
                                                   )/(2*(meansd[frame+1]**2))

            data1 = {'Frame': xyft['farray'].flatten('F'),
                     'Track_ID': xyft['tarray'].flatten('F'),
                     'X': xyft['xarray'].flatten('F'),
                     'Y': xyft['yarray'].flatten('F'),
                     'MSDs': meansd.flatten('F'),
                     'Gauss': gauss.flatten('F')}
            new_data = pd.DataFrame(data=data1)
        except ValueError:
            data1 = {'Frame': [],
                     'Track_ID': [],
                     'X': [],
                     'Y': [],
                     'MSDs': [],
                     'Gauss': []}
            new_data = pd.DataFrame(data=data1)
        except IndexError:
            data1 = {'Frame': [],
                     'Track_ID': [],
                     'X': [],
                     'Y': [],
                     'MSDs': [],
                     'Gauss': []}
            new_data = pd.DataFrame(data=data1)
    else:
        data1 = {'Frame': [],
                 'Track_ID': [],
                 'X': [],
                 'Y': [],
                 'MSDs': [],
                 'Gauss': []}
        new_data = pd.DataFrame(data=data1)

    return new_data


def geomean_msdisp(prefix, umppx=0.16, fps=100.02, upload=True,
                   remote_folder="01_18_Experiment", bucket='ccurtis.data'):
    """Comptes geometric averages of mean squared displacement datasets

    Calculates geometric averages and stadard errors for MSD datasets. Might
    error out if not formatted as output from all_msds2.

    Parameters
    ----------
    prefix : string
        Prefix of file name to be plotted e.g. features_P1.csv prefix is P1.
    umppx : float
        Microns per pixel of original images.
    fps : float
        Frames per second of video.
    upload : bool
        True if you want to upload to s3.
    remote_folder : string
        Folder in S3 bucket to upload to.
    bucket : string
        Name of S3 bucket to upload to.

    Returns
    -------
    geo_mean : numpy.ndarray
        Geometric mean of trajectory MSDs at all time points.
    geo_stder : numpy.ndarray
        Geometric standard errot of trajectory MSDs at all time points.

    """

    merged = pd.read_csv('msd_{}.csv'.format(prefix))
    try:
        particles = int(max(merged['Track_ID']))
        frames = int(max(merged['Frame']))
        ypos = np.zeros((particles+1, frames+1))

        for i in range(0, particles+1):
            ypos[i, :] = merged.loc[merged.Track_ID == i, 'MSDs']*umppx*umppx
            xpos = merged.loc[merged.Track_ID == i, 'Frame']/fps

        geo_mean = np.nanmean(ma.log(ypos), axis=0)
        geo_stder = ma.masked_equal(stats.sem(ma.log(ypos), axis=0,
                                              nan_policy='omit'), 0.0)

    except ValueError:
        geo_mean = np.nan*np.ones(1+int(max(merged['Frame'])))
        geo_stder = np.nan*np.ones(1+int(max(merged['Frame'])))

    np.savetxt('geomean_{}.csv'.format(prefix), geo_mean, delimiter=",")
    np.savetxt('geoSEM_{}.csv'.format(prefix), geo_stder, delimiter=",")

    if upload:
        aws.upload_s3('geomean_{}.csv'.format(prefix),
                      remote_folder+'/'+'geomean_{}.csv'.format(prefix),
                      bucket_name=bucket)
        aws.upload_s3('geoSEM_{}.csv'.format(prefix),
                      remote_folder+'/'+'geoSEM_{}.csv'.format(prefix),
                      bucket_name=bucket)

    return geo_mean, geo_stder


def binning(experiments, wells=4, prefix='test'):
    """Split set of input experiments into groups.

    Parameters
    ----------
    experiments : list of str
        List of experiment names.
    wells : int
        Number of groups to divide experiments into.

    Returns
    -------
    slices : int
        Number of experiments per group.
    bins : dict of list of str
        Dictionary, keys corresponding to group names, and elements containing
        lists of experiments in each group.
    bin_names : list of str
        List of group names

    """

    total_videos = len(experiments)
    bins = {}
    slices = int(total_videos/wells)
    bin_names = []

    for num in range(0, wells):
        slice1 = num*slices
        slice2 = (num+1)*(slices)
        pref = '{}_W{}'.format(prefix, num)
        bins[pref] = experiments[slice1:slice2]
        bin_names.append(pref)
    return slices, bins, bin_names


def precision_weight(group, geo_stder):
    """Calculates precision-based weights from input standard error data

    Calculates precision weights to be used in precision-averaged MSD
    calculations.

    Parameters
    ----------
    group : list of str
        List of experiment names to average. Each element corresponds to a key
        in geo_stder and geomean.
    geo_stder : dict of numpy.ndarray
        Each entry in dictionary corresponds to the standard errors of an MSD
        profile, the key corresponding to an experiment name.

    Returns
    -------
    weights: numpy.ndarray
        Precision weights to be used in precision averaging.
    w_holder : numpy.ndarray
        Precision values of each video at each time point.

    """

    frames = np.shape(geo_stder[group[0]])[0]
    slices = len(group)
    video_counter = 0
    w_holder = np.zeros((slices, frames))
    for sample in group:
        w_holder[video_counter, :] = 1/(geo_stder[sample]*geo_stder[sample])
        video_counter = video_counter + 1

    w_holder = ma.masked_equal(w_holder, 0.0)
    w_holder = ma.masked_equal(w_holder, 1.0)

    weights = ma.sum(w_holder, axis=0)

    return weights, w_holder


def precision_averaging(group, geomean, geo_stder, weights, save=True,
                        bucket='ccurtis.data', folder='test',
                        experiment='test'):
    """Calculates precision-weighted averages of MSD datasets.

    Parameters
    ----------
    group : list of str
        List of experiment names to average. Each element corresponds to a key
        in geo_stder and geomean.
    geomean : dict of numpy.ndarray
        Each entry in dictionary corresponds to an MSD profiles, they key
        corresponding to an experiment name.
    geo_stder : dict of numpy.ndarray
        Each entry in dictionary corresponds to the standard errors of an MSD
        profile, the key corresponding to an experiment name.
    weights : numpy.ndarray
        Precision weights to be used in precision averaging.

    Returns
    -------
    geo : numpy.ndarray
        Precision-weighted averaged MSDs from experiments specified in group
    geo_stder : numpy.ndarray
        Precision-weighted averaged SEMs from experiments specified in group

    """

    frames = np.shape(geo_stder[group[0]])[0]
    slices = len(group)

    video_counter = 0
    geo_holder = np.zeros((slices, frames))
    gstder_holder = np.zeros((slices, frames))
    w_holder = np.zeros((slices, frames))
    for sample in group:
        w_holder[video_counter, :] = (1/(geo_stder[sample]*geo_stder[sample])
                                      )/weights
        geo_holder[video_counter, :] = w_holder[video_counter, :
                                                ] * geomean[sample]
        gstder_holder[video_counter, :] = 1/(geo_stder[sample]*geo_stder[sample]
                                             )
        video_counter = video_counter + 1

    w_holder = ma.masked_equal(w_holder, 0.0)
    w_holder = ma.masked_equal(w_holder, 1.0)
    geo_holder = ma.masked_equal(geo_holder, 0.0)
    geo_holder = ma.masked_equal(geo_holder, 1.0)
    gstder_holder = ma.masked_equal(gstder_holder, 0.0)
    gstder_holder = ma.masked_equal(gstder_holder, 1.0)

    geo = ma.sum(geo_holder, axis=0)
    geo_stder = ma.sqrt((1/ma.sum(gstder_holder, axis=0)))

    if save:
        geo_f = 'geomean_{}.csv'.format(experiment)
        gstder_f = 'geoSEM_{}.csv'.format(experiment)
        np.savetxt(geo_f, geo, delimiter=',')
        np.savetxt(gstder_f, geo_stder, delimiter=',')
        aws.upload_s3(geo_f, '{}/{}'.format(folder, geo_f), bucket_name=bucket)
        aws.upload_s3(gstder_f, '{}/{}'.format(folder, gstder_f),
                      bucket_name=bucket)

    geodata = Bunch(geomean=geo, geostd=geo_stder, weighthold=w_holder,
                    geostdhold=gstder_holder)

    return geodata


def plot_all_experiments(experiments, bucket='ccurtis.data', folder='test',
                         yrange=(10**-1, 10**1), fps=100.02,
                         xrange=(10**-2, 10**0), upload=True,
                         outfile='test.png', exponential=True):
    """Plots precision-weighted averages of MSD datasets.

    Plots pre-calculated precision-weighted averages of MSD datasets calculated
    from precision_averaging and stored in an AWS S3 bucket.

    Parameters
    ----------
    group : list of str
        List of experiment names to plot. Each experiment must have an MSD and
        SEM file associated with it in s3.
    bucket : str
        S3 bucket from which to download data.
    folder : str
        Folder in s3 bucket from which to download data.
    yrange : list of float
        Y range of plot
    xrange: list of float
        X range of plot
    upload : bool
        True to upload to S3
    outfile : str
        Filename of output image

    """

    n = len(experiments)

    color = iter(cm.viridis(np.linspace(0, 0.9, n)))

    fig = plt.figure(figsize=(8.5, 8.5))
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    plt.xlabel('Tau (s)', fontsize=25)
    plt.ylabel(r'Mean Squared Displacement ($\mu$m$^2$)', fontsize=25)

    geo = {}
    gstder = {}
    counter = 0
    for experiment in experiments:
        aws.download_s3('{}/geomean_{}.csv'.format(folder, experiment),
                        'geomean_{}.csv'.format(experiment), bucket_name=bucket)
        aws.download_s3('{}/geoSEM_{}.csv'.format(folder, experiment),
                        'geoSEM_{}.csv'.format(experiment), bucket_name=bucket)

        geo[counter] = np.genfromtxt('geomean_{}.csv'.format(experiment))
        gstder[counter] = np.genfromtxt('geoSEM_{}.csv'.format(experiment))
        geo[counter] = ma.masked_equal(geo[counter], 0.0)
        gstder[counter] = ma.masked_equal(gstder[counter], 0.0)

        frames = np.shape(gstder[counter])[0]
        xpos = np.linspace(0, frames-1, frames)/fps
        c = next(color)

        if exponential:
            plt.loglog(xpos, np.exp(geo[counter]), c=c, linewidth=6,
                       label=experiment)
            plt.loglog(xpos, np.exp(geo[counter] - 1.96*gstder[counter]),
                       c=c, dashes=[6,2], linewidth=4)
            plt.loglog(xpos, np.exp(geo[counter] + 1.96*gstder[counter]),
                       c=c, dashes=[6,2], linewidth=4)
        else:
            plt.loglog(xpos, geo[counter], c=c, linewidth=6,
                       label=experiment)
            plt.loglog(xpos, geo[counter] - 1.96*gstder[counter], c=c,
                       dashes=[6,2], linewidth=4)
            plt.loglog(xpos, geo[counter] + 1.96*gstder[counter], c=c,
                       dashes=[6,2], linewidth=4)

        counter = counter + 1

    plt.legend(frameon=False, prop={'size': 16})

    if upload:
        fig.savefig(outfile, bbox_inches='tight')
        aws.upload_s3(outfile, folder+'/'+outfile, bucket_name=bucket)


def random_walk(nsteps=100, seed=1, start=(0, 0)):
    """Creates 2d random walk trajectory.

    Parameters
    ----------
    nsteps : int
        Number of steps for trajectory to move.
    seed : int
        Seed for pseudo-random number generator for reproducability.
    start : tuple of int or float
        Starting xy coordinates at which the random walk begins.

    Returns
    -------
    x : numpy.ndarray
        Array of x coordinates of random walk.
    y : numpy.ndarray
        Array of y coordinates of random walk.

    """

    rand.seed(a=seed)

    x = np.zeros(nsteps)
    y = np.zeros(nsteps)
    x[0] = start[0]
    y[0] = start[1]

    for i in range(1, nsteps):
        val = rand.randint(1, 4)
        if val == 1:
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif val == 2:
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif val == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        else:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1

    return x, y


def random_traj_dataset(nframes=100, nparts=30, seed=1, fsize=(0, 512),
                        ndist=(1, 2)):
    """Creates a random population of random walks.

    Parameters
    ----------
    nframes : int
        Number of frames for each random trajectory.
    nparts : int
        Number of particles in trajectory dataset.
    seed : int
        Seed for pseudo-random number generator for reproducability.
    fsize : tuple of int or float
        Scope of points over which particles may start at.
    ndist : tuple of int or float
        Parameters to generate normal distribution, mu and sigma.

    Returns
    -------
    dataf : pandas.core.frame.DataFrame
        Trajectory data containing a 'Frame', 'Track_ID', 'X', and
        'Y' column.

    """

    frames = []
    trackid = []
    x = []
    y = []
    start = [0, 0]
    pseed = seed

    for i in range(nparts):
        rand.seed(a=i+pseed)
        start[0] = rand.randint(fsize[0], fsize[1])
        rand.seed(a=i+3+pseed)
        start[1] = rand.randint(fsize[0], fsize[1])
        rand.seed(a=i+5+pseed)
        weight = rand.normalvariate(mu=ndist[0], sigma=ndist[1])

        trackid = np.append(trackid, np.array([i]*nframes))
        xi, yi = random_walk(nsteps=nframes, seed=i)
        x = np.append(x, weight*xi+start[0])
        y = np.append(y, weight*yi+start[1])
        frames = np.append(frames, np.linspace(0, nframes-1, nframes))

    datai = {'Frame': frames,
             'Track_ID': trackid,
             'X': x,
             'Y': y}
    dataf = pd.DataFrame(data=datai)

    return dataf


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
