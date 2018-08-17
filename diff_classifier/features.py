"""Functions to calculate trajectory features from input trajectory data

This module provides functions to calculate trajectory features based off the
ImageJ plugin TrajClassifer by Thorsten Wagner. See details at
https://imagej.net/TraJClassifier.

"""

import math
import struct

import pandas as pd
import numpy as np
import numpy.linalg as LA
import numpy.ma as ma
from scipy.optimize import curve_fit
import diff_classifier.msd as msd


def unmask_track(track):
    """Removes empty frames from inpute trajectory datset.

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        At a minimum, must contain a Frame, Track_ID, X, Y, MSDs, and
        Gauss column.

    Returns
    -------
    comp_track : pandas.core.frame.DataFrame
        Similar to track, but has all masked components removed.

    """
    xpos = ma.masked_invalid(track['X'])
    msds = ma.masked_invalid(track['MSDs'])
    x_mask = ma.getmask(xpos)
    msd_mask = ma.getmask(msds)
    comp_frame = ma.compressed(ma.masked_where(msd_mask, track['Frame']))
    compid = ma.compressed(ma.masked_where(msd_mask, track['Track_ID']))
    comp_x = ma.compressed(ma.masked_where(x_mask, track['X']))
    comp_y = ma.compressed(ma.masked_where(x_mask, track['Y']))
    comp_msd = ma.compressed(ma.masked_where(msd_mask, track['MSDs']))
    comp_gauss = ma.compressed(ma.masked_where(msd_mask, track['Gauss']))

    data1 = {'Frame': comp_frame,
             'Track_ID': compid,
             'X': comp_x,
             'Y': comp_y,
             'MSDs': comp_msd,
             'Gauss': comp_gauss
             }
    comp_track = pd.DataFrame(data=data1)
    return comp_track


def alpha_calc(track):
    """Calculates alpha, the exponential fit parameter for MSD data

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        At a minimum, must contain a Frames and a MSDs column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    alph : numpy.float64
        The anomalous exponent derived by fitting MSD values to the function,
        <rad**2(n)> = 4*dcoef*(n*delt)**alph
    dcoef : numpy.float64
        The fitted diffusion coefficient derived by fitting MSD values to the
        function above.

    Examples
    --------
    >>> frames = 5
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> alpha_calc(dframe)
    (2.0000000000000004, 0.4999999999999999)

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames)+3),
                 'Y': np.cos(np.linspace(1, frames, frames)+3)}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> alpha_calc(dframe)
    (0.023690002018364065, 0.5144436515510022)
    """

    ypos = track['MSDs']
    xpos = track['Frame']

    def msd_alpha(xpos, alph, dcoef):
        return 4*dcoef*(xpos**alph)

    try:
        popt, pcov = curve_fit(msd_alpha, xpos, ypos)
        alph = popt[0]
        dcoef = popt[1]
    except RuntimeError:
        print('Optimal parameters not found. Print NaN instead.')
        alph = np.nan
        dcoef = np.nan
    return alph, dcoef


def gyration_tensor(track):
    """Calculates the eigenvalues and eigenvectors of the gyration tensor of the
    input trajectory.

    Parameters
    ----------
    track : pandas DataFrame
        At a minimum, must contain an X and Y column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    eig1 : numpy.float64
        Dominant eigenvalue of the gyration tensor.
    eig2 : numpy.float64
        Secondary eigenvalue of the gyration tensor.
    eigv1 : numpy.ndarray
        Dominant eigenvector of the gyration tensor.
    eigv2 : numpy.ndarray
        Secondary eigenvector of the gyration tensor.

    Examples
    --------
    >>> frames = 5
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> gyration_tensor(dframe)
    (4.0,
    4.4408920985006262e-16,
    array([ 0.70710678, -0.70710678]),
    array([ 0.70710678,  0.70710678]))

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames)+3),
                 'Y': np.cos(np.linspace(1, frames, frames)+3)}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> gyration_tensor(dframe)
    (0.53232560128104522,
    0.42766829138901619,
    array([ 0.6020119 , -0.79848711]),
    array([-0.79848711, -0.6020119 ]))
    """

    dframe = track
    assert isinstance(dframe, pd.core.frame.DataFrame), "track must be a pandas\
     dataframe."
    assert isinstance(dframe['X'], pd.core.series.Series), "track must contain\
     X column."
    assert isinstance(dframe['Y'], pd.core.series.Series), "track must contain\
     Y column."
    assert dframe.shape[0] > 0, "track must not be empty."

    matrixa = np.sum((dframe['X'] - np.mean(
                     dframe['X']))**2)/dframe['X'].shape[0]
    matrixb = np.sum((dframe['Y'] - np.mean(
                     dframe['Y']))**2)/dframe['Y'].shape[0]
    matrixab = np.sum((dframe['X'] - np.mean(
                      dframe['X']))*(dframe['Y'] - np.mean(
                                     dframe['Y'])))/dframe['X'].shape[0]

    eigvals, eigvecs = LA.eig(np.array([[matrixa, matrixab],
                                       [matrixab, matrixb]]))
    dom = np.argmax(np.abs(eigvals))
    rec = np.argmin(np.abs(eigvals))
    eig1 = eigvals[dom]
    eig2 = eigvals[rec]
    eigv1 = eigvecs[dom]
    eigv2 = eigvecs[rec]
    return eig1, eig2, eigv1, eigv2


def kurtosis(track):
    """Calculates the kurtosis of input track.

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        At a minimum, must contain an X and Y column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    kurt : numpy.float64
        Kurtosis of the input track.  Calculation based on projected 2D
        positions on the dominant eigenvector of the radius of gyration tensor.

    Examples
    --------
    >>> frames = 5
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> kurtosis(dframe)
    2.5147928994082829

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames)+3),
                 'Y': np.cos(np.linspace(1, frames, frames)+3)}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> kurtosis(dframe)
    1.8515139698652476

    """

    dframe = track
    assert isinstance(dframe, pd.core.frame.DataFrame), "track must be a pandas\
     dataframe."
    assert isinstance(dframe['X'], pd.core.series.Series), "track must contain\
     X column."
    assert isinstance(dframe['Y'], pd.core.series.Series), "track must contain\
     Y column."
    assert dframe.shape[0] > 0, "track must not be empty."

    eig1, eig2, eigv1, eigv2 = gyration_tensor(dframe)
    projection = dframe['X']*eigv1[0] + dframe['Y']*eigv1[1]

    kurt = np.mean((projection - np.mean(
                   projection))**4/(np.std(projection)**4))

    return kurt


def asymmetry(track):
    """Calculates the asymmetry of the trajectory.

    Parameters
    ----------
    track : pandas DataFrame
        At a minimum, must contain an X and Y column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    eig1 : numpy.float64
        Dominant eigenvalue of the gyration tensor.
    eig2 : numpy.float64
        Secondary eigenvalue of the gyration tensor.
    asym1 : numpy.float64
        asymmetry of the input track.  Equal to 0 for circularly symmetric
        tracks, and 1 for linear tracks.
    asym2 : numpy.float64
        alternate definition of asymmetry.  Equal to 1 for circularly
        symmetric tracks, and 0 for linear tracks.
    asym3 : numpy.float64
        alternate definition of asymmetry.

    Examples
    --------
    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
             'X': np.linspace(1, frames, frames)+5,
             'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> asymmetry(dframe)
    (16.5, 0.0, 1.0, 0.0, 0.69314718055994529)

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
               'X': np.sin(np.linspace(1, frames, frames)+3),
               'Y': np.cos(np.linspace(1, frames, frames)+3)}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> asymmetry(dframe)
    (0.53232560128104522,
    0.42766829138901619,
    0.046430119259539708,
    0.80339606128247354,
    0.0059602683290953052)

    """
    dframe = track
    assert isinstance(dframe, pd.core.frame.DataFrame), "track must be a pandas\
     dataframe."
    assert isinstance(dframe['X'], pd.core.series.Series), "track must contain\
     X column."
    assert isinstance(dframe['Y'], pd.core.series.Series), "track must contain\
     Y column."
    assert dframe.shape[0] > 0, "track must not be empty."

    eig1, eig2, eigv1, eigv2 = gyration_tensor(track)
    asym1 = (eig1**2 - eig2**2)**2/(eig1**2 + eig2**2)**2
    asym2 = eig2/eig1
    asym3 = -np.log(1-((eig1-eig2)**2)/(2*(eig1+eig2)**2))

    return eig1, eig2, asym1, asym2, asym3


def minboundrect(track):
    """Calculates the minimum bounding rectangle of an input trajectory.

    Parameters
    ----------
    dframe : pandas.core.frame.DataFrame
        At a minimum, must contain an X and Y column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    rot_angle : numpy.float64
        Angle of rotation of the bounding box.
    area : numpy.float64
        Area of the bounding box.
    width : numpy.float64
        Width of the bounding box.
    height : numpy.float64
        Height of the bounding box.
    center_point : numpy.ndarray
        Center point of the bounding box.
    corner_pts : numpy.ndarray
        Corner points of the bounding box.

    Examples
    --------
    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> minboundrect(dframe)
    (-2.3561944901923448,
    2.8261664256307952e-14,
    12.727922061357855,
    2.2204460492503131e-15,
    array([ 10.5,   8.5]),
    array([[  6.,   4.],
           [ 15.,  13.],
           [ 15.,  13.],
           [  6.,   4.]]))

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames))+3,
                 'Y': np.cos(np.linspace(1, frames, frames))+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> minboundrect(dframe)
    (0.78318530717958657,
    3.6189901131223992,
    1.9949899732081091,
    1.8140392491811692,
    array([ 3.02076903,  2.97913884]),
    array([[ 4.3676025 ,  3.04013439],
           [ 2.95381341,  1.63258851],
           [ 1.67393557,  2.9181433 ],
           [ 3.08772466,  4.32568917]]))

    Notes
    -----
    Based off of code from the following repo:
    https://github.com/dbworth/minimum-area-bounding-rectangle/blob/master/
    python/min_bounding_rect.py
    """

    dframe = track
    assert isinstance(dframe, pd.core.frame.DataFrame), "track must be a pandas\
     dataframe."
    assert isinstance(dframe['X'], pd.core.series.Series), "track must contain\
     X column."
    assert isinstance(dframe['Y'], pd.core.series.Series), "track must contain\
     Y column."
    assert dframe.shape[0] > 0, "track must not be empty."

    df2 = np.zeros((dframe.shape[0]+1, 2))
    df2[:-1, :] = dframe[['X', 'Y']].values
    df2[-1, :] = dframe[['X', 'Y']].values[0, :]
    hull_points_2d = df2

    edges = np.zeros((len(hull_points_2d)-1, 2))

    for i in range(len(edges)):
        edge_x = hull_points_2d[i+1, 0] - hull_points_2d[i, 0]
        edge_y = hull_points_2d[i+1, 1] - hull_points_2d[i, 1]
        edges[i] = [edge_x, edge_y]

    edge_angles = np.zeros((len(edges)))

    for i in range(len(edge_angles)):
        edge_angles[i] = math.atan2(edges[i, 1], edges[i, 0])
    edge_angles = np.unique(edge_angles)

    start_area = 2 ** (struct.Struct('i').size * 8 - 1) - 1
    min_bbox = (0, start_area, 0, 0, 0, 0, 0, 0)
    for i in range(len(edge_angles)):
        rads = np.array([[math.cos(edge_angles[i]),
                          math.cos(edge_angles[i]-(math.pi/2))],
                        [math.cos(edge_angles[i]+(math.pi/2)),
                         math.cos(edge_angles[i])]])

        rot_points = np.dot(rads, np.transpose(hull_points_2d))

        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        width = max_x - min_x
        height = max_y - min_y
        area = width*height

        if area < min_bbox[1]:
            min_bbox = (edge_angles[i], area, width, height,
                        min_x, max_x, min_y, max_y)

    angle = min_bbox[0]
    rads = np.array([[math.cos(angle), math.cos(angle-(math.pi/2))],
                     [math.cos(angle+(math.pi/2)), math.cos(angle)]])

    min_x = min_bbox[4]
    max_x = min_bbox[5]
    min_y = min_bbox[6]
    max_y = min_bbox[7]

    center_x = (min_x + max_x)/2
    center_y = (min_y + max_y)/2
    center_point = np.dot([center_x, center_y], rads)

    corner_pts = np.zeros((4, 2))
    corner_pts[0] = np.dot([max_x, min_y], rads)
    corner_pts[1] = np.dot([min_x, min_y], rads)
    corner_pts[2] = np.dot([min_x, max_y], rads)
    corner_pts[3] = np.dot([max_x, max_y], rads)

    return (angle, min_bbox[1], min_bbox[2], min_bbox[3],
            center_point, corner_pts)


def aspectratio(track):
    """Calculates the aspect ratio of the rectangle containing the input track.

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        At a minimum, must contain an X and Y column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    aspratio : numpy.float64
        aspect ratio of the trajectory.  Always >= 1.
    elong : numpy.float64
        elongation of the trajectory.  A transformation of the aspect ratio
        given by 1 - aspratio**-1.

    Examples
    --------
    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> aspectratio(dframe)
    (5732146505273195.0, 0.99999999999999978)

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames))+3,
                 'Y': np.cos(np.linspace(1, frames, frames))+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> aspectratio(dframe)
    (1.0997501702946164, 0.090702573174318291)

    """

    dframe = track
    assert isinstance(dframe, pd.core.frame.DataFrame), "track must be a pandas\
     dataframe."
    assert isinstance(dframe['X'], pd.core.series.Series), "track must contain\
     X column."
    assert isinstance(dframe['Y'], pd.core.series.Series), "track must contain\
     Y column."
    assert dframe.shape[0] > 0, "track must not be empty."

    rangle, area, width, height, center_point, corner_pts = minboundrect(track)
    aspratio = width/height
    if aspratio > 1:
        print()
    else:
        aspratio = 1/aspratio
    elong = 1 - (1/aspratio)

    return aspratio, elong, center_point


def boundedness(track, framerate=1):
    """
    Calculates the boundedness, fractal dimension, and trappedness of the input track.

    Parameters
    ----------
    track : pandas DataFrame
        At a minimum, must contain a Frames and a MSDs column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.
    framerate : framrate of the video being analyzed.  Actually cancels out. So
        why did I include this. Default is 1.

    Returns
    -------
    bound : numpy.float64
        Boundedness of the input track.  Quantifies how much a particle with
        diffusion coefficient dcoef is restricted by a circular confinement of radius
        rad when it diffuses for a time duration N*delt.  Defined as bound = dcoef*N*delt/rad**2.
        For this case, dcoef is the short time diffusion coefficient (after 2 frames),
        and rad is half the maximum distance between any two positions.
    fractd : numpy.float64
        The fractal path dimension defined as fractd = log(N)/log(N*data1*l**-1) where netdisp
        is the total length (sum over all steplengths), N is the number of steps,
        and data1 is the largest distance between any two positions.
    probf : numpy.float64
        The probability that a particle with diffusion coefficient dcoef and traced
        for a period of time N*delt is trapped in region r0.  Given by
        pt = 1 - exp(0.2048 - 0.25117*(dcoef*N*delt/r0**2))
        For this case, dcoef is the short time diffusion coefficient, and r0 is half
        the maximum distance between any two positions.

    Examples
    --------
    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> boundedness(dframe)
    (1.0, 1.0000000000000002, 0.045311337970735499)

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames)+3),
                 'Y': np.cos(np.linspace(1, frames, frames)+3)}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> boundedness(dframe)
    (0.96037058689895005, 2.9989749477908401, 0.03576118370932313)
    """

    dframe = track
    assert isinstance(dframe, pd.core.frame.DataFrame), "track must be a pandas\
     dataframe."
    assert isinstance(dframe['X'], pd.core.series.Series), "track must contain\
     X column."
    assert isinstance(dframe['Y'], pd.core.series.Series), "track must contain\
     Y column."
    assert dframe.shape[0] > 0, "track must not be empty."

    dframe = track

    if dframe.shape[0] > 2:
        length = dframe.shape[0]
        distance = np.zeros((length, length))

        for frame in range(0, length-1):
            distance[frame, 0:length-frame-1] = (np.sqrt(msd.nth_diff(dframe['X'], frame+1)**2 + msd.nth_diff(dframe['Y'], frame+1)**2).values)

        netdisp = np.sum((np.sqrt(msd.nth_diff(dframe['X'], 1)**2 + msd.nth_diff(dframe['Y'], 1)**2).values))
        rad = np.max(distance)/2
        N = dframe['Frame'][dframe['Frame'].shape[0]-1]
        fram = N*framerate
        dcoef = dframe['MSDs'][2]/(4*fram)

        bound = dcoef*fram/(rad**2)
        fractd = np.log(N)/np.log(N*2*rad/netdisp)
        probf = 1 - np.exp(0.2048 - 0.25117*(dcoef*fram/(rad**2)))
    else:
        bound = np.nan
        fractd = np.nan
        probf = np.nan

    return bound, fractd, probf


def efficiency(track):
    """Calculates the efficiency and straitness of the input track

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        At a minimum, must contain a Frames and a MSDs column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.

    Returns
    -------
    eff : numpy.float64
        Efficiency of the input track.  Relates the sum of squared step
        lengths.  Based on Helmuth et al. (2007) and defined as:
        E = |xpos(N-1)-xpos(0)|**2/SUM(|xpos(i) - xpos(i-1)|**2
    strait : numpy.float64
        Relates the net displacement netdisp to the sum of step lengths and is
        defined as:
        S = |xpos(N-1)-xpos(0)|/SUM(|xpos(i) - xpos(i-1)|

    Examples
    --------
    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> ft.efficiency(dframe)
    (9.0, 0.9999999999999999)

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames))+3,
                 'Y': np.cos(np.linspace(1, frames, frames))+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> ft.efficiency(dframe)
    (0.46192924086141945, 0.22655125514290225)

    """

    dframe = track
    length = dframe.shape[0]
    num = (msd.nth_diff(dframe['X'],
                        length-1)**2 + msd.nth_diff(dframe['Y'],
                                                    length-1)**2)[0]
    num2 = np.sqrt(num)

    den = np.sum(msd.nth_diff(dframe['X'],
                              1)**2 + msd.nth_diff(dframe['Y'], 1)**2)
    den2 = np.sum(np.sqrt(msd.nth_diff(dframe['X'],
                          1)**2 + msd.nth_diff(dframe['Y'], 1)**2))

    eff = num/den
    strait = num2/den2
    return eff, strait


def msd_ratio(track, fram1=3, fram2=100):
    """Calculates the MSD ratio of the input track at the specified frames.

    Parameters
    ----------
    track : pandas.core.frame.DataFrame
        At a minimum, must contain a Frames and a MSDs column.  The function
        msd_calc can be used to generate the correctly formatted pd dataframe.
    fram1 : int
        First frame at which to calculate the MSD ratio.
    fram2 : int
        Last frame at which to calculate the MSD ratio.

    Returns
    -------
    ratio: numpy.float64
        MSD ratio as defined by
        [MSD(fram1)/MSD(fram2)] - [fram1/fram2]
        where fram1 < fram2.  For Brownian motion, it is 0; for restricted
        motion it is < 0.  For directed motion it is > 0.

    Examples
    --------
    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> ft.msd_ratio(dframe, 1, 9)
    -0.18765432098765433

    >>> frames = 10
    >>> data1 = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames))+3,
                 'Y': np.cos(np.linspace(1, frames, frames))+3}
    >>> dframe = pd.DataFrame(data=data1)
    >>> dframe['MSDs'], dframe['Gauss'] = msd_calc(dframe)
    >>> ft.msd_ratio(dframe, 1, 9)
    0.04053708075268797

    """

    dframe = track
    assert fram1 < fram2, "fram1 must be less than fram2"
    ratio = (dframe['MSDs'][fram1]/dframe['MSDs'][fram2]) - (
             dframe['Frame'][fram1]/dframe['Frame'][fram2])
    return ratio


def calculate_features(dframe, framerate=1):
    """Calculates multiple features from input MSD dataset and stores in pandas
    dataframe.

    Parameters
    ----------
    dframe : pandas.core.frame.DataFrame
        Output from msd.all_msds2.  Must have at a minimum the following
        columns:
        Track_ID, Frame, X, Y, and MSDs.
    framerate : int or float
        Framerate of the input videos from which trajectories were calculated.
        Required for accurate calculation of some features.  Default is 1.
        Possibly not required. Ignore if performing all calcuations without
        units.

    Returns
    -------
    datai: pandas.core.frame.DataFrame
        Contains a row for each trajectory in dframe.  Holds the following
        features of each trajetory: Track_ID, alpha, D_fit, kurtosis,
        asymmetry1, asymmetry2, asymmetry3, aspect ratio (AR), elongation,
        boundedness, fractal dimension (fractal_dim), trappedness, efficiency,
        straightness, MSD ratio, frames, X, and Y.

    Examples
    --------
    See example outputs from individual feature functions.

    """

    # Skeleton of Trajectory features metadata table.
    # Builds entry for each unique Track ID.
    holder = dframe.Track_ID.unique().astype(float)
    die = {'Track_ID': holder,
           'alpha': holder,
           'D_fit': holder,
           'kurtosis': holder,
           'asymmetry1': holder,
           'asymmetry2': holder,
           'asymmetry3': holder,
           'AR': holder,
           'elongation': holder,
           'boundedness': holder,
           'fractal_dim': holder,
           'trappedness': holder,
           'efficiency': holder,
           'straightness': holder,
           'MSD_ratio': holder,
           'frames': holder,
           'X': holder,
           'Y': holder}

    datai = pd.DataFrame(data=die)

    trackids = dframe.Track_ID.unique()
    partcount = trackids.shape[0]

    for particle in range(0, partcount):
        single_track_masked = dframe.loc[dframe['Track_ID'] == trackids[
                                         particle]].sort_values(
                                         ['Track_ID', 'Frame'],
                                         ascending=[1,
                                                    1]).reset_index(drop=True)
        single_track = unmask_track(single_track_masked)
        (datai['alpha'][particle],
         datai['D_fit'][particle]) = alpha_calc(single_track)
        datai['kurtosis'][particle] = kurtosis(single_track)
        (eig1, eig2, datai['asymmetry1'][particle],
         datai['asymmetry2'][particle],
         datai['asymmetry3'][particle]) = asymmetry(single_track)
        (datai['AR'][particle], datai['elongation'][particle],
         (datai['X'][particle],
          datai['Y'][particle])) = aspectratio(single_track)
        (datai['boundedness'][particle], datai['fractal_dim'][particle],
         datai['trappedness'][particle]) = boundedness(single_track, framerate)
        (datai['efficiency'][particle],
         datai['straightness'][particle]) = efficiency(single_track)
        datai['frames'][particle] = single_track.shape[0]
        if single_track['Frame'][single_track.shape[0]-2] > 2:
            datai['MSD_ratio'][particle] = msd_ratio(single_track, 2,
                                                     single_track['Frame'][
                                                      single_track.shape[0]-2])
        else:
            datai['MSD_ratio'][particle] = np.nan

    return datai
