import pandas as pd
import numpy as np
import skimage.io as sio
import numpy.ma as ma


def nth_diff(dataframe, n=1):
    """
    nth_diff(dataframe, n=int)

    Returns a new vector of size N - n containing the nth difference between vector elements.

    Parameters
    ----------
    dataframe : pandas column of floats or ints
        input data on which differences are to be calculated.
    n : int, default is 1
        Function calculated x(i) - x(i - n) for all values in pandas column

    Returns
    ----------
    diff : pandas column
        Pandas column of size N - n, where N is the original size of dataframe.

    Examples
    ----------
    >>> d = {'col1': [1, 2, 3, 4, 5]}
    >>> df = pd.DataFrame(data=d)
    >>> nth_diff(df['col1'], 1)
    0   1
    1   1
    2   1
    3   1
    Name: col1, dtype: int64

    >>> nth_diff(df['col1'], 2)
    0   2
    1   2
    2   2
    Name: col1, dtype: int64
    """

    assert type(dataframe) == 'pandas.core.frame.DataFrame', "dataframe must be a pandas dataframe."
    assert type(n) == 'int', "n must be an integer."

    test1 = dataframe[:-n].reset_index(drop=True)
    test2 = dataframe[n:].reset_index(drop=True)
    diff = test2 - test1
    return diff


def msd_calc(track):
    """
    msdcalc(track = pdarray)

    Returns numpy array containing MSD data calculated from an individual track.

    Parameters
    ----------
    track : pandas dataframe containing, at a minimum a 'Frame', 'X', and 'Y' column

    Returns
    ----------
    msd : numpy array the same length as track containing the calculated MSDs using the
          formula MSD = <(x-x0)**2>

    Examples
    ----------
    >>> d = {'Frame': [1, 2, 3, 4, 5],
             'X': [5, 6, 7, 8, 9],
             'Y': [6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data=d)
    >>> msd_calc(df)
    array([  0., 2., 8., 18., 32.])
    """

    assert type(track['Frame']) == 'pandas.core.series.Series', "track must contain column 'Frames'"
    assert type(track['X']) == 'pandas.core.series.Series', "track must contain column 'X'"
    assert type(track['Y']) == 'pandas.core.series.Series', "track must contain column 'Y'"
    assert track.shape[0] > 0, "track is empty"
    assert track['Frame'].dtype == 'int64', "Data in 'Frame' must be if type int64."
    assert track['X'].dtype == 'int64', "Data in 'X' must be if type int64."
    assert track['Y'].dtype == 'int64', "Data in 'Y' must be if type int64."

    length = track.shape[0]
    msd = np.zeros(length)

    for frame in range(0, length-1):
        # creates array to ignore when particles skip frames.
        inc = ma.masked_where(nth_diff(track['Frame'], n=frame+1) != frame+1, nth_diff(track['Frame'], n=frame+1))

        x = ma.array(np.square(nth_diff(track['X'], n=frame+1)), mask=inc.mask)
        y = ma.array(np.square(nth_diff(track['Y'], n=frame+1)), mask=inc.mask)

        msd[frame+1] = ma.mean(x + y)

    return msd


def all_msds(data):
    """
    Returns numpy array containing MSD data of all tracks in a trajectory pandas datframe.

    Parameters
    ----------
    data : pandas dataframe containing, at a minimum a 'Frame', 'Track_ID', 'X', and
           'Y' column.

    Returns
    ----------
    msd : numpy array the same length as data containing the calculated MSDs using the
          formula MSD = <(x-x0)**2>

    Examples
    ----------
    >>> d = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
             'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    >>> df = pd.DataFrame(data=d)
    >>> all_msds(df)
    array([0., 2., 8., 18., 32., 0., 2., 8., 18., 32.])
    """

    assert type(data['Frame']) == 'pandas.core.series.Series', "data must contain column 'Frames'"
    assert type(data['Track_ID']) == 'pandas.core.series.Series', "data must contain column 'Track_ID'"
    assert type(data['X']) == 'pandas.core.series.Series', "data must contain column 'X'"
    assert type(data['Y']) == 'pandas.core.series.Series', "data must contain column 'Y'"
    assert data.shape[0] > 0, "data is empty"
    assert data['Frame'].dtype == 'int64', "Data in 'Frame' must be if type int64."
    assert data['Track_ID'].dtype == 'int64', "Data in 'Track_ID' must be if type int64."
    assert data['X'].dtype == 'int64', "Data in 'X' must be if type int64."
    assert data['Y'].dtype == 'int64', "Data in 'Y' must be if type int64."

    trackids = data.Track_ID.unique()
    partcount = trackids.shape[0]
    msds = np.zeros(data.shape[0])

    for particle in range(0, partcount):
        single_track = data.loc[data['Track_ID'] == trackids[particle]].sort_values(['Track_ID', 'Frame'],
                                                                                    ascending=[1, 1]).reset_index(drop=True)
        if particle == 0:
            index1 = 0
            index2 = single_track.shape[0]
        else:
            index1 = index2
            index2 = index1 + single_track.shape[0]
        msds[index1:index2] = msd_calc(single_track)
    return msds
