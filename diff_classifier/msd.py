import pandas as pd
import numpy as np
import skimage.io as sio
import numpy.ma as ma
from scipy import interpolate


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

    #assert type(dataframe) == pd.core.series.Series, "dataframe must be a pandas dataframe."
    #assert type(n) == int, "n must be an integer."
   
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
            test1 = dataframe[:-n, :]
            test2 = dataframe[n:, :]
            diff = test2 - test1
        else:
            diff = np.array([np.nan, np.nan])
      
    return diff


def msd_calc(track, length=10):
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
    gauss : numpy array the same length as track containing the calculated Gaussianity

    Examples
    ----------
    >>> d = {'Frame': [1, 2, 3, 4, 5],
             'X': [5, 6, 7, 8, 9],
             'Y': [6, 7, 8, 9, 10]}
    >>> df = pd.DataFrame(data=d)
    >>> msd_calc(df)
    (array([  0.,   2.,   8.,  18.,  32.]),
     array([ 0.  ,  0.25,  0.25,  0.25,  0.25]))
    """

    #assert type(track['Frame']) == pd.core.series.Series, "track must contain column 'Frame'"
    #assert type(track['X']) == pd.core.series.Series, "track must contain column 'X'"
    #assert type(track['Y']) == pd.core.series.Series, "track must contain column 'Y'"
    #assert track.shape[0] > 0, "track is empty"
    #assert track['Frame'].dtype == np.int64 or np.float64, "Data in 'Frame' must be if type int64."
    #assert track['X'].dtype == np.int64 or np.float64, "Data in 'X' must be if type int64."
    #assert track['Y'].dtype == np.int64 or np.float64, "Data in 'Y' must be if type int64."

    MSD = np.zeros(length)
    gauss = np.zeros(length)

    inc = nth_diff(track['Frame'], n=1)
    new_x = np.zeros(length)
    new_y = np.zeros(length)

#     smaller = np.shape(track)[0]

#     new_x[0] = track['X'][0]
#     new_y[0] = track['Y'][0]
#     current = 1
#     for i in range(1, smaller):
#         if inc[i-1] == 1:
#             new_x[current] = track['X'][i]
#             new_y[current] = track['Y'][i]
#             current = current + 1
#         else:
#             new_x[current:int(current+inc[i-1])-1] = track['X'][i-1]
#             new_x[int(current+inc[i-1])-1] = track['X'][i]
#             new_y[current:int(current+inc[i-1])-1] = track['Y'][i-1]
#             new_y[int(current+inc[i-1])-1] = track['Y'][i]
#             current = int(current + inc[i-1])

    new_frame = np.linspace(1, length, length)
    old_frame = track['Frame']
    old_x = track['X']
    old_y = track['Y']
    fx = interpolate.interp1d(old_frame, old_x, bounds_error = False, fill_value = np.nan)
    fy = interpolate.interp1d(old_frame, old_y, bounds_error = False, fill_value = np.nan)

    int_x = ma.masked_equal(fx(new_frame), np.nan)
    int_y = ma.masked_equal(fy(new_frame), np.nan)
    d = {'Frame': new_frame,
                 'X': int_x,
                 'Y': int_y}
    new_track = pd.DataFrame(data=d)

    for frame in range(0, length-1):
        # creates array to ignore when particles skip frames.
        #inc = ma.masked_where(msd.nth_diff(track['Frame'], n=frame+1) != frame+1, msd.nth_diff(track['Frame'], n=frame+1))

        x = np.square(nth_diff(new_track['X'], n=frame+1))
        y = np.square(nth_diff(new_track['Y'], n=frame+1))

        MSD[frame+1] = np.nanmean(x + y)
        gauss[frame+1] = np.nanmean(x**2 + y**2)/(2*(MSD[frame+1]**2))

    new_track['MSDs'] = pd.Series(MSD, index=new_track.index)
    new_track['Gauss'] = pd.Series(gauss, index=new_track.index)

    return new_track


def all_msds(data):
    """
    Returns numpy array containing MSD data of all tracks in a trajectory pandas dataframe.

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
    """

    #assert type(data['Frame']) == pd.core.series.Series, "data must contain column 'Frame'"
    #assert type(data['Track_ID']) == pd.core.series.Series, "data must contain column 'Track_ID'"
    #assert type(data['X']) == pd.core.series.Series, "data must contain column 'X'"
    #assert type(data['Y']) == pd.core.series.Series, "data must contain column 'Y'"
    #assert data.shape[0] > 0, "data is empty"
    #assert data['Frame'].dtype == np.int64 or np.float64, "Data in 'Frame' must be if type int64."
    #assert data['Track_ID'].dtype == np.int64 or np.float64, "Data in 'Track_ID' must be if type int64."
    #assert data['X'].dtype == np.int64 or np.float64, "Data in 'X' must be if type int64."
    #assert data['Y'].dtype == np.int64 or np.float64, "Data in 'Y' must be if type int64."

    trackids = data.Track_ID.unique()
    partcount = trackids.shape[0]
    data['MSDs'] = np.zeros(data.shape[0])
    data['Gauss'] = np.zeros(data.shape[0])
    length = int(max(data['Frame'])) + 1

    new_length = partcount*(length)
    new_frame = np.zeros(new_length)
    new_ID = np.zeros(new_length)
    new_x = np.zeros(new_length)
    new_y = np.zeros(new_length)
    MSD = np.zeros(new_length)
    gauss = np.zeros(new_length)

    for particle in range(0, partcount):
        single_track = data.loc[data['Track_ID'] == trackids[particle]].sort_values(['Track_ID', 'Frame'],
                                                                                    ascending=[1, 1]).reset_index(drop=True)
        if particle == 0:
            index1 = 0
            index2 = length
        else:
            index1 = index2
            index2 = index2 + length
        #data['MSDs'][index1:index2], data['Gauss'][index1:index2] = msd_calc(single_track)
        #data['Frame'][index1:index2] = data['Frame'][index1:index2] - (data['Frame'][index1] - 1)
        new_single_track =  msd_calc(single_track, length=length)
        new_frame[index1:index2] = np.linspace(1, length, length)
        new_ID[index1:index2] = particle+1
        new_x[index1:index2] = new_single_track['X']
        new_y[index1:index2] = new_single_track['Y']
        MSD[index1:index2] = new_single_track['MSDs']
        gauss[index1:index2] = new_single_track['Gauss']

    d = {'Frame': new_frame,
         'Track_ID': new_ID,
         'X': new_x,
         'Y': new_y,
         'MSDs': MSD,
         'Gauss': gauss}
    new_data = pd.DataFrame(data=d)
   
    return new_data


def make_xyarray(data, length=651):
    
    #Initial values
    first_p = int(min(data['Track_ID']))
    particles = int(max(data['Track_ID'])) - first_p
    x_array = np.zeros((length, particles))
    y_array = np.zeros((length, particles))
    f_array = np.zeros((length, particles))
    t_array = np.zeros((length, particles))

    track = data[data['Track_ID']==first_p].sort_values(['Track_ID', 'Frame'], ascending=[1, 1]).reset_index(drop=True)
    new_frame = np.linspace(0, length-1, length)

    old_frame = track['Frame']
    old_x = track['X']
    old_y = track['Y']
    fx = interpolate.interp1d(old_frame, old_x, bounds_error = False, fill_value = np.nan)
    fy = interpolate.interp1d(old_frame, old_y, bounds_error = False, fill_value = np.nan)

    int_x = fx(new_frame)
    int_y = fy(new_frame)

    #Fill in entire array
    x_array[:, 0] = int_x
    y_array[:, 0] = int_y
    f_array[:, 0] = new_frame
    t_array[:, 0] = first_p

    for part in range(first_p+1, first_p+particles):
        track = data[data['Track_ID']==part].sort_values(['Track_ID', 'Frame'], ascending=[1, 1]).reset_index(drop=True)

        old_frame = track['Frame']
        old_x = track['X']
        old_y = track['Y']
        fx = interpolate.interp1d(old_frame, old_x, bounds_error = False, fill_value = np.nan)
        fy = interpolate.interp1d(old_frame, old_y, bounds_error = False, fill_value = np.nan)

        int_x = fx(new_frame)
        int_y = fy(new_frame)

        x_array[:, part-first_p] = int_x
        y_array[:, part-first_p] = int_y
        f_array[:, part-first_p] = new_frame
        t_array[:, part-first_p] = part
    
    return f_array, t_array, x_array, y_array


def all_msds2(data, frames=651):
    try:
        f_array, t_array, x_array, y_array = make_xyarray(data, length=frames)

        length = int(x_array.shape[0])
        particles = int(x_array.shape[1])

        MSD = np.zeros((length, particles))
        gauss = np.zeros((length, particles))

        for frame in range(0, length-1):
            x = np.square(nth_diff(x_array, n=frame+1))
            y = np.square(nth_diff(y_array, n=frame+1))

            MSD[frame+1, :] = np.nanmean(x + y, axis=0)
            gauss[frame+1, :] = np.nanmean(x**2 + y**2, axis=0)/(2*(MSD[frame+1]**2))

        d = {'Frame': f_array.flatten('F'),
             'Track_ID': t_array.flatten('F'),
             'X': x_array.flatten('F'),
             'Y': y_array.flatten('F'),
             'MSDs': MSD.flatten('F'),
             'Gauss': gauss.flatten('F')}
        new_data = pd.DataFrame(data=d)
    except ValueError:
        d = {'Frame': [],
             'Track_ID': [],
             'X': [],
             'Y': [],
             'MSDs': [],
             'Gauss': []}
        new_data = pd.DataFrame(data=d)
    except IndexError:
        d = {'Frame': [],
             'Track_ID': [],
             'X': [],
             'Y': [],
             'MSDs': [],
             'Gauss': []}
        new_data = pd.DataFrame(data=d)
    
    return new_data