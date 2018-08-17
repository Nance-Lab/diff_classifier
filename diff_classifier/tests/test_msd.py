import pandas as pd
import numpy as np
import skimage.io as sio
import numpy.ma as ma
import pandas.util.testing as pdt
import numpy.testing as npt
import diff_classifier.msd as msd


def test_nth_diff():

    data1 = {'col1': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data=data1)

    test_d = {'col1': [1, 1, 1, 1]}
    test_df = pd.DataFrame(data=test_d)

    pdt.assert_series_equal(msd.nth_diff(df['col1'], 1), test_df['col1'])

    # test2
    df = np.ones((5, 10))
    test_df = np.zeros((5, 9))
    npt.assert_equal(msd.nth_diff(df, 1, 1), test_df)

    df = np.ones((5, 10))
    test_df = np.zeros((4, 10))
    npt.assert_equal(msd.nth_diff(df, 1, 0), test_df)


def test_msd_calc():

    data1 = {'Frame': [1, 2, 3, 4, 5],
             'X': [5, 6, 7, 8, 9],
             'Y': [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data=data1)
    new_track = msd.msd_calc(df, 5)

    npt.assert_equal(np.array([0, 2, 8, 18, 32]
                              ).astype('float64'), new_track['MSDs'])
    npt.assert_equal(np.array([0, 0.25, 0.25, 0.25, 0.25]
                              ).astype('float64'), new_track['Gauss'])

    data1 = {'Frame': [1, 2, 3, 4, 5],
             'X': [5, 6, 7, 8, 9],
             'Y': [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data=data1)
    new_track = msd.msd_calc(df)

    npt.assert_equal(np.array([0, 2, 8, 18, 32, np.nan, np.nan, np.nan, np.nan,
                               np.nan]).astype('float64'), new_track['MSDs'])
    npt.assert_equal(np.array([0, 0.25, 0.25, 0.25, 0.25, np.nan, np.nan,
                               np.nan, np.nan, np.nan]
                              ).astype('float64'), new_track['Gauss'])


def test_all_msds():

    data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
             'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data=data1)

    di = {'Frame': [float(i) for i in[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]],
          'Track_ID': [float(i) for i in[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
          'X': [float(i) for i in[5, 6, 7, 8, 9, 1, 2, 3, 4, 5]],
          'Y': [float(i) for i in[6, 7, 8, 9, 10, 2, 3, 4, 5, 6]],
          'MSDs': [float(i) for i in[0, 2, 8, 18, 32, 0, 2, 8, 18, 32]],
          'Gauss': [0, 0.25, 0.25, 0.25, 0.25, 0, 0.25, 0.25, 0.25, 0.25]}
    cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss']

    dfi = pd.DataFrame(data=di)[cols]

    pdt.assert_frame_equal(dfi, msd.all_msds(df)[cols])


def test_make_xyarray():

    data1 = {'Frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
             'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data=data1)

    length = max(df['Frame']) + 1
    xyft = msd.make_xyarray(df, length=length)

    tt_array = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]).astype(float)
    ft_array = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]).astype(float)
    xt_array = np.array([[5, 1], [6, 2], [7, 3], [8, 4], [9, 5]]).astype(float)
    yt_array = np.array([[6, 2], [7, 3], [8, 4], [9, 5], [10, 6]]).astype(float)

    npt.assert_equal(xyft['tarray'], tt_array)
    npt.assert_equal(xyft['farray'], ft_array)
    npt.assert_equal(xyft['xarray'], xt_array)
    npt.assert_equal(xyft['yarray'], yt_array)

    # Second test
    data1 = {'Frame': [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
             'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data=data1)

    length = max(df['Frame']) + 1
    xyft = msd.make_xyarray(df, length=length)

    tt_array = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
                        ).astype(float)
    ft_array = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
                        ).astype(float)
    xt_array = np.array([[5, np.nan], [6, np.nan], [7, 1], [8, 2], [9, 3],
                         [np.nan, 4], [np.nan, 5]]).astype(float)
    yt_array = np.array([[6, np.nan], [7, np.nan], [8, 2], [9, 3], [10, 4],
                         [np.nan, 5], [np.nan, 6]]).astype(float)

    npt.assert_equal(xyft['tarray'], tt_array)
    npt.assert_equal(xyft['farray'], ft_array)
    npt.assert_equal(xyft['xarray'], xt_array)
    npt.assert_equal(xyft['yarray'], yt_array)


def test_all_msds2():

    data1 = {'Frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
             'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
             'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data=data1)

    di = {'Frame': [float(i) for i in[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
          'Track_ID': [float(i) for i in[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
          'X': [float(i) for i in[5, 6, 7, 8, 9, 1, 2, 3, 4, 5]],
          'Y': [float(i) for i in[6, 7, 8, 9, 10, 2, 3, 4, 5, 6]],
          'MSDs': [float(i) for i in[0, 2, 8, 18, 32, 0, 2, 8, 18, 32]],
          'Gauss': [0, 0.25, 0.25, 0.25, 0.25, 0, 0.25, 0.25, 0.25, 0.25]}
    cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss']

    dfi = pd.DataFrame(data=di)[cols]

    length = max(df['Frame']) + 1
    pdt.assert_frame_equal(dfi, msd.all_msds2(df, frames=length)[cols])
