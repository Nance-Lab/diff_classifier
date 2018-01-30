import pandas as pd
import numpy as np
import skimage.io as sio
import numpy.ma as ma
import pandas.util.testing as pdt
import numpy.testing as npt
from diff_classifier.msd import all_msds, nth_diff, msd_calc


def test_nth_diff():

    d = {'col1': [1, 2, 3, 4, 5]}
    df = pd.DataFrame(data=d)

    test_d = {'col1': [1, 1, 1, 1]}
    test_df = pd.DataFrame(data=test_d)

    pdt.assert_series_equal(nth_diff(df['col1'], 1), test_df['col1'])


def test_msd_calc():

    d = {'Frame': [1, 2, 3, 4, 5],
         'X': [5, 6, 7, 8, 9],
         'Y': [6, 7, 8, 9, 10]}
    df = pd.DataFrame(data=d)
    msd, gauss = msd_calc(df)

    npt.assert_equal(np.array([0, 2, 8, 18, 32]).astype('float64'), msd)
    npt.assert_equal(np.array([0, 0.25, 0.25, 0.25, 0.25]).astype('float64'), gauss)
    


def test_all_msds():

    d = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
         'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
         'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
         'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data=d)
    
    di = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
          'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
          'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
          'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6],
          'MSDs': [float(i) for i in[0, 2, 8, 18, 32, 0, 2, 8, 18, 32]],
          'Gauss': [0, 0.25, 0.25, 0.25, 0.25, 0, 0.25, 0.25, 0.25, 0.25]}
    cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss']
    dfi = pd.DataFrame(data=di)[cols]

    pdt.assert_frame_equal(dfi, all_msds(df))
