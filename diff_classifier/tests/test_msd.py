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
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6],
             'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
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
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6],
             'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
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
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6],
             'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
    df = pd.DataFrame(data=data1)

    di = {'Frame': [float(i) for i in[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
          'Track_ID': [float(i) for i in[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
          'X': [float(i) for i in[5, 6, 7, 8, 9, 1, 2, 3, 4, 5]],
          'Y': [float(i) for i in[6, 7, 8, 9, 10, 2, 3, 4, 5, 6]],
          'MSDs': [float(i) for i in[0, 2, 8, 18, 32, 0, 2, 8, 18, 32]],
          'Gauss': [0, 0.25, 0.25, 0.25, 0.25, 0, 0.25, 0.25, 0.25, 0.25],
          'Quality': [float(i) for i in[10, 10, 10, 10, 10,
                      10, 10, 10, 10, 10]],
          'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
          'Mean_Intensity': [float(i) for i in[10, 10, 10, 10, 10,
                             10, 10, 10, 10, 10]]}
    cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss', 'Quality',
            'SN_Ratio', 'Mean_Intensity']

    dfi = pd.DataFrame(data=di)[cols]

    length = max(df['Frame']) + 1
    pdt.assert_frame_equal(dfi, msd.all_msds2(df, frames=length)[cols])


def test_geomean_msdisp():
    data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
             'Track_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
             'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
             'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6],
             'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}

    geomean_t = np.array([2., 8., 18., 32.])
    geostder_t = np.array([])
    df = pd.DataFrame(data=data1)
    msds = msd.all_msds2(df)
    msds.to_csv('msd_test.csv')

    geomean, geostder = msd.geomean_msdisp('test', umppx=1, fps=1, upload=False)
    npt.assert_equal(np.round(np.exp(geomean[geomean.mask == False].data), 1),
                     geomean_t)
    npt.assert_equal(np.round(np.exp(geostder[geostder.mask == False].data), 1),
                     geostder_t)

    # test 2
    data1 = {'Frame': [1, 2, 1, 2],
             'Track_ID': [1, 1, 2, 2],
             'X': [1, 2, 3, 4],
             'Y': [1, 2, 3, 4],
             'Quality': [10, 10, 10, 10],
             'SN_Ratio': [0.1, 0.1, 0.1, 0.1],
             'Mean_Intensity': [10, 10, 10, 10]}
    df = pd.DataFrame(data=data1)
    msds = msd.all_msds2(df)
    msds.to_csv('msd_test.csv')
    geomean, geostder = msd.geomean_msdisp('test', umppx=1, fps=1, upload=False)
    npt.assert_equal(geomean, np.nan*np.ones(651))
    npt.assert_equal(geostder, np.nan*np.ones(651))

    # test 3
    data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
             'Track_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
             'X': [5, 6, 7, 8, 9, 2, 4, 6, 8, 10],
             'Y': [6, 7, 8, 9, 10, 6, 8, 10, 12, 14],
             'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
             'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
    df = pd.DataFrame(data=data1)
    geomean_t = np.array([4., 16., 36., 64.])
    geostder_t = np.array([2., 2., 2., 2])
    msds = msd.all_msds2(df)
    msds.to_csv('msd_test.csv')

    geomean, geostder = msd.geomean_msdisp('test', umppx=1, fps=1, upload=False)
    npt.assert_equal(np.round(np.exp(geomean[geomean.mask == False].data), 1),
                     geomean_t)
    npt.assert_equal(np.round(np.exp(geostder[geostder.mask == False].data), 1),
                     geostder_t)


def test_binning():
    experiments = []
    for num in range(8):
        experiments.append('test_{}'.format(num))
    bins_t = {'test_W0': ['test_0', 'test_1'],
              'test_W1': ['test_2', 'test_3'],
              'test_W2': ['test_4', 'test_5'],
              'test_W3': ['test_6', 'test_7']}
    bin_names_t = ['test_W0', 'test_W1', 'test_W2', 'test_W3']
    slices, bins, bin_names = msd.binning(experiments)

    assert slices == 2
    assert bins == bins_t
    assert bin_names == bin_names_t


def test_precision_weight():
    experiments = []
    geomean = {}
    geostder = {}
    for num in range(4):
        name = 'test_{}'.format(num)
        experiments.append(name)
        data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                 'Track_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                 'X': [x*(num+1) for x in [5, 6, 7, 8, 9, 2, 4, 6, 8, 10]],
                 'Y': [x*(num+1) for x in [6, 7, 8, 9, 10, 6, 8, 10, 12, 14]],
                 'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                 'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data=data1)
        msds = msd.all_msds2(df)
        msds.to_csv('msd_test_{}.csv'.format(num))
        geomean[name], geostder[name] = msd.geomean_msdisp(name, umppx=1, fps=1,
                                                           upload=False)

    slices, bins, bin_names = msd.binning(experiments, wells=1)
    weights, w_holder = msd.precision_weight(experiments, geostder)
    weights_t = np.array([8.3, 8.3, 8.3, 8.3])
    npt.assert_equal(np.round(weights[weights.mask == False].data, 1),
                     weights_t)


def test_precision_averaging():
    experiments = []
    geomean = {}
    geostder = {}
    for num in range(4):
        name = 'test_{}'.format(num)
        experiments.append(name)
        data1 = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                 'Track_ID': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                 'X': [x*(num+1) for x in [5, 6, 7, 8, 9, 2, 4, 6, 8, 10]],
                 'Y': [x*(num+1) for x in [6, 7, 8, 9, 10, 6, 8, 10, 12, 14]],
                 'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                 'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                 'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data=data1)
        msds = msd.all_msds2(df)
        msds.to_csv('msd_test_{}.csv'.format(num))
        geomean[name], geostder[name] = msd.geomean_msdisp(name, umppx=1, fps=1,
                                                           upload=False)

    slices, bins, bin_names = msd.binning(experiments, wells=1)
    weights, w_holder = msd.precision_weight(experiments, geostder)
    geodata = msd.precision_averaging(experiments, geomean, geostder, weights,
                                      save=False)

    geostd_t = np.array([0.3, 0.3, 0.3, 0.3])
    geo_t = np.array([19.6,  78.4, 176.4, 313.5])
    npt.assert_equal(np.round(geodata.geostd[geodata.geostd.mask == False].data,
                              1), geostd_t)
    npt.assert_equal(np.round(
                     np.exp(geodata.geomean[
                              geodata.geomean.mask == False].data), 1), geo_t)


def test_random_walk():
    xi = np.array([0., 1.,  2.,  2.,  1.])
    yi = np.array([0., 0., 0., 1., 1.])
    x, y = msd.random_walk(nsteps=5)
    npt.assert_equal(xi, x)
    npt.assert_equal(yi, y)


def test_random_traj_dataset():
    di = {'Frame': [float(i) for i in[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]],
          'Track_ID': [float(i) for i in[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
          'X': np.array([1., 1.93045975532, 1.0, 1.0, 1.0, 0.0, 0.288183500979, 0.576367001958,
                        0.864550502937, 0.864550502937]),
          'Y': np.array([1., 0.06954024, -0.86091951, -0.86091951, 0.06954024,
                        4., 4., 4., 4.2881835, 4.2881835]),
          'Quality': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
          'SN_Ratio': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
          'Mean_Intensity': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]}
    cols = ['Frame', 'Track_ID', 'X', 'Y']
    dfi = pd.DataFrame(data=di)[cols]

    pdt.assert_frame_equal(dfi, msd.random_traj_dataset(nframes=5, nparts=2,
                                                        fsize=(0, 5))[cols])


def test_plot_all_experiments():
    print('To do later.')
