import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import diff_classifier.features as ft
import diff_classifier.msd as msd


def test_make_xyarray():
    data = {'Frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'X': [np.nan, 6, 7, 8, 9, 1, 2, 3, 4, np.nan],
            'Y': [np.nan, 7, 8, 9, 10, 2, 3, 4, 5, np.nan]}
    dframe = pd.DataFrame(data=data)
    cols = ['Frame', 'Track_ID', 'X', 'Y', 'MSDs', 'Gauss']
    length = max(dframe['Frame']) + 1
    m_df = msd.all_msds2(dframe, frames=length)[cols]

    datat = {'Frame': [float(i) for i in[0, 1, 2, 3]],
             'Track_ID': [float(i) for i in[2, 2, 2, 2]],
             'X': [float(i) for i in[1, 2, 3, 4]],
             'Y': [float(i) for i in[2, 3, 4, 5]],
             'MSDs': [float(i) for i in[0, 2, 8, 18]],
             'Gauss': [float(i) for i in[0, 0.25, 0.25, 0.25]]}
    dft = pd.DataFrame(data=datat)

    pdt.assert_frame_equal(ft.unmask_track(m_df[m_df['Track_ID'] == 2]), dft)


def test_alpha_calc():
    frames = 5
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.alpha_calc(dframe) == (2.0000000000000004, 0.4999999999999998)

    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+5),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.alpha_calc(dframe) == (0.8201034110620524, 0.1494342948594476)


def test_gyration_tensor():
    frames = 6
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    o1, o2, o3, o4 = (8.0, 0.0, np.array([0.70710678, -0.70710678]),
                      np.array([0.70710678, 0.70710678]))
    d1, d2, d3, d4 = ft.gyration_tensor(dframe)

    assert d1 == o1
    assert d2 == o2
    npt.assert_almost_equal(o3, d3)
    npt.assert_almost_equal(o4, d4)

    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+5),
            'Y': np.cos(np.linspace(0, frames, frames)+5),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    o1, o2, o3, o4 = (0.47248734315843355, 0.3447097846562249,
                      np.array([0.83907153, 0.54402111]),
                      np.array([-0.54402111,  0.83907153]))
    d1, d2, d3, d4 = ft.gyration_tensor(dframe)

    assert d1 == o1
    assert d2 == o2
    npt.assert_almost_equal(o3, d3)
    npt.assert_almost_equal(o4, d4)


def test_kurtosis():
    frames = 5
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.kurtosis(dframe) == 4.079999999999999

    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+3),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.kurtosis(dframe) == 1.4759027695843443


def test_asymmetry():
    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    o1, o2, o3, o4, o5 = (20.0, 0.0, 1.0, 0.0, 0.69314718)
    d1, d2, d3, d4, d5 = ft.asymmetry(dframe)
    assert math.isclose(o1, d1, abs_tol=1e-10)
    assert math.isclose(o2, d2, abs_tol=1e-10)
    assert math.isclose(o3, d3, abs_tol=1e-10)
    assert math.isclose(o4, d4, abs_tol=1e-10)
    assert math.isclose(o5, d5, abs_tol=1e-10)

    frames = 100
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+3),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    o1, o2, o3, o4, o5 = (0.4254120816156, 0.42004967815488, 0.0001609000151811,
                          0.9873948021401, 2.0114322402896e-05)
    d1, d2, d3, d4, d5 = ft.asymmetry(dframe)
    assert math.isclose(o1, d1)
    assert math.isclose(o2, d2)
    assert math.isclose(o3, d3)
    assert math.isclose(o4, d4)
    assert math.isclose(o5, d5)


def test_minboundrect():
    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    d1, d2, d3, d4, d5, d6 = ft.minboundrect(dframe)
    o1, o2, o3, o4 = (-2.356194490192, 0, 14.142135623730, 0)
    o5 = np.array([10, 8])
    o6 = np.array([[5., 3.], [15., 13.], [15., 13.], [5., 3.]])

    # assert math.isclose(d1, o1, abs_tol=1e-10)
    assert math.isclose(d2, o2, abs_tol=1e-10)
    assert math.isclose(d3, o3, abs_tol=1e-10)
    assert math.isclose(d4, o4, abs_tol=1e-10)
    npt.assert_almost_equal(d5, o5)
    # npt.assert_almost_equal(d6, o6)

    frames = 100
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+3),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    d1, d2, d3, d4, d5, d6 = ft.minboundrect(dframe)
    o1, o2, o3, o4 = (-2.7345175425633, 3.7067697307443, 1.899593160348,
                      1.951349272106)
    o5 = np.array([-0.00098312,  0.00228019])
    o6 = np.array([[-1.2594591,  0.52217706],
                   [0.4849046,  1.27427376],
                   [1.25749286, -0.51761668],
                   [-0.48687084, -1.26971339]])

    # assert math.isclose(d1, o1, abs_tol=1e-10)
    assert math.isclose(d2, o2, abs_tol=1e-10)
    assert math.isclose(d3, o3, abs_tol=1e-10)
    assert math.isclose(d4, o4, abs_tol=1e-10)
    npt.assert_almost_equal(d5, o5)
    # npt.assert_almost_equal(d6, o6)


def test_aspectratio():
    frames = 6
    data = {'Frame': np.linspace(0, frames, frames),
            'X': [0, 1, 1, 2, 2, 3],
            'Y': [0, 0, 1, 1, 2, 2],
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.aspectratio(dframe)[0:2] == (3.9000000000000026, 0.7435897435897438)
    npt.assert_almost_equal(ft.aspectratio(dframe)[2], np.array([1.5, 1.]))


def test_boundedness():
    frames = 100
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+3),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.boundedness(dframe) == (0.607673328076712, 5.674370543833708,
                                      -0.0535555587618044)

    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)
    assert ft.boundedness(dframe) == (0.039999999999999994, 1.0,
                                      -0.21501108474766228)


def test_efficiency():
    frames = 100
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+3),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    assert ft.efficiency(dframe) ==\
        (0.003548421265914009, 0.0059620286331768385)

    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    assert ft.efficiency(dframe) == (10.0, 1.0)


def test_msd_ratio():
    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.sin(np.linspace(0, frames, frames)+3),
            'Y': np.cos(np.linspace(0, frames, frames)+3),
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    assert ft.msd_ratio(dframe, 1, 9) == 0.09708430006771959

    frames = 10
    data = {'Frame': np.linspace(0, frames, frames),
            'X': np.linspace(0, frames, frames)+5,
            'Y': np.linspace(0, frames, frames)+3,
            'Track_ID': np.ones(frames)}
    dframe = pd.DataFrame(data=data)
    dframe = msd.all_msds2(dframe, frames=frames+1)

    assert ft.msd_ratio(dframe, 1, 9) == -0.09876543209876543

# def test_calculate_features():
#     data = {'Frame': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
#          'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
#          'X': [0, 0, 1, 1, 2, 1, 1, 2, 2, 3],
#          'Y': [0, 1, 1, 2, 2, 0, 1, 1, 2, 2]}
#     dframe = pd.DataFrame(data=data)
#     dfi = msd.all_msds2(dframe, frames = 5)
#     feat = ft.calculate_features(dfi)

#     data = {'AR': np.ones(2)*3.9999999999999996,
#          'D_fit': np.ones(2)*0.1705189932550273,
#          'MSD_ratio': np.ones(2)*-0.2666666666666666,
#          'X': [0.75, 1.75],
#          'Y': [1.25, 1.25],
#          'Track_ID': [1.0, 2.0],
#          'alpha': np.ones(2)*1.7793370720777268,
#          'asymmetry1': np.ones(2)*0.9440237239896903,
#          'asymmetry2': np.ones(2)*0.12,
#          'asymmetry3': np.ones(2)*0.3691430189107616,
#          'boundedness': np.ones(2)*0.25,
#          'efficiency': np.ones(2)*2.0,
#          'elongation': np.ones(2)*0.75,
#          'fractal_dim': np.ones(2)*1.333333333333333,
#          'frames': [5.0, 5.0],
#          'kurtosis': np.ones(2)*1.166666666666667,
#          'straightness': np.ones(2)*0.7071067811865476,
#          'trappedness': np.ones(2)*-0.15258529289428524}
#     dfi = pd.DataFrame(data=data)

#     pdt.assert_frame_equal(dfi, feat)


def test_unmask_track():
    size = 10
    ID = np.ones(size)
    frame = np.linspace(5, size-1+5, size)
    x = frame + 1
    y = frame + 3

    data = {'Frame': frame,
            'Track_ID': ID,
            'X': x,
            'Y': y}
    di = pd.DataFrame(data=data)
    track = msd.all_msds2(di, frames=20)
    output = ft.unmask_track(track)

    data2 = {'Frame': frame-5,
             'Track_ID': ID,
             'X': x,
             'Y': y,
             'MSDs': np.array((0, 2, 8, 18, 32, 50, 72, 98, 128,
                               162)).astype('float64'),
             'Gauss': np.array((0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                                0.25, 0.25))}
    check = pd.DataFrame(data=data2)

    pdt.assert_frame_equal(output, check)
