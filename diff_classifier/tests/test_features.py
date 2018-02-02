import diff_classifier.features as ft
from diff_classifier.msd import all_msds, nth_diff, msd_calc
import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np
import pandas as pd
import math


def test_alpha_calc():
    frames = 5
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.alpha_calc(df) == (2.0000000000000004, 0.4999999999999999)

    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames)+3),
         'Y': np.cos(np.linspace(1, frames, frames)+3)}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.alpha_calc(df) == (0.023690002018364065, 0.5144436515510022)
    

def test_gyration_tensor():
    frames = 5
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    o1, o2, o3, o4 = (4.0, 4.4408920985006262e-16, np.array([ 0.70710678, -0.70710678]), np.array([0.70710678, 0.70710678]))
    d1, d2, d3, d4 = ft.gyration_tensor(df)

    assert d1 == o1
    assert d2 == o2
    npt.assert_almost_equal(o3, d3)
    npt.assert_almost_equal(o4, d4)
    
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames)+3),
         'Y': np.cos(np.linspace(1, frames, frames)+3)}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    o1, o2, o3, o4 = (0.53232560128104522, 0.42766829138901619, np.array([ 0.6020119 , -0.79848711]), np.array([-0.79848711, -0.6020119 ]))
    d1, d2, d3, d4 = ft.gyration_tensor(df)

    assert d1 == o1
    assert d2 == o2
    npt.assert_almost_equal(o3, d3)
    npt.assert_almost_equal(o4, d4)


def test_kurtosis():
    frames = 5
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.kurtosis(df) == 2.5147928994082829

    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames)+3),
         'Y': np.cos(np.linspace(1, frames, frames)+3)}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.kurtosis(df) == 1.8515139698652476

    
def test_asymmetry():
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    o1, o2, o3, o4, o5 = (16.5, 0.0, 1.0, 0.0, 0.69314718055994)
    d1, d2, d3, d4, d5 = ft.asymmetry(df)
    assert math.isclose(o1, d1)
    assert math.isclose(o2, d2)
    assert math.isclose(o3, d3)
    assert math.isclose(o4, d4)
    assert math.isclose(o5, d5)

    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames)+3),
         'Y': np.cos(np.linspace(1, frames, frames)+3)}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    o1, o2, o3, o4, o5 = (0.53232560128104, 0.42766829138901, 0.046430119259539, 0.80339606128247, 0.0059602683290953)
    d1, d2, d3, d4, d5 = ft.asymmetry(df)
    assert math.isclose(o1, d1)
    assert math.isclose(o2, d2)
    assert math.isclose(o3, d3)
    assert math.isclose(o4, d4)
    assert math.isclose(o5, d5)


def test_minBoundingRect():
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    d1, d2, d3, d4, d5, d6 = ft.minBoundingRect(df)
    o1, o2, o3, o4 = (-2.3561944901923, 0, 12.727922061357, 0)
    o5 = np.array([10.5, 8.5])
    o6 = np.array([[6., 4.], [15., 13.], [15., 13.], [6., 4.]])

    assert math.isclose(d1, o1, abs_tol=1e-13)
    assert math.isclose(d2, o2, abs_tol=1e-13)
    assert math.isclose(d3, o3, abs_tol=1e-13)
    assert math.isclose(d4, o4, abs_tol=1e-13)
    npt.assert_almost_equal(d5, o5)
    npt.assert_almost_equal(d6, o6)

    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames))+3,
         'Y': np.cos(np.linspace(1, frames, frames))+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    d1, d2, d3, d4, d5, d6 = ft.minBoundingRect(df)
    o1, o2, o3, o4 = (0.78318530717, 3.6189901131, 1.9949899732, 1.8140392491)
    o5 = np.array([3.02076903, 2.97913884])
    o6 = np.array([[4.3676025, 3.04013439], [2.95381341, 1.63258851], [1.67393557, 2.9181433 ], [3.08772466, 4.32568917]])

    assert math.isclose(d1, o1, abs_tol=1e-13)
    assert math.isclose(d2, o2, abs_tol=1e-13)
    assert math.isclose(d3, o3, abs_tol=1e-13)
    assert math.isclose(d4, o4, abs_tol=1e-13)
    npt.assert_almost_equal(d5, o5)
    npt.assert_almost_equal(d6, o6)
    
def test_aspectratio():
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.aspectratio(df) == (7165183131591494.0, 0.9999999999999999)

    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames))+3,
         'Y': np.cos(np.linspace(1, frames, frames))+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.aspectratio(df) == (1.0997501702946162, 0.09070257317431807)

    
def test_boundedness():
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.linspace(1, frames, frames)+5,
         'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.boundedness(df) == (1.0, 1.0, 0.0453113379707355)

    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
         'X': np.sin(np.linspace(1, frames, frames)+3),
         'Y': np.cos(np.linspace(1, frames, frames)+3)}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    assert ft.boundedness(df) == (0.9603705868989502, 2.7476524601589434, 0.03576118370932313)
    
    
def test_efficiency():
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    
    assert ft.efficiency(df) == (9.0, 0.9999999999999999)
    
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames))+3,
                 'Y': np.cos(np.linspace(1, frames, frames))+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    
    assert ft.efficiency(df) == (0.46192924086141945, 0.22655125514290225)
    
def test_msd_ratio():
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
                 'X': np.linspace(1, frames, frames)+5,
                 'Y': np.linspace(1, frames, frames)+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    
    assert ft.msd_ratio(df, 1, 9) == -0.18765432098765433
    
    frames = 10
    d = {'Frame': np.linspace(1, frames, frames),
                 'X': np.sin(np.linspace(1, frames, frames))+3,
                 'Y': np.cos(np.linspace(1, frames, frames))+3}
    df = pd.DataFrame(data=d)
    df['MSDs'], df['Gauss'] = msd_calc(df)
    
    assert ft.msd_ratio(df, 1, 9) == 0.04053708075268797
    
def test_calculate_features():
    d = {'Frame': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
         'Track_ID': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
         'X': [5, 6, 7, 8, 9, 1, 2, 3, 4, 5],
         'Y': [6, 7, 8, 9, 10, 2, 3, 4, 5, 6]}
    df = pd.DataFrame(data=d)
    all_msds(df)
    feat = ft.calculate_features(df)
    
    d = {'AR': np.ones(2)*1.698414e16,
         'D_fit': np.ones(2)*0.5,
         'MSD_ratio': np.ones(2)*-0.35,
         'Track_ID': [1, 2],
         'alpha': np.ones(2)*2.0,
         'asymmetry1': np.ones(2)*1.0,
         'asymmetry2': np.ones(2)*1.110223e-16,
         'asymmetry3': np.ones(2)*0.693147,
         'boundedness': np.ones(2)*1.0,
         'efficiency': np.ones(2)*4.0,
         'elongation': np.ones(2)*1.0,
         'fractal_dim': np.ones(2)*1.0,
         'kurtosis': np.ones(2)*2.283058,
         'straightness': np.ones(2)*1.0,
         'trappedness': np.ones(2)*0.04531133797073539}
    dfi = pd.DataFrame(data=d)
    
    pdt.assert_frame_equal(dfi, feat)