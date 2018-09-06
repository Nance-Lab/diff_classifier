import os
import pytest
import numpy as np
import numpy.testing as npt
import pandas as pd
import diff_classifier.msd as msd
import diff_classifier.pca as pca
import diff_classifier.features as ft

is_travis = "CI" in os.environ.keys()


# @pytest.mark.skipif(is_travis, reason="Function behaves differently on Travis.")
@pytest.mark.xfail
def test_partial_corr():
    dataf = msd.random_traj_dataset()
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcorr = pca.partial_corr(feat)
    npt.assert_equal(24.0, np.round(np.sum(pcorr), 1))

    dataf = msd.random_traj_dataset(nparts=10)
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcorr = pca.partial_corr(feat)
    npt.assert_equal(47.9, np.round(np.sum(pcorr), 1))

    dataf = msd.random_traj_dataset(nparts=10, seed=9)
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcorr = pca.partial_corr(feat)
    npt.assert_equal(33.4, np.round(np.sum(pcorr), 1))

    dataf = msd.random_traj_dataset(nparts=10, nframes=40, seed=9)
    msds = msd.all_msds2(dataf, frames=40)
    feat = ft.calculate_features(msds)
    pcorr = pca.partial_corr(feat)
    npt.assert_equal(17.4, np.round(np.sum(pcorr), 1))

    dataf = msd.random_traj_dataset(nparts=10, nframes=40, ndist=(3, 5), seed=9)
    msds = msd.all_msds2(dataf, frames=40)
    feat = ft.calculate_features(msds)
    pcorr = pca.partial_corr(feat)
    npt.assert_equal(35.7, np.round(np.sum(pcorr), 1))


# @pytest.mark.skipif(is_travis, reason="Function behaves differently on Travis.")
@pytest.mark.xfail
def test_kmo():
    dataf = msd.random_traj_dataset(nparts=10, ndist=(1, 1), seed=3)
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    dataset = feat.drop(['frames', 'Track_ID'], axis=1)
    corrmatrix = np.corrcoef(dataset.transpose())
    npt.assert_equal(np.round(np.sum(corrmatrix), 1), 7.3)


def test_pca_analysis():
    dataf = msd.random_traj_dataset(nparts=10, ndist=(2, 6))
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcadataset = pca.pca_analysis(feat, dropcols=['frames', 'Track_ID'],
                                  n_components=5)
    npt.assert_equal(np.round(np.sum(pcadataset.components.values), 3), -0.971)


def test_plot_pca():
    print()


def test_build_KNN_model():
    output = ['F']*1000 + ['M']*1000
    data = {'output': output,
            0: np.append(np.random.normal(1, 1, size=1000),
                         np.random.normal(2, 1, size=1000)),
            1: np.append(np.random.normal(0.1, 0.1, size=1000),
                         np.random.normal(0.2, 0.1, size=1000))}
    dataf = pd.DataFrame(data)

    model, X, Y = pca.build_KNN_model(dataf, 'output', ['F', 'M'],
                                      equal_sampling=False, tsize=25,
                                      n_neighbors=5, input_cols=2)

    assert X.shape == (25, 2)
    assert Y.shape == (25,)


def test_predict_KNN():
    output = ['F']*1000 + ['M']*1000
    data = {'output': output,
            0: np.append(np.random.normal(1, 1, size=1000),
                         np.random.normal(2, 1, size=1000)),
            1: np.append(np.random.normal(0.1, 0.1, size=1000),
                         np.random.normal(0.2, 0.1, size=1000))}
    dataf = pd.DataFrame(data)

    model, X, Y = pca.build_KNN_model(dataf, 'output', ['F', 'M'],
                                      equal_sampling=False, tsize=25,
                                      n_neighbors=5, input_cols=2)

    testp = np.array([])
    for i in range(0, 30):
        KNNmod, X, y = pca.build_KNN_model(dataf, 'output', ['F', 'M'],
                                           equal_sampling=True, tsize=25,
                                           n_neighbors=5, input_cols=2)

        X2 = dataf.values[:, -2:]
        y2 = dataf.values[:, 0]
        testp = np.append(testp, pca.predict_KNN(KNNmod, X2, y2))

    assert np.mean(testp) > 0.6

    # test 2
    data = {'output': output,
            0: np.append(np.random.normal(1, 1, size=1000),
                         np.random.normal(1000, 1, size=1000)),
            1: np.append(np.random.normal(0.1, 0.1, size=1000),
                         np.random.normal(100, 0.1, size=1000))}
    dataf = pd.DataFrame(data)

    model, X, Y = pca.build_KNN_model(dataf, 'output', ['F', 'M'],
                                      equal_sampling=False, tsize=25,
                                      n_neighbors=5, input_cols=2)

    testp = np.array([])
    for i in range(0, 30):
        KNNmod, X, y = pca.build_KNN_model(dataf, 'output', ['F', 'M'],
                                           equal_sampling=True, tsize=25,
                                           n_neighbors=5, input_cols=2)

        X2 = dataf.values[:, -2:]
        y2 = dataf.values[:, 0]
        testp = np.append(testp, pca.predict_KNN(KNNmod, X2, y2))

    assert np.mean(testp) > 0.95


def test_feature_violin():

    np.random.seed(seed=1)
    dataset = {'label': 10*['yes'] + 10*['no'],
               0: np.random.normal(0.5, 1, size=20),
               1: np.random.normal(1, 2, size=20),
               2: np.random.normal(3, 10, size=20)
               }
    df = pd.DataFrame(data=dataset)

    to_violin = pca.feature_violin(df, fname='test.png')

    assert to_violin.values.shape == (60, 3)
    assert np.round(np.mean(to_violin['Feature Value']), 1) == 2.1


def test_feature_plot_2D():

    np.random.seed(seed=1)
    dataset = {'label': 500*['yes'] + 500*['no'],
               0: np.random.normal(0.5, 1, size=1000),
               1: np.random.normal(1, 2, size=1000),
               2: np.random.normal(3, 10, size=1000)
               }
    df = pd.DataFrame(data=dataset)

    xy = pca.feature_plot_2D(df, label='label', features=[0, 1], randsel=True,
                             fname='test1.png')
    assert len(xy[1]) == 200
    assert os.path.isfile('test1.png')

    xy = pca.feature_plot_2D(df, label='label', features=[0, 1], randsel=False)
    assert len(xy[1]) == 500
