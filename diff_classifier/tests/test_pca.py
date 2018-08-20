import numpy as np
import numpy.testing as npt
import diff_classifier.msd as msd
import diff_classifier.pca as pca
import diff_classifier.features as ft


def test_partial_corr():
    dataf = msd.random_traj_dataset()
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcorr = pca.partial_corr(feat)
    npt.assert_equal(24.0, np.round(np.sum(pcorr), 1))


def test_kmo():
    dataf = msd.random_traj_dataset(nparts=300, ndist=(2, 6))
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    dataset = feat.drop(['frames', 'Track_ID'], axis=1)
    npt.assert_equal(np.round(pca.kmo(dataset), 3), 0.958)


def test_pca_analysis():
    dataf = msd.random_traj_dataset(nparts=300, ndist=(2, 6))
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcadataset = pca.pca_analysis(feat, dropcols=['frames', 'Track_ID'],
                                  n_components=5)
    npt.assert_equal(np.round(np.sum(pcadataset.components.values), 3), -0.193)


def test_plot_pca():
    print()
