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
    dataf = msd.random_traj_dataset(nparts=10, ndist=(1, 1), seed=3)
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    dataset = feat.drop(['frames', 'Track_ID'], axis=1)
    corrmatrix = np.corrcoef(dataset.transpose())
    npt.assert_equal(np.round(np.sum(corrmatrix), 1), 7.3)
    
    
    # Correlation matrix and the partial covariance matrix.
    corrmatrix = np.corrcoef(dataset.transpose())
    pcorr = pca.partial_corr(dataset)

    # Calculation of the KMO statistic
    matrix = np.multiply(corrmatrix, corrmatrix)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    rij = np.sum(matrix) - np.trace(matrix)
    uij = np.sum(pcorr) - np.trace(pcorr)
    kmostat = rij/(rij+uij)
    print(kmostat)
    npt.assert_equal(np.round(np.sum(pcorr), 1), 37.1)
    npt.assert_equal(np.round(uij, 1), 21.1)
    npt.assert_equal(np.round(rij, 1), 73.6)
    npt.assert_equal(np.round(pca.kmo(dataset), 3), 0.777)


def test_pca_analysis():
    dataf = msd.random_traj_dataset(nparts=10, ndist=(2, 6))
    msds = msd.all_msds2(dataf, frames=100)
    feat = ft.calculate_features(msds)
    pcadataset = pca.pca_analysis(feat, dropcols=['frames', 'Track_ID'],
                                  n_components=5)
    npt.assert_equal(np.round(np.sum(pcadataset.components.values), 3), -0.971)


def test_plot_pca():
    print()
