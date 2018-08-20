"""Performs principle component analysis on input datasets.

This module performs principle component analysis on input datasets using
functions from scikit-learn. It is optimized to data formats used in
diff_classifier, but can potentially be extended to other applications.

"""

import pandas as pd
import numpy as np
from scipy import stats, linalg
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler as stscale
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def partial_corr(mtrx):
    """Calculates linear partial correlation coefficients

    Returns the sample linear partial correlation coefficients between pairs of
    variables in mtrx, controlling for the remaining variables in mtrx.



    Parameters
    ----------
    mtrx : array-like, shape (n, p)
        Array with the different variables. Each column of mtrx is taken as a
        variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of mtrx[:, i] and mtrx[:, j]
        controlling for the remaining variables in mtrx.

    Notes
    -----

    Partial Correlation in Python (clone of Matlab's partialcorr)

    This uses the linear regression approach to compute the partial
    correlation (might be slow for a huge number of variables). The
    algorithm is detailed here:

    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression

    Taking X and Y two variables of interest and Z the matrix with all the
    variable minus {X, Y}, the algorithm can be summarized as

    1) perform a normal linear least-squares regression with X as the target
       and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and
       Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2
       and #4

    The result is the partial correlation between X and Y while controlling for
    the effect of Z

    Adapted from code by Fabian Pedregosa-Izquierdo:
    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com

    """

    mtrx = np.asarray(mtrx)
    pfeat = mtrx.shape[1]
    pcorr = np.zeros((pfeat, pfeat), dtype=np.float)
    for i in range(pfeat):
        pcorr[i, i] = 1
        for j in range(i+1, pfeat):
            idx = np.ones(pfeat, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(mtrx[:, idx], mtrx[:, j])[0]
            beta_j = linalg.lstsq(mtrx[:, idx], mtrx[:, i])[0]

            res_j = mtrx[:, j] - mtrx[:, idx].dot(beta_i)
            res_i = mtrx[:, i] - mtrx[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)[0]
            pcorr[i, j] = corr
            pcorr[j, i] = corr

    return pcorr


def kmo(dataset):
    """Calculates the Kaiser-Meyer-Olkin measure on an input dataset

    Parameters
    ----------
    dataset : array-like, shape (n, p)
        Array containing n samples and p features. Must have no NaNs.
        Ideally scaled before performing test.

    Returns
    -------
    kmostat : float
        KMO test value

    Notes
    -----
    Based on calculations shown here:

    http://www.statisticshowto.com/kaiser-meyer-olkin/

        -- 0.00-0.49  unacceptable
        -- 0.50-0.59  miserable
        -- 0.60-0.69  mediocre
        -- 0.70-0.79  middling
        -- 0.80-0.89  meritorious
        -- 0.90-1.00  marvelous

    """

    # Correlation matrix and the partial covariance matrix.
    corrmatrix = np.corrcoef(dataset.transpose())
    pcorr = partial_corr(dataset)

    # Calculation of the KMO statistic
    matrix = corrmatrix*corrmatrix
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    rij = 0
    uij = 0
    for row in range(0, rows):
        for col in range(0, cols):
            if not row == col:
                rij = rij + matrix[row, col]
                uij = uij + pcorr[row, col]

    kmostat = rij/(rij+uij)
    print(kmostat)
    return kmostat


def pca_analysis(dataset, dropcols=[], imputenans=True, scale=True,
                 n_components=5):
    """Performs a primary component analysis on an input dataset

    Parameters
    ----------
    dataset : pandas.core.frame.DataFrame, shape (n, p)
        Input dataset with n samples and p features
    dropcols : list
        Columns to exclude from pca analysis. At a minimum, user must exclude
        non-numeric columns.
    imputenans : bool
        If True, impute NaN values as column means.
    scale : bool
        If True, columns will be scaled to a mean of zero and a standard
        deviation of 1.
    n_components : int
        Desired number of components in principle component analysis.

    Returns
    -------
    pcadataset : diff_classifier.pca.Bunch
        Contains outputs of PCA analysis, including:
        scaled : numpy.ndarray, shape (n, p)
            Scaled dataset with n samples and p features
        pcavals : pandas.core.frame.DataFrame, shape (n, n_components)
            Output array of n_component features of each original sample
        final : pandas.core.frame.DataFrame, shape (n, p+n_components)
            Output array with principle components append to original array.
        prcomps : pandas.core.frame.DataFrame, shape (5, n_components)
            Output array displaying the top 5 features contributing to each
            principle component.
        prvals : dict of list of str
            Output dictionary of of the pca scores for the top 5 features
            contributing to each principle component.
        components : pandas.core.frame.DataFrame, shape (p, n_components)
            Raw pca scores.

    """

    dataset_num = dataset.drop(dropcols, axis=1)
    dataset_raw = dataset_num.values

    # Fill in NaN values
    if imputenans:
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(dataset_raw)
        dataset_clean = imp.transform(dataset_raw)
    else:
        dataset_clean = dataset_raw

    # Scale inputs
    if scale:
        scaler = stscale()
        scaler.fit(dataset_clean)
        dataset_scaled = scaler.transform(dataset_clean)
    else:
        dataset_scaled = dataset_clean

    pcadataset = Bunch(scaled=dataset_scaled)
    pca1 = pca(n_components=n_components)
    pca1.fit(dataset_scaled)

    # Cumulative explained variance ratio
    cum_var = 0
    explained_v = pca1.explained_variance_ratio_
    print('Cumulative explained variance:')
    for i in range(0, n_components):
        cum_var = cum_var + explained_v[i]
        print('{} component: {}'.format(i, cum_var))

    prim_comps = {}
    pcadataset.prvals = {}
    comps = pca1.components_
    pcadataset.components = pd.DataFrame(comps.transpose())
    for num in range(0, n_components):
        highest = np.abs(pcadataset.components[
                         num]).values.argsort()[-5:][::-1]
        pels = []
        pcadataset.prvals[num] = pcadataset.components[num].values[highest]
        for col in highest:
            pels.append(dataset_num.columns[col])
        prim_comps[num] = pels

    # Main contributors to each primary component
    pcadataset.prcomps = pd.DataFrame.from_dict(prim_comps)
    pcadataset.pcavals = pd.DataFrame(pca1.transform(dataset_scaled))
    pcadataset.final = pd.concat([dataset, pcadataset.pcavals], axis=1)

    return pcadataset


def plot_pca(datasets, figsize=(8, 8), lwidth=8.0,
             labels=['Sample1', 'Sample2'], savefig=True, filename='test.png'):
    """Plots the average output features from a PCA analysis in polar
    coordinates

    Parameters
    ----------
    datasets : dict of numpy.ndarray
        Dictionary with n samples and p features to plot.
    figize : list
        Dimensions of output figure e.g. (8, 8)
    lwidth : float
        Width of plotted lines in figure
    labels : list of str
        Labels to display in legend.
    savefig : bool
        If True, saves figure
    filename : str
        Desired output filename

    """

    fig = plt.figure(figsize=figsize)
    for key in datasets:
        N = datasets[key].shape[0]
    width = (2*np.pi) / N
    color = iter(cm.viridis(np.linspace(0, 1, N)))

    theta = np.linspace(0.0, 2 * np.pi, N+1, endpoint=True)
    radii = {}
    bars = {}

    ax = plt.subplot(111, polar=True)
    counter = 0
    for key in datasets:
        c = next(color)
        radii[key] = np.append(datasets[key], datasets[key][0])
        bars[key] = ax.plot(theta, radii[key], linewidth=lwidth, color=c,
                            label=labels[counter])
        counter = counter + 1
    plt.legend(bbox_to_anchor=(0.90, 1), loc=2, borderaxespad=0.,
               frameon=False, fontsize=20)

    # # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(np.abs(r / 2.5)))
    #     bar.set_alpha(0.8)
    ax.set_xticks(np.pi/180. * np.linspace(0, 360, N, endpoint=False))
    ax.set_xticklabels(list(range(0, N)))

    if savefig:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()
