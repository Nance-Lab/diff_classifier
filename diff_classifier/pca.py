
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler as stscale
from sklearn.preprocessing import Imputer
import scipy.stats as stats
from scipy import stats, linalg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.

    Partial Correlation in Python (clone of Matlab's partialcorr)

    This uses the linear regression approach to compute the partial 
    correlation (might be slow for a huge number of variables). The 
    algorithm is detailed here:

        http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression

    Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
    the algorithm can be summarized as

        1) perform a normal linear least-squares regression with X as the target and Z as the predictor
        2) calculate the residuals in Step #1
        3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
        4) calculate the residuals in Step #3
        5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 

        The result is the partial correlation between X and Y while controlling for the effect of Z


    Date: Nov 2014
    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com

    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr


def kmo(dataset):
    """
    Calculates the Kaiser-Meyer-Olkin measure on an input dataset.
    
    Based on calculations shown here:
    
    http://www.statisticshowto.com/kaiser-meyer-olkin/
    
        -- 0.00-0.49  unacceptable
        -- 0.50-0.59  miserable
        -- 0.60-0.69  mediocre
        -- 0.70-0.79  middling
        -- 0.80-0.89  meritorious
        -- 0.90-1.00  marvelous
    
    Parameters
    ----------
    dataset : array-like, shape (n, p)
              Array containing n samples and p features. Must have no NaNs.
              Ideally scaled before performing test.
    
    Returns
    -------
    mo : KMO test value
    
    """
    
    #Correlation matrix and the partial covariance matrix.
    corrmatrix = np.corrcoef(dataset.transpose())
    pcorr = partial_corr(dataset)

    #Calculation of the KMO statistic
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

    mo = rij/(rij+uij)
    print(mo)
    return mo


def pca_analysis(dataset, dropcols=[], imputenans=True, scale=True, n_components=5):
    """
    Performs a primary component analysis on an input dataset
    
    Parameters
    ----------
    dataset : pandas dataframe of shape (n, p)
        Input dataset with n samples and p features
    dropcols : list
        Columns to exclude from pca analysis. At a minimum, user must exclude
        non-numeric columns.
    imputenans : boolean
        If True, impute NaN values as column means.
    scale : boolean
        If True, columns will be scaled to a mean of zero and a standard deviation of 1.
    n_components : integer
        Desired number of components in principle component analysis.
    
    Returns
    -------
    dataset_scaled : numpy array of shape (n, p)
        Scaled dataset with n samples and p features
    dataset_pca : Pandas dataframe of shape (n, n_components)
        Output array of n_component features of each original sample
    dataset_final : Pandas dataframe of shape (n, p+n_components)
        Output array with principle components append to original array.
    prcs : Pandas dataframe of shape (5, n_components)
        Output array displaying the top 5 features contributing to each
        principle component.
    prim_vals : Dictionary of lists
        Output dictionary of of the pca scores for the top 5 features
        contributing to each principle component.
    components : Pandas dataframe of shape (p, n_components)
        Raw pca scores.
        
    Examples
    --------
    
    """
    
    dataset_num = dataset.drop(dropcols, axis=1)
    dataset_raw = dataset.as_matrix()
    
    if imputenans:
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(dataset_raw)
        dataset_clean = imp.transform(dataset_raw)
    else:
        dataset_clean = dataset_raw
        
    if scale:
        scaler = stscale()
        scaler.fit(dataset_clean)
        dataset_scaled = scaler.transform(dataset_clean)
    else:
        dataset_scaled = dataset_clean
    
    pca1 = pca(n_components=n_components)
    pca1.fit(dataset_scaled)
    
    #Cumulative explained variance ratio
    x = 0
    explained_v = pca1.explained_variance_ratio_
    print('Cumulative explained variance:')
    for i in range(0, n_components):
        x = x + explained_v[i]
        print('{} component: {}'.format(i, x))
    
    prim_comps = {}
    prim_vals = {}
    comps = pca1.components_
    components = pd.DataFrame(comps.transpose())

    for num in range(0, n_components):
        highest = np.abs(components[num]).as_matrix().argsort()[-5:][::-1]
        pels = []
        prim_vals[num] = components[num].as_matrix()[highest]
        for col in highest:
            pels.append(dataset.columns[col])
        prim_comps[num] = pels
    
    #Main contributors to each primary component
    prcs = pd.DataFrame.from_dict(prim_comps)
    
    dataset_pca = pd.DataFrame(pca1.transform(dataset_scaled))
    dataset_final = pd.concat([dataset, dataset_pca], axis=1)
    
    return dataset_scaled, dataset_pca, dataset_final, prcs, prim_vals, components


def plot_pca(datasets, figsize=(8, 8), lwidth=8.0,
             labels = ['Sample1', 'Sample2'], savefig=True, filename='test.png'):
    
    """
    Plots the average output features from a PCA analysis in polar coordinates.
    
    Parameters
    ----------
    datasets : dictionary (keys = n) of numpy arrays of shape p
        Dictionary with n samples and p features to plot.
    figize : list
        Dimensions of output figure e.g. (8, 8)
    lwidth : float
        Width of plotted lines in figure
    labels : list of string
        Labels to display in legend.
    savefig : boolean
        If True, saves figure
    filename : string
        Desired output filename
        
    Returns
    -------
    
    """

    fig = plt.figure(figsize=figsize)
    for key in datasets:
        N = datasets[key].shape[0]
    width = (2*np.pi) / N
    color=iter(cm.viridis(np.linspace(0,1,N)))
    
    theta = np.linspace(0.0, 2 * np.pi, N+1, endpoint=True)
    radii = {}
    bars = {}
    
    ax = plt.subplot(111, polar=True)
    counter = 0
    for key in datasets:
        c=next(color)
        radii[key] = np.append(datasets[key], datasets[key][0]) 
        bars[key] = ax.plot(theta, radii[key], linewidth=lwidth, color=c, label=labels[counter])
        counter = counter + 1
    plt.legend(bbox_to_anchor=(0.90, 1), loc=2, borderaxespad=0., frameon=False, fontsize=20)

    # # Use custom colors and opacity
    # for r, bar in zip(radii, bars):
    #     bar.set_facecolor(plt.cm.jet(np.abs(r / 2.5)))
    #     bar.set_alpha(0.8)
    ax.set_xticks(np.pi/180. * np.linspace(0, 360, N, endpoint=False))
    ax.set_xticklabels(list(range(0, N)))
    
    if savefig:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.show()