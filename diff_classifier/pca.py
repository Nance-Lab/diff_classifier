"""Performs principle component analysis on input datasets.

This module performs principle component analysis on input datasets using
functions from scikit-learn. It is optimized to data formats used in
diff_classifier, but can potentially be extended to other applications.

"""

import random
import pandas as pd
import numpy as np
from scipy import stats, linalg
import seaborn as sns
from sklearn import neighbors
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler as stscale
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D


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
    matrix = np.multiply(corrmatrix, corrmatrix)
    rows = matrix.shape[0]
    cols = matrix.shape[1]
    rij = np.sum(matrix) - np.trace(matrix)
    uij = np.sum(pcorr) - np.trace(pcorr)
    kmostat = rij/(rij+uij)
    print(kmostat)
    return kmostat


def pca_analysis(dataset, dropcols=[], imputenans=True, scale=True,
                 rem_outliers=True, out_thresh=10, n_components=5):
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
    pd.options.mode.chained_assignment = None  # default='warn'
    dataset_num = dataset.drop(dropcols, axis=1)

    if rem_outliers:
        for i in range(10):
            for col in dataset_num.columns:
                xmean = np.mean(dataset_num[col])
                xstd = np.std(dataset_num[col])

                counter = 0
                for x in dataset_num[col]:
                    if x > xmean + out_thresh*xstd:
                        dataset[col][counter] = np.nan
                        dataset_num[col][counter] = np.nan
                    if x < xmean - out_thresh*xstd:
                        dataset[col][counter] = np.nan
                        dataset_num[col][counter] = np.nan
                    counter = counter + 1

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
    pcadataset.pcamodel = pca1

    return pcadataset


def recycle_pcamodel(pcamodel, df, imputenans=True, scale=True):
    if imputenans:
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df_clean = imp.transform(df)
    else:
        df_clean = df
        
    # Scale inputs
    if scale:
        scaler = stscale()
        scaler.fit(df_clean)
        df_scaled = scaler.transform(df_clean)
    else:
        df_scaled = df_clean
        
    pcamodel.fit(df_scaled)
    pcavals = pd.DataFrame(pcamodel.transform(df_scaled))
    pcafinal = pd.concat([df, pcavals], axis=1)
    
    return pcafinal


def plot_pca(datasets, figsize=(8, 8), lwidth=8.0,
             labels=['Sample1', 'Sample2'], savefig=True, filename='test.png',
             rticks=np.linspace(-2, 2, 5)):
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
    color = iter(cm.viridis(np.linspace(0, 0.9, len(datasets))))

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
    ax.set_ylim([min(rticks), max(rticks)])
    ax.set_yticks(rticks)

    if savefig:
        plt.savefig(filename, bbox_inches='tight')

    plt.show()


def build_model(rawdata, feature, featvals, equal_sampling=True,
                    tsize=20, from_end=True, input_cols=6, model='KNN',
                    **kwargs):
    """Builds a K-nearest neighbor model using an input dataset.

    Parameters
    ----------
    rawdata : pandas.core.frames.DataFrame
        Raw dataset of n samples and p features.
    feature : string or int
        Feature in rawdata containing output values on which KNN
        model is to be based.
    featvals : string or int
        All values that feature can take.
    equal_sampling : bool
        If True, training dataset will contain an equal number
        of samples that take each value of featvals. If false,
        each sample in training dataset will be taken randomly
        from rawdata.
    tsize : int
        Size of training dataset. If equal_sampling is False,
        training dataset will be exactly this size. If True,
        training dataset will contain N x tsize where N is the
        number of unique values in featvals.
    n_neighbors : int
        Number of nearest neighbors to be used in KNN
        algorithm.
    from_end : int
        If True, in_cols will select features to be used as
        training data defined end of rawdata e.g.
        rawdata[:, -6:]. If False, input_cols will be read
        as a tuple e.g. rawdata[:, 10:15].
    input_col : int or tuple
        Defined in from_end above.

    Returns
    -------
    clf : sklearn.neighbors.classification.KNeighborsClassifier
        KNN model
    X : numpy.ndarray
        training input dataset used to create clf
    y : numpy.ndarray
        training output dataset used to create clf

    """

    defaults = {'n_neighbors': 5, 'NNsolver': 'lbfgs', 'NNalpha': 1e-5,
                'NNhidden_layer': (5, 2), 'NNrandom_state': 1,
                'n_estimators': 10}

    for defkey in defaults.keys():
        if defkey not in kwargs.keys():
            kwargs[defkey] = defaults[defkey]
    
    if equal_sampling:
        for featval in featvals:
            if from_end:
                test = rawdata[rawdata[feature] == featval
                               ].values[:, -input_cols:]
            else:
                test = rawdata[rawdata[feature] == featval
                               ].values[:, input_cols[0]:input_cols[1]]
            to_plot = np.array(random.sample(range(0, test.shape[0]
                                                   ), tsize))
            if featval == featvals[0]:
                X = test[to_plot, :]
                y = rawdata[rawdata[feature] == featval
                            ][feature].values[to_plot]
            else:
                X = np.append(X, test[to_plot, :], axis=0)
                y = np.append(y, rawdata[rawdata[feature] == featval
                                         ][feature].values[to_plot], axis=0)

    else:
        if from_end:
            test = rawdata.values[:, -input_cols:]
        else:
            test = rawdata.values[:, input_cols[0]:input_cols[1]]
        to_plot = np.array(random.sample(range(0, test.shape[0]), tsize))
        X = test[to_plot, :]
        y = rawdata[feature].values[to_plot]

    if model is 'KNN':
        clf = neighbors.KNeighborsClassifier(kwargs['n_neighbors'])
    elif model is 'MLP':
        clf = MLPClassifier(solver=kwargs['NNsolver'], alpha=kwargs['NNalpha'],
                            hidden_layer_sizes=kwargs['NNhidden_layer'],
                            random_state=kwargs['NNrandom_state'])
    else:
        clf = RandomForestClassifier(n_estimators=kwargs['n_estimators'])
    
    clf.fit(X, y)

    return clf, X, y


def predict_model(model, X, y):
    """Calculates fraction correctly predicted using input KNN
    model

    Parameters
    ----------
    model : sklearn.neighbors.classification.KNeighborsClassifier
        KNN model
    X : numpy.ndarray
        training input dataset used to create clf
    y : numpy.ndarray
        training output dataset used to create clf

    Returns
    -------
    pcorrect : float
        Fraction of correctly predicted outputs using the
        input KNN model and the input test dataset X and y

    """
    yp = model.predict(X)
    correct = np.zeros(y.shape[0])
    for i in range(0, y.shape[0]):
        if y[i] == yp[i]:
            correct[i] = 1

    pcorrect = np.average(correct)
    # print(pcorrect)
    return pcorrect


def feature_violin(df, label='label', lvals=['yes', 'no'], fsubset=3, **kwargs):
    """Creates violinplot of input feature dataset

    Designed to plot PCA components from pca_analysis.

    Parameters
    ----------
    df : pandas.core.frames.DataFrame
        Must contain a group name column, and numerical feature columns.
    label : string or int
        Name of group column.
    lvals : list of string or int
        All values that group column can take
    fsubset : int or list of int
        Features to be plotted. If integer, will plot range(fsubset).
        If list, will only plot features contained in fsubset.
    **kwargs : variable
        figsize : tuple of int or float
            Dimensions of output figure
        yrange : list of int or float
            Range of y axis
        xlabel : string
            Label of x axis
        labelsize : int or float
            Font size of x label
        ticksize : int or float
            Font size of y tick labels
        fname : None or string
            Name of output file
        legendfontsize : int or float
            Font size of legend
        legendloc : int
            Location of legend in plot e.g. 1, 2, 3, 4

    """

    defaults = {'figsize': (12, 5), 'yrange': [-20, 20], 'xlabel': 'Feature',
                'labelsize': 20, 'ticksize': 16, 'fname': None,
                'legendfontsize': 12, 'legendloc': 1}

    for defkey in defaults.keys():
        if defkey not in kwargs.keys():
            kwargs[defkey] = defaults[defkey]

    # Restacking input data
    groupsize = []
    featcol = []
    valcol = []
    feattype = []

    if isinstance(fsubset, int):
        frange = range(fsubset)
    else:
        frange = fsubset

    for feat in frange:
        groupsize.extend(df[label].values)
        featcol.extend([feat]*df[label].values.shape[0])
        valcol.extend(df[feat].values)

    to_violind = {'label': groupsize, 'Feature': featcol,
                  'Feature Value': valcol}
    to_violin = pd.DataFrame(data=to_violind)

    # Plotting function
    fig, ax = plt.subplots(figsize=kwargs['figsize'])
    sns.violinplot(x="Feature", y="Feature Value", hue="label", data=to_violin,
                   palette="Pastel1", hue_order=lvals,
                   figsize=kwargs['figsize'])

    # kwargs
    ax.tick_params(axis='both', which='major', labelsize=kwargs['ticksize'])
    plt.xlabel(kwargs['xlabel'], fontsize=kwargs['labelsize'])
    plt.ylabel('', fontsize=kwargs['labelsize'])
    plt.ylim(kwargs['yrange'])
    plt.legend(loc=kwargs['legendloc'], prop={'size': kwargs['legendfontsize']})
    if kwargs['fname'] is None:
        plt.show()
    else:
        plt.savefig(kwargs['fname'])

    return to_violin


def feature_plot_2D(dataset, label, features=[0, 1], lvals=['PEG', 'PS'],
                    randsel=True, randcount=200, **kwargs):
    """Plots two features against each other from feature dataset.

    Parameters
    ----------
    dataset : pandas.core.frames.DataFrame
        Must comtain a group column and numerical features columns
    labels : string or int
        Group column name
    features : list of int
        Names of columns to be plotted
    randsel : bool
        If True, downsamples from original dataset
    randcount : int
        Size of downsampled dataset
    **kwargs : variable
        figsize : tuple of int or float
            Size of output figure
        dotsize : float or int
            Size of plotting markers
        alpha : float or int
            Transparency factor
        xlim : list of float or int
            X range of output plot
        ylim : list of float or int
            Y range of output plot
        legendfontsize : float or int
            Font size of legend
        labelfontsize : float or int
            Font size of labels
        fname : string
            Filename of output figure

    Returns
    -------
    xy : list of lists
        Coordinates of data on plot

    """
    defaults = {'figsize': (8, 8), 'dotsize': 70, 'alpha': 0.7, 'xlim': None,
                'ylim': None, 'legendfontsize': 12, 'labelfontsize': 20,
                'fname': None}

    for defkey in defaults.keys():
        if defkey not in kwargs.keys():
            kwargs[defkey] = defaults[defkey]

    tgroups = {}
    xy = {}
    counter = 0
    labels = dataset[label].unique()
    for lval in lvals:
        tgroups[counter] = dataset[dataset[label] == lval]
        counter = counter + 1

    N = len(tgroups)
    color = iter(cm.viridis(np.linspace(0, 0.9, N)))

    fig = plt.figure(figsize=kwargs['figsize'])
    ax1 = fig.add_subplot(111)
    counter = 0
    for key in tgroups:
        c = next(color)
        xy = []
        if randsel:
            to_plot = random.sample(range(0, len(tgroups[key][0].tolist())),
                                    randcount)
            for key2 in features:
                xy.append(list(tgroups[key][key2].tolist()[i] for i in to_plot))
        else:
            for key2 in features:
                xy.append(tgroups[key][key2])
        ax1 = plt.scatter(xy[0], xy[1], c=c, s=kwargs['dotsize'],
                          alpha=kwargs['alpha'], label=labels[counter])
        counter = counter + 1

    if kwargs['xlim'] is not None:
        plt.xlim(kwargs['xlim'])
    if kwargs['ylim'] is not None:
        plt.ylim(kwargs['ylim'])

    plt.legend(fontsize=kwargs['legendfontsize'], frameon=False)
    plt.xlabel('Prin. Component {}'.format(features[0]),
               fontsize=kwargs['labelfontsize'])
    plt.ylabel('Prin. Component {}'.format(features[1]),
               fontsize=kwargs['labelfontsize'])

    if kwargs['fname'] is None:
        plt.show()
    else:
        plt.savefig(kwargs['fname'])

    return xy


def feature_plot_3D(dataset, label, features=[0, 1, 2], lvals=['PEG', 'PS'],
                    randsel=True, randcount=200, **kwargs):
    """Plots three features against each other from feature dataset.

    Parameters
    ----------
    dataset : pandas.core.frames.DataFrame
        Must comtain a group column and numerical features columns
    labels : string or int
        Group column name
    features : list of int
        Names of columns to be plotted
    randsel : bool
        If True, downsamples from original dataset
    randcount : int
        Size of downsampled dataset
    **kwargs : variable
        figsize : tuple of int or float
            Size of output figure
        dotsize : float or int
            Size of plotting markers
        alpha : float or int
            Transparency factor
        xlim : list of float or int
            X range of output plot
        ylim : list of float or int
            Y range of output plot
        zlim : list of float or int
            Z range of output plot
        legendfontsize : float or int
            Font size of legend
        labelfontsize : float or int
            Font size of labels
        fname : string
            Filename of output figure

    Returns
    -------
    xy : list of lists
        Coordinates of data on plot

    """

    defaults = {'figsize': (8, 8), 'dotsize': 70, 'alpha': 0.7, 'xlim': None,
                'ylim': None, 'zlim': None, 'legendfontsize': 12,
                'labelfontsize': 10, 'fname': None}

    for defkey in defaults.keys():
        if defkey not in kwargs.keys():
            kwargs[defkey] = defaults[defkey]

    axes = {}
    fig = plt.figure(figsize=(14, 14))
    axes[1] = fig.add_subplot(221, projection='3d')
    axes[2] = fig.add_subplot(222, projection='3d')
    axes[3] = fig.add_subplot(223, projection='3d')
    axes[4] = fig.add_subplot(224, projection='3d')
    color = iter(cm.viridis(np.linspace(0, 0.9, 3)))
    angle1 = [60, 0, 0, 0]
    angle2 = [240, 240, 10, 190]

    tgroups = {}
    xy = {}
    counter = 0
    #labels = dataset[label].unique()
    for lval in lvals:
        tgroups[counter] = dataset[dataset[label] == lval]
        #print(lval)
        #print(tgroups[counter].shape)
        counter = counter + 1

    N = len(tgroups)
    color = iter(cm.viridis(np.linspace(0, 0.9, N)))

    counter = 0
    for key in tgroups:
        c = next(color)
        xy = []
        if randsel:
            #print(range(0, len(tgroups[key][0].tolist())))
            to_plot = random.sample(range(0, len(tgroups[key][0].tolist())),
                                    randcount)
            for key2 in features:
                xy.append(list(tgroups[key][key2].tolist()[i] for i in to_plot))
        else:
            for key2 in features:
                xy.append(tgroups[key][key2])

        acount = 0
        for ax in axes:
            axes[ax].scatter(xy[0], xy[1], xy[2], c=c, s=kwargs['dotsize'], alpha=kwargs['alpha'])#, label=labels[counter])
            if kwargs['xlim'] is not None:
                axes[ax].set_xlim3d(kwargs['xlim'][0], kwargs['xlim'][1])
            if kwargs['ylim'] is not None:
                axes[ax].set_ylim3d(kwargs['ylim'][0], kwargs['ylim'][1])
            if kwargs['zlim'] is not None:
                axes[ax].set_zlim3d(kwargs['zlim'][0], kwargs['zlim'][1])
            axes[ax].view_init(angle1[acount], angle2[acount])
            axes[ax].set_xlabel('Prin. Component {}'.format(features[0]),
                                fontsize=kwargs['labelfontsize'])
            axes[ax].set_ylabel('Prin. Component {}'.format(features[1]),
                                fontsize=kwargs['labelfontsize'])
            axes[ax].set_zlabel('Prin. Component {}'.format(features[2]),
                                fontsize=kwargs['labelfontsize'])
            acount = acount + 1
        counter = counter + 1

    # plt.legend(fontsize=kwargs['legendfontsize'], frameon=False)
    axes[3].set_xticks([])
    axes[4].set_xticks([])

    if kwargs['fname'] is None:
        plt.show()
    else:
        plt.savefig(kwargs['fname'])
