import pandas as pd
import numpy as np
import skimage.io as sio


def csv_to_pd(csvfname):
    """
    csv_to_pd(csvfname)

    Reads Trackmate csv output file and converts to a pandas dataframe.

    Parameters
    ----------
    csvfname : string
        Output csv from a file similar to trackmate_template.  Must
        include line 'Data starts here.\n' line in order to parse correctly.

    Returns
    -------
    data : pandas dataframe
        Contains all trajectories from csvfname.

    Examples
    --------
    >>> data = csv_to_pd('../data/test.csv')

    """
    csvfile = open(csvfname)

    try:
        line = 'test'
        counter = 0
        while line != 'Data starts here.\n':
            line = csvfile.readline()
            counter = counter + 1
            if counter > 2000:
                break

        data = pd.read_csv(csvfname, skiprows=counter)
        data.sort_values(['Track_ID', 'Frame'], ascending=[1, 1])
        data = data.astype('float64')

        part_IDs = data.Track_ID.unique()
        counter = 0
        for ID in part_IDs:
            data.loc[data.Track_ID == ID, 'Track_ID'] = counter
            counter = counter + 1
    except:
        print('No data in csv file.')
        d = {'Track_ID': [],
             'Spot_ID': [],
             'Frame': [],
             'X': [],
             'Y': [],
             'Quality': [],
             'SN_Ratio': [],
             'Mean_Intensity': []}
        cols = ['Track_ID', 'Spot_ID', 'Frame', 'X', 'Y', 'Quality', 'SN_Ratio', 'Mean_Intensity']
        data = pd.DataFrame(data=d, index=[])
        data = data[cols]
        data = data.astype('float64')

    return data
