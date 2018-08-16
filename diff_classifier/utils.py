"""Utility functions used throughout diff_classifier.

This module includes general functions for tasks such as importing files and
converting between data types. Currently only includes a function to generate
pandas dataframes for csv output from Trackmate.

"""
import pandas as pd


def csv_to_pd(csvfname):
    """Reads Trackmate csv output file and converts to pandas dataframe.

    A specialized function designed specifically for TrackMate output files.
    This edits out the header at the beginning of the file.

    Parameters
    ----------
    csvfname : string
        Output csv from a file similar to trackmate_template.  Must
        include line 'Data starts here.\n' line in order to parse correctly.

    Returns
    -------
    data : pandas DataFrame
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

        partids = data.Track_ID.unique()
        counter = 0
        for partid in partids:
            data.loc[data.Track_ID == partid, 'Track_ID'] = counter
            counter = counter + 1
    except:
        print('No data in csv file.')
        rawd = {'Track_ID': [],
                'Spot_ID': [],
                'Frame': [],
                'X': [],
                'Y': [],
                'Quality': [],
                'SN_Ratio': [],
                'Mean_Intensity': []}
        cols = ['Track_ID', 'Spot_ID', 'Frame', 'X', 'Y', 'Quality', 'SN_Ratio', 'Mean_Intensity']
        data = pd.DataFrame(data=rawd, index=[])
        data = data[cols]
        data = data.astype('float64')

    return data
