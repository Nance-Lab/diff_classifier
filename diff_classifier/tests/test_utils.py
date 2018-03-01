import tempfile
import pandas as pd
import diff_classifier.utils as ut
import sys
from io import StringIO
import pandas.util.testing as pdt

def test_csv_to_pd():
    tf = tempfile.NamedTemporaryFile(suffix=".csv")
    fid = open(tf.name, 'w')
    fid.write("This file won't work. \n This file won't work. \n This file won't work.")
    fid.close()
    
    stdout_ = sys.stdout 
    stream = StringIO()
    sys.stdout = stream
    test = ut.csv_to_pd(tf.name)
    sys.stdout = stdout_
    variable = stream.getvalue()
    test_string = 'No data in csv file.\n'
    assert(variable==test_string)
    
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
    pdt.assert_frame_equal(test, data)
    
    tf = tempfile.NamedTemporaryFile(suffix=".csv")
    fid = open(tf.name, 'w')
    fid.write('Found 0 tracks.\nData starts here.\nTrack_ID,Spot_ID,Frame,X,Y,Quality,SN_Ratio,Mean_Intensity\n')
    fid.close()
    
    stdout_ = sys.stdout 
    stream = StringIO()
    sys.stdout = stream
    test = ut.csv_to_pd(tf.name)
    sys.stdout = stdout_
    variable = stream.getvalue()
    test_string = ''
    assert(variable==test_string)
    pdt.assert_frame_equal(test, data)
    