import os
import pytest
import sys
import tempfile
import string

import numpy as np
import os.path as op
import diff_classifier.imagej as ij
from diff_classifier.utils import csv_to_pd

from urllib.request import urlretrieve

is_travis = "CI" in os.environ.keys()
is_mac = sys.platform == "darwin"


#@pytest.mark.skipif(is_travis, reason="We're running this on Travis")
#@pytest.mark.skipif(is_mac, reason="This doesn't work on Macs yet")
@pytest.mark.xfail
def test_run_tracking():
    tempf = tempfile.NamedTemporaryFile(suffix='.csv')
    ij.track('http://fiji.sc/samples/FakeTracks.tif', tempf.name)
    assert op.exists(tempf.name)

    df = csv_to_pd(tempf.name)
    assert df.shape == (84, 8)


def test_mean_intensity():
    fname = 'FakeTracks.tif'
    cwd = os.getcwd()
    fullname = os.path.join(cwd, fname)
    urlretrieve('http://fiji.sc/samples/FakeTracks.tif', filename=fullname)

    test = np.round(ij.mean_intensity(fname, frame=0), 1)
    assert test == 20.0


def test_partition_im():
    fname = 'FakeTracks.tif'
    cwd = os.getcwd()
    fullname = os.path.join(cwd, fname)
    urlretrieve('http://fiji.sc/samples/FakeTracks.tif', filename=fullname)

    rows = 2
    cols = 2
    names = ij.partition_im(fullname, irows=rows, icols=2, ores=(128, 128),
                            ires=(64, 64))
    for name in names:
        assert os.path.isfile(name)


def test_regress_sys():
    fname = 'FakeTracks.tif'
    cwd = os.getcwd()
    fullname = os.path.join(cwd, fname)
    urlretrieve('http://fiji.sc/samples/FakeTracks.tif', filename=fullname)

    all_videos = ['FakeTracks']*10
    yfit = [10, 9]
    training_size = 2

    tracks = ij.regress_sys(cwd, all_videos, yfit, training_size,
                            have_output=False, download=False)
    for track in tracks:
        assert track == 'FakeTracks'

    regress = ij.regress_sys(cwd, all_videos, yfit, training_size,
                             have_output=True, download=False)
    assert len(regress) == 8

    all_videos = list(string.ascii_lowercase)
    yfinal = ['e', 'b']

    tracks = ij.regress_sys(cwd, all_videos, yfit, training_size,
                            have_output=False, download=False)
    counter = 0
    for track in tracks:
        assert track == yfinal[counter]
        counter = counter + 1


def test_regress_tracking_params():
    fname = 'FakeTracks.tif'
    cwd = os.getcwd()
    fullname = os.path.join(cwd, fname)
    urlretrieve('http://fiji.sc/samples/FakeTracks.tif', filename=fullname)

    all_videos = ['FakeTracks']*10
    yfit = [10, 9]
    training_size = 2

    regress = ij.regress_sys(cwd, all_videos, yfit, training_size,
                             have_output=True, download=False)

    quality = ij.regress_tracking_params(regress, 'FakeTracks', frame=0)
    assert quality == 9.5
    quality = ij.regress_tracking_params(regress, 'FakeTracks', regmethod='SVR',
                                         frame=0)
    assert quality == 9.5
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='BayesianRidge', frame=0)
    assert quality == 9.5
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='SGDRegressor', frame=0)
    assert quality < -10000
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='LassoLars', frame=0)
    assert quality == 9.5
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='ARDRegression', frame=0)
    assert quality == 9.5
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='PassiveAggressiveRegressor',
                                         frame=0)
    assert np.round(quality, 1) == 9.1 or np.round(quality, 1) == 9.9
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='TheilSenRegressor', frame=0)
    assert quality == 9.5
    quality = ij.regress_tracking_params(regress, 'FakeTracks',
                                         regmethod='None', frame=0)
    assert quality == 3.0
