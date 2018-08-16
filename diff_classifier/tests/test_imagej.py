import os
import pytest
import sys
import tempfile

import numpy as np
import os.path as op
import diff_classifier.imagej as ij
from diff_classifier.utils import csv_to_pd

from urllib.request import urlretrieve

is_travis = "CI" in os.environ.keys()
is_mac = sys.platform == "darwin"


@pytest.mark.skipif(is_travis, reason="We're running this on Travis")
@pytest.mark.skipif(is_mac, reason="This doesn't work on Macs yet")
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
