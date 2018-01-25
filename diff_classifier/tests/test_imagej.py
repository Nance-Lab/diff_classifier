import os
import pytest
import sys
import diff_classifier.imagej as ij
import os.path as op
import tempfile
from diff_classifier.utils import csv_to_pd

is_travis = "CI" in os.environ.keys()
is_mac = sys.platform == "darwin"

@pytest.mark.skipif(is_travis, reason="We're running this on Travis")
@pytest.mark.skipif(is_mac, reason="This doesn't work on Macs yet")
def test_run_tracking():
    tf  = tempfile.NamedTemporaryFile(suffix='.csv')
    ij.track('http://fiji.sc/samples/FakeTracks.tif', tf.name)
    assert op.exists(tf.name)

    df = csv_to_pd(tf.name)
    assert df.shape == (84, 8)
