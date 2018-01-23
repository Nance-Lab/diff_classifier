import diff_classifier.imagej as ij
import os.path as op
import tempfile
import pandas as pd
from diff_classifier.utils import csv_to_pd

def test_run_tracking():
    tf  = tempfile.NamedTemporaryFile(suffix='.csv')
    ij.track('http://fiji.sc/samples/FakeTracks.tif', tf.name)
    assert op.exists(tf.name)

    df = csv_to_pd(tf.name)
    assert df.shape == (84, 8)
