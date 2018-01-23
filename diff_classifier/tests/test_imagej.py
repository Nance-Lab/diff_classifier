import diff_classifier.imagej as ij


def test_run_tracking():
    ij.track('http://fiji.sc/samples/FakeTracks.tif', 'foo.csv')
