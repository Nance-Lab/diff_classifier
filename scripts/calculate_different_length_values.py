import diff_classifier
import numpy as np
import pandas as pd
import os
import sys
import tqdm


data = pd.read_csv(sys.argv[1])

for track in data['Track_ID'].unique():
    track_data = diff_classifier.features.unmask_track(data[data['Track_ID'] == track])
    print('Track ID: {}'.format(track))

