from __future__ import print_function
import collections
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd


# pylint: disable=E0401
# pylint: disable=W0105
# pylint: disable=C0103
# pylint: disable=R0914
# pylint: disable=R0915
# pylint: disable=E501
#
if __name__ == '__main__':
    """
    In order to run the following script by itself, output trajectory data into a csv file and call function as:
    `python trajectory_RNN --dataset traj.csv --output outputfile`
    """
    import argparse
    import os

    # construct the argument parse and parse the arguments
    # will output args['dataset'] and args['output']
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to input dataset")
    # ap.add_argument("-m", "--output", required=True,
                    # help="path to output model file")
    args = vars(ap.parse_args())

    print("[INFO] describing dataset...")
    trajectoryPaths = args['dataset']
    # initialize the data matrix and labels list
    print(trajectoryPaths)
    data = []
    labels = []

    """
    --------------------------------------------------------------------------
    Now that we have our `imagePaths`, we can loop over them individually,
    load them from disk, convert the images to feature vectors, and the update
    the data  and `labels` lists:
    --------------------------------------------------------------------------
    """
    """
    for (i, trajectoryPath) in enumerate(trajectoryPaths):
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        trajectory = cv2.imread(trajectoryPath)
        label = trajectoryPath.split(os.path.sep)[-1].split(".")[0]
        # construct a feature vector raw pixel intensities, then update
        # the data matrix and labels list
        features = image_to_feature_vector(trajectory)
        data.append(features)
        labels.append(label)
        # show an update every 1,000 images
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {}/{}".format(i, len(trajectoryPaths)))
    """

