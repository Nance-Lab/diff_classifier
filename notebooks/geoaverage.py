#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn
import numpy as np
import diff_classifier.aws as aws
import diff_classifier.msd as msd

folder = '08_28_18_varying_PEG_redo'
bucket = 'evanepst.data'
#experiment = 'test' #Used for naming purposes. Should exclude XY and well information

#vids = 2
to_track = []
frames = 651
fps = 100.02
umppx = 0.07

vids = 10
covers = ['COOH', 'pt10xs', 'pt15xs', 'pt25xs', 'pt40xs']
for cover in covers:
    for num in range(1, vids+1):
        #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
        to_track.append('3mM_100_{}_XY{}'.format(cover, '%02d' % num))

geomean = {}
gSEM = {}
for sample_name in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    # Users can toggle between using pre-calculated geomean files and calculating new values by commenting out the relevant
    # lines of code within the for loop.
    #aws.download_s3('{}/geomean_{}.csv'.format(folder, sample_name), 'geomean_{}.csv'.format(sample_name), bucket_name=bucket)
    #aws.download_s3('{}/geoSEM_{}.csv'.format(folder, sample_name), 'geoSEM_{}.csv'.format(sample_name), bucket_name=bucket)
    #geomean[sample_name] = np.genfromtxt('geomean_{}.csv'.format(sample_name))
    #gSEM[sample_name] = np.genfromtxt('geoSEM_{}.csv'.format(sample_name))

    aws.download_s3('{}/msd_{}.csv'.format(folder, sample_name), 'msd_{}.csv'.format(sample_name), bucket_name=bucket)
    geomean[sample_name], gSEM[sample_name] = msd.geomean_msdisp(sample_name, umppx=umppx, fps=fps,
                                                                 remote_folder=folder, bucket=bucket)
    print('Done with {}'.format(sample_name))
