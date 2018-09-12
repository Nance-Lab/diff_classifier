#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = '08_28_18_varying_PEG_redo' #Folder in AWS S3 containing files to be analyzed
bucket = 'evanepst.data'
vids = 10
covers = ['COOH', 'pt10xs', 'pt15xs', 'pt25xs', 'pt40xs']
for cover in covers:
    for num in range(1, vids+1):
        #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
        to_track.append('3mM_100_{}_XY{}'.format(cover, '%02d' % num))

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.assemble_msds(prefix, remote_folder, bucket=bucket)
    print('Successfully output msds for {}'.format(prefix))