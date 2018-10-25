#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = '10_05_18_coverage' #Folder in AWS S3 containing files to be analyzed
bucket = 'evanepst.data'
vids = 10
pups = [2, 3]
types = ['0_10xs', '0_15xs', '0_20xs', '0_25xs', '0_40xs', '0_50xs', '0_60xs', '0_75xs', '1xs', 'PSCOOH']
for typ in types:
    for num in range(1, vids+1):
        #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
        to_track.append('5mM_{}_XY{}'.format(typ, '%02d' % num))

#to_track = [ '100x_0_4_0_6_gel_0_6_bulk_vid_5',
# 	     '100x_0_4_1_2_gel_0_4_bulk_vid_3']

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.assemble_msds(prefix, remote_folder, bucket=bucket)
    print('Successfully output msds for {}'.format(prefix))

#kn.assemble_msds(sys.argv[1], remote_folder, bucket=bucket)
