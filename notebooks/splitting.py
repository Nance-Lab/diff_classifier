#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = 'Gel_Studies/11_21_18_coverage' #Folder in AWS S3 containing files to be analyzed
bucket = 'ccurtis.data'
vids = 20
types = ['PSCOOH', 'p0_1', 'p0_2', 'p0_4', 'p0_5', 'p0_6', 'p0_75', 'p1']
for typ in types:
    for num in range(1, vids+1):
        #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
        to_track.append('{}_XY{}'.format(typ, '%02d' % num))

#to_track = [ '100x_0_4_0_6_gel_0_6_bulk_vid_5',
# 	     '100x_0_4_1_2_gel_0_4_bulk_vid_3']

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.split(prefix, remote_folder, bucket=bucket)
    print('Successfully output subimages for {}'.format(prefix))

#kn.assemble_msds(sys.argv[1], remote_folder, bucket=bucket)
