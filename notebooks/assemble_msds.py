#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = '9_5_18_Gel_Interface_Vids' #Folder in AWS S3 containing files to be analyzed
bucket = 'mckenna.data'
vids = 3
visc = ['0_4', '1_2']
for vis in visc:
    for num in range(1, vids+1):
        #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
        to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
for num in range(1, 2*vids+1):
    to_track.append('100x_0_4_1_2_gel_interface_vid_{}'.format(num))

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.assemble_msds(prefix, remote_folder, bucket='mckenna.data')
    print('Successfully output msds for {}'.format(prefix))