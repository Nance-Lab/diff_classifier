#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = '1_7_19_P01_region_dependent_MPT' #Folder in AWS S3 containing files to be analyzed
bucket = 'mckenna.data'
vids = 5
inflams = ['PAM']
hemis = ['contra', 'ipsi']
regions = ['cc', 'cortex']

for inflam in inflams:
    for hemi in hemis:
        for region in regions:
            for num in range(1, vids+1):
                #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
                to_track.append('{}_{}_{}_vid_{}'.format(inflam, hemi, region, '%01d' % num))

#to_track = [ '100x_0_4_0_6_gel_0_6_bulk_vid_5',
# 	     '100x_0_4_1_2_gel_0_4_bulk_vid_3']

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.split(prefix, remote_folder, bucket=bucket)
    print('Successfully output subimages for {}'.format(prefix))

#kn.assemble_msds(sys.argv[1], remote_folder, bucket=bucket)
