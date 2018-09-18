#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = 'Gel_studies' #Folder in AWS S3 containing files to be analyzed
bucket = 'dtoghani.data'
vids = 10
pups = [2, 3]
types = ['10k_PEG', '5k_PEG', '1k_PEG', '5k_PEG_NH2', 'PS_NH2', 'PS_COOH']
for typ in types:
    for pup in pups:
    	for num in range(1, vids+1):
            #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
            to_track.append('{}_{}mM_XY{}'.format(typ, pup, '%02d' % num))

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.assemble_msds(prefix, remote_folder, bucket=bucket)
    print('Successfully output msds for {}'.format(prefix))
