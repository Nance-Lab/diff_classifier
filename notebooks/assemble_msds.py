#!/usr/bin/env python

import sys
import diff_classifier.knotlets as kn

to_track = []
result_futures = {}

remote_folder = 'Tissue_Studies/09_11_18_Regional' #Folder in AWS S3 containing files to be analyzed
bucket = 'ccurtis.data'
vids = 15
pups = [2, 3]
slices = [1, 2, 3]
types = ['PS', 'PEG']
for typ in types:
    for pup in pups:
    	for slic in slices:
    	    for num in range(1, vids+1):
        	#to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
        	to_track.append('{}_P{}_S{}_XY{}'.format(typ, pup, slic, '%02d' % num))

for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    kn.assemble_msds(prefix, remote_folder, bucket=bucket)
    print('Successfully output msds for {}'.format(prefix))
