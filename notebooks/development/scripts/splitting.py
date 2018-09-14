to_track = []
result_futures = {}
start_knot = 15 #Must be unique number for every run on Cloudknot.

remote_folder = 'Tissue_Studies/09_11_18_Regional' #Folder in AWS S3 containing files to be analyzed
bucket = 'ccurtis.data'
vids = 15
types = ['PEG', 'PS']
pups = [2, 3]
slices = [1, 2, 3]
for typ in types:
    for pup in pups:
        for slic in slices:
            for num in range(1, vids+1):
                #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
                to_track.append('{}_P{}_S{}_XY{}'.format(typ, pup, slic, '%02d' % num))

import diff_classifier.knotlets as kn
for prefix in to_track[18:]:
    kn.split(prefix, remote_folder=remote_folder, bucket=bucket)