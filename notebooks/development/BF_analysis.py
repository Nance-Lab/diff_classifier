import sys
import matplotlib
matplotlib.use('Agg')
import skimage.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diff_classifier.aws as aws
from skimage.filters import roberts, sobel, scharr, prewitt, median, rank
from skimage import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, disk, reconstruction
from scipy.ndimage.morphology import distance_transform_edt as EuclideanTransform
from operator import itemgetter

to_track = []
result_futures = {}

remote_folder = 'Cell_Studies/10_16_18_cell_study' #Folder in AWS S3 containing files to be analyzed
bucket = 'ccurtis.data'
vids = 5
types = ['PS', 'PEG']
slices = [1, 2]

for typ in types:
    for slic in slices:
        for num in range(1, vids+1):
            #to_track.append('100x_0_4_1_2_gel_{}_bulk_vid_{}'.format(vis, num))
            to_track.append('{}_{}_XY{}'.format(typ, slic, num))


def binary_BF(image, meanse=disk(10), edgefilt='prewitt', opense=disk(10),
          fill_first=False, bi_thresh=0.000025, tophatse=disk(20)):
    
    #convertim = img_as_ubyte(image)
    meanim = rank.mean(image, meanse)
    if edgefilt is 'prewitt':
        edgeim = prewitt(meanim)
    elif edgefilt is 'sobel':
        edgeim = sobel(meanim)
    elif edgefilt is 'scharr':
        edgeim = scharr(meanim)
    elif edgefilt is 'roberts':
        edgeim = roberts(meanim)
    
    closeim = closing(edgeim, opense)
    openim = opening(closeim, opense)
    if fill_first:
        seed = np.copy(openim)
        seed[1:-1, 1:-1] = openim.max()
        mask = openim
        filledim = reconstruction(seed, mask, method='erosion')
        binarim = filledim > bi_thresh
    else:
        binarim = openim > bi_thresh*np.mean(openim)
        seed = np.copy(binarim)
        seed[1:-1, 1:-1] = binarim.max()
        mask = binarim
        filledim = reconstruction(seed, mask, method='erosion')

    tophim = filledim - closing(white_tophat(filledim, tophatse), opense)>0.01

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    ax[0][0].imshow(image, cmap='gray')
    ax[0][1].imshow(meanim, cmap='gray')
    ax[0][2].imshow(edgeim, cmap='gray', vmax=4*np.mean(edgeim))
    ax[0][3].imshow(closeim, cmap='gray', vmax=4*np.mean(closeim))
    ax[1][0].imshow(openim, cmap='gray', vmax=4*np.mean(openim))
    ax[1][1].imshow(binarim, cmap='gray')
    ax[1][2].imshow(filledim, cmap='gray')
    ax[1][3].imshow(tophim, cmap='gray')
    for axes in ax:
        for axe in axes:
            axe.axis('off')
    fig.tight_layout()
    
    return tophim


def BF_cell_features(prefix, folder, bucket='ccurtis.data'):
    
    ffilename = 'features_{}.csv'.format(prefix)
    mfilename = 'msd_{}.csv'.format(prefix)
    bffilename = 'BF_cells_{}.tif'.format(prefix)
    biim = 'bi_BF_cells_{}.tif'.format(prefix)
    bimages = 'biproc_BF_cells_{}.png'.format(prefix)
    
    aws.download_s3('{}/{}'.format(folder, ffilename), ffilename, bucket_name=bucket)
    aws.download_s3('{}/{}'.format(folder, mfilename), mfilename, bucket_name=bucket)
    aws.download_s3('{}/{}'.format(folder, bffilename), bffilename, bucket_name=bucket)
    print('Successfully downloaded files')
    
    fstats = pd.read_csv(ffilename, encoding = "ISO-8859-1")
    msds = pd.read_csv(mfilename, encoding = "ISO-8859-1")
    bfimage = plt.imread(bffilename)
    tophimage = binary_BF(bfimage, opense=disk(12), bi_thresh=1.2, tophatse=disk(20))
    plt.savefig(bimages)
    euimage = EuclideanTransform(tophimage)+EuclideanTransform(~tophimage)
    print('Successfully performed image processing')
    
    xa = -np.reshape(np.clip((fstats.Y.values-1).astype(int), a_min=0, a_max=2043), newshape=(fstats.Y.shape[0], 1))
    ya = np.reshape(np.clip((fstats.X.values-1).astype(int), a_min=0, a_max=2043), newshape=(fstats.X.shape[0], 1))
    xya = [tuple(l) for l in np.concatenate((xa, ya), axis=1).tolist()]
    fstats['Cell Status'] = itemgetter(*xya)(tophimage)
    fstats['Cell Distance'] = itemgetter(*xya)(euimage)

    print('Successfully calculated Cell Status Params')
    
    frames = 651
    xb = -np.reshape(np.clip((msds.Y.values-1).astype(int), a_min=0, a_max=2043), newshape=(int(msds.Y.shape[0]), 1))
    yb = np.reshape(np.clip((msds.X.values-1).astype(int), a_min=0, a_max=2043), newshape=(int(msds.X.shape[0]), 1))
    xyb = [tuple(l) for l in np.concatenate((xb, yb), axis=1).tolist()]
    msds['Cell Status'] = itemgetter(*xyb)(tophimage)
    msds['Cell Distance'] = itemgetter(*xyb)(euimage)
    
    msds_cell_status = np.reshape(msds['Cell Status'].values, newshape=(int(msds.X.shape[0]/frames), frames))
    msds_cell_distance = np.reshape(msds['Cell Distance'].values, newshape=(int(msds.X.shape[0]/frames), frames))
    fstats['Membrane Xing'] = np.sum(np.diff(msds_cell_status, axis=1) == True, axis=1)
    fstats['Distance Towards Cell'] = np.sum(np.diff(msds_cell_distance, axis=1), axis=1)
    fstats['Percent Towards Cell'] = np.mean(np.diff(msds_cell_distance, axis=1) > 0, axis=1)
    print('Successfully calculated Membrane Xing Params')
    
    fstats.to_csv(ffilename, sep=',', encoding = "ISO-8859-1")
    msds.to_csv(mfilename, sep=',', encoding = "ISO-8859-1")
    plt.imsave(biim, tophimage, cmap='gray')
    
    aws.upload_s3(ffilename, '{}/{}'.format(folder, ffilename), bucket_name=bucket)
    aws.upload_s3(mfilename, '{}/{}'.format(folder, mfilename), bucket_name=bucket)
    aws.upload_s3(biim, '{}/{}'.format(folder, biim), bucket_name=bucket)
    aws.upload_s3(bimages, '{}/{}'.format(folder, bimages, bucket_name=bucket))
    print('Successfully uploaded files')
    
    return fstats


for prefix in to_track[int(sys.argv[1]):int(sys.argv[2])]:
    fstats = BF_cell_features(prefix, remote_folder, bucket=bucket)
    print('Successfully output cell features for {}'.format(prefix))
    
