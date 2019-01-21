import os
import os.path as op
import boto3
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import skimage.filters
from skimage.filters import try_all_threshold
from diff_classifier import aws
import matplotlib.image as mpimg
from scipy import ndimage as ndi
from skimage.transform import resize
import numpy.ma as ma
from skimage import morphology
from scipy.ndimage.morphology import distance_transform_edt as Euclidean

def PNN_binning(raw_img, input_df, num_bins=7, threshold='Otsu', min_obj_size=3000, bin_cuts=[0,3.57,7.14,14.28,142.8,714.3], \
                feat='Deff1', box_plot=True):
    
    
    # First, you have to scale the PNN confocal image so it aligns with the size of the MPT video
    # (usually, it must be scaled from a 512x512 image to a 2048x2048 image)
    img_scaled = resize(raw_img[0,:,:],(2048,2048))
    scaled_plot = plt.imshow(img_scaled, cmap='gray')
    plt.show()
    
    # this thresholds the scaled image using the user-input threshold type, converting the image to a binary
    
    if threshold == 'Otsu':
        thresh = skimage.filters.threshold_otsu(img_scaled)
    elif threshold == 'Mean':
        thresh = skimage.filters.threshold_mean(img_scaled)
    elif threshold == 'Triangle':
        thresh = skimage.filters.threshold_triangle(img_scaled)
    elif threshold == 'Yen':
        thresh = skimage.filters.threshold_yen(img_scaled)
    elif threshold == 'Li':
        thresh = skimage.filters.threshold_li(img_scaled)
    elif threshold == 'Isodata':
        thresh = skimage.filters.threshold_isodata(img_scaled)
    else:
        thresh = skimage.filters.threshold_minimum(img_scaled)
    
    pnnbinary = img_scaled > thresh
    
    # these next few lines of code fill any holes existing in the binary image and removes any 
    # objects smaller than the user-defined minimum object size
    
    pnnbinary_filled = ndi.binary_fill_holes(pnnbinary)
    binary_filledplot = plt.imshow(pnnbinary_filled, cmap='gray')
    plt.show()
    pnn_clean = morphology.remove_small_objects(pnnbinary_filled, min_size=min_obj_size)
    pnn_clean_plot = plt.imshow(pnn_clean, cmap='gray')
    plt.show()
    
    # Now, the Euclidean distance (to the nearest PNN) is caluclated at each pixel location in the 2048x2048 image
    
    euc_img = Euclidean(1-pnn_clean) #1- represents going outwards from the cells
    plt.imshow(euc_img)
    plt.show
    
    # With the Euclidean distance matrix generated, we now switch over to dataframe modification
    
    raw_df = input_df
    raw_df['Euc'] = np.nan
    raw_df['Bin'] = np.nan
    
    tot_traj = int(max(raw_df['Track_ID']))
    
    # The following for loop goes through all trajectories in the features dataframe, calculates the Euclidean 
    # distance from the centroid of the trajectory, and bins the trajectories based on those distances
    
    # don't want to deal with it now, but will have to adjust this section of code to be able to adjust the 
    # number of elif statements to match the number of bins chosen by the user
    
    counts = np.zeros(num_bins)

    for i in range(0,tot_traj+1):
        raw_df['Euc'][i] = euc_img[int(round(raw_df['X'][i])),int(round(raw_df['Y'][i]))]
    
        if raw_df['Euc'][i] == 0:
            raw_df['Bin'][i] = 1
            counts[0] = counts[0]+1
        elif raw_df['Euc'][i] <= bin_cuts[1]: #250 nm
            raw_df['Bin'][i] = 2
            counts[1] = counts[1]+1
        elif raw_df['Euc'][i] <= bin_cuts[2]: #500 nm
            raw_df['Bin'][i] = 3
            counts[2] = counts[2]+1
        elif raw_df['Euc'][i] <= bin_cuts[3]: # 1 um
            raw_df['Bin'][i] = 4
            counts[3] = counts[3]+1
        elif raw_df['Euc'][i] <= bin_cuts[4]: # 10 um
            raw_df['Bin'][i] = 5
            counts[4] = counts[4]+1
        elif raw_df['Euc'][i] <= bin_cuts[5]: # 50 um
            raw_df['Bin'][i] = 6
            counts[5] = counts[5]+1
        else:
            raw_df['Bin'][i] = 7
            counts[6] = counts[6]+1
    
    # This next for loop is used to create an array that summarizes the binning that's taken place. I had to 
    # do this to perform some statistics-based calculations and help with plot generation. It's final output 
    # is an array, where column # corresponds to bin #, and each row represents a distinct trajectory's 
    # value for the feature the user is interested in comparing
    
    summary_array = np.zeros((int(max(counts)), int(num_bins)))
    counts2 = np.zeros(num_bins)

    for i in range(0,tot_traj+1):
        if raw_df['Bin'][i] == 1:
            summary_array[int(counts2[0]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[0] = counts2[0] + 1
        elif raw_df['Bin'][i] == 2:
            summary_array[int(counts2[1]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[1] = counts2[1] + 1
        elif raw_df['Bin'][i] == 3:
            summary_array[int(counts2[2]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[2] = counts2[2] + 1
        elif raw_df['Bin'][i] == 4:
            summary_array[int(counts2[3]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[3] = counts2[3] + 1
        elif raw_df['Bin'][i] == 5:
            summary_array[int(counts2[4]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[4] = counts2[4] + 1
        elif raw_df['Bin'][i] == 6:
            summary_array[int(counts2[5]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[5] = counts2[5] + 1
        else:
            summary_array[int(counts2[6]), int(raw_df['Bin'][i]) - 1] = raw_df[feat][i]
            counts2[6] = counts2[6] + 1

    # This masks out any entries that are zero (blank), and prepares an array that allows for plotting
    
    masked_summary = ma.masked_where(summary_array == 0, summary_array)
    plot_array = [[y for y in row if y] for row in masked_summary.T]
    
    # This section simply allows you to overlay individual data points onto a boxplot of the data
    
    x = np.zeros(tot_traj+1)
    y = np.zeros(tot_traj+1)

    p = 0
    for i in range(np.size(plot_array)):
        for j in range(len(plot_array[i])):
            x[p] = i+1
            y[p] = plot_array[i][j]
            p = p + 1

    if box_plot==True:
        plt.subplot(211)
        plt.boxplot(plot_array, showfliers = False)
        plt.plot(x,y,'k.')
        plt.show()
    
    
    return(raw_df, plot_array, x, y)