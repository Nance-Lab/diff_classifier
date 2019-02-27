'''Functions to submit tracking jobs to AWS Batch with Cloudknot

This is a set of custom functions for use with Cloutknot for parallelized
multi-particle tracking workflows. These can also be used as template if
users want to build their own parallelized workflows. See Cloudknot
documentation at https://richford.github.io/cloudknot/documentation.html
for more information.

The base set of functions is split, tracking, and assemble_msds. The split
function splits large images into smaller images that are manageable for a
single EC2 instance. The tracking function tracks nanoparticle trajectories in
a single sub-image from the split function. The assemble_msds function operates
on all sub-image trajectory csv files from the tracking function, calculates
MSDs and features and assembles them into a single msd csv file and a single
features csv file. The workflow looks something like this:

                  |-track---|
                  |-track---|
(image) -split----|         |--assemble_msds-----> (msd/feature files)
                  |-track---|
                  |-track---|

'''


def split(prefix, remote_folder, bucket,
          rows=4, cols=4, ores=(2048, 2048), ires=(512, 512)):
    '''Splits input image file into smaller images.

    A function based on imagej.partition_im that download images from an S3
    bucket, splits it into smaller images, and uploads these to S3. Designed to
    work with Cloudknot for parallelizable workflows. Typically, this function
    is used in conjunction with kn.tracking and kn.assemble_msds for a complete
    analysis.

    Parameters
    ----------
    prefix : string
        Prefix (everything except file extension and folder name) of image file
        to be tracked. Must be available on S3.
    remote_folder : string
        Folder name where file is contained on S3 in the bucket specified by
        'bucket'.
    bucket : string
        S3 bucket where file is contained.
    rows : int
        Number of rows to split image into.
    cols : int
        Number of columns to split image into.
    ores : tuple of int
        Original resolution of input image.
    ires : tuple of int
        Resolution of split images. Really just a sanity check to make sure you
        correctly splitting.

    '''

    import os
    import boto3
    import diff_classifier.aws as aws
    import diff_classifier.imagej as ij

    local_folder = os.getcwd()
    filename = '{}.tif'.format(prefix)
    remote_name = remote_folder+'/'+filename
    local_name = local_folder+'/'+filename
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)
    aws.download_s3(remote_name, local_name, bucket_name=bucket)

    s3 = boto3.client('s3')

    # Splitting section
    names = ij.partition_im(local_name, irows=rows, icols=cols,
                            ores=ores, ires=ires)

    # Names of subfiles
    # names = []
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         names.append('{}_{}_{}.tif'.format(prefix, i, j))

    for name in names:
        aws.upload_s3(name, remote_folder+'/'+name, bucket_name=bucket)
        os.remove(name)
        print("Done with splitting. Should output file of name {}".format(
              remote_folder+'/'+name))

    os.remove(filename)


def tracking(subprefix, remote_folder, bucket, tparams,
             regress_f='regress.obj', rows=4, cols=4, ires=(512, 512)):
    '''Tracks particles in input image using Trackmate.

    A function based on imagej.track that downloads the image from S3, tracks
    particles using Trackmate, and uploads the resulting trajectory file to S3.
    Designed to work with Cloudknot for parallelizable workflows. Typically,
    this function is used in conjunction with kn.split and kn.assemble_msds for
    a complete analysis.

    Parameters
    ----------
    subprefix : string
        Prefix (everything except file extension and folder name) of image file
        to be tracked. Must be available on S3.
    remote_folder : string
        Folder name where file is contained on S3 in the bucket specified by
        'bucket'.
    bucket : string
        S3 bucket where file is contained.
    regress_f : string
        Name of regress object used to predict quality parameter.
    rows : int
        Number of rows to split image into.
    cols : int
        Number of columns to split image into.
    ires : tuple of int
        Resolution of split images. Really just a sanity check to make sure you
        correctly splitting.
    tparams : dict
        Dictionary containing tracking parameters to Trackmate analysis.

    '''

    import os
    import os.path as op
    import boto3
    from sklearn.externals import joblib
    import diff_classifier.aws as aws
    import diff_classifier.utils as ut
    import diff_classifier.msd as msd
    import diff_classifier.features as ft
    import diff_classifier.imagej as ij

    local_folder = os.getcwd()
    filename = '{}.tif'.format(subprefix)
    remote_name = remote_folder+'/'+filename
    local_name = local_folder+'/'+filename
    outfile = 'Traj_' + subprefix + '.csv'
    local_im = op.join(local_folder, '{}.tif'.format(subprefix))
    row = int(subprefix.split('_')[-2])
    col = int(subprefix.split('_')[-1])

    aws.download_s3(remote_folder+'/'+regress_f, regress_f, bucket_name=bucket)
    with open(regress_f, 'rb') as fp:
        regress = joblib.load(fp)

    s3 = boto3.client('s3')

    aws.download_s3('{}/{}'.format(remote_folder,
                    '{}.tif'.format(subprefix)),
                    local_im, bucket_name=bucket)
    tparams['quality'] = ij.regress_tracking_params(regress, subprefix,
                                                    regmethod='PassiveAggressiveRegressor')

    if row == rows-1:
        tparams['ydims'] = (tparams['ydims'][0], ires[1] - 27)

    ij.track(local_im, outfile, template=None, fiji_bin=None,
             tparams=tparams)
    aws.upload_s3(outfile, remote_folder+'/'+outfile, bucket_name=bucket)
    print("Done with tracking.  Should output file of name {}".format(
          remote_folder+'/'+outfile))


def assemble_msds(prefix, remote_folder, bucket,
                  ires=(512, 512), frames=651):
    '''Calculates MSDs and features from input trajectory files

    A function based on msd.all_msds2 and features.calculate_features, creates
    msd and feature csv files from input trajectory files and uploads to S3.
    Designed to work with Cloudknot for parallelizable workflows. Typically,
    this function is used in conjunction with kn.split and kn.tracking for an
    entire workflow.

    prefix : string
        Prefix (everything except file extension and folder name) of image file
        to be tracked. Must be available on S3.
    remote_folder : string
        Folder name where file is contained on S3 in the bucket specified by
        'bucket'.
    bucket : string
        S3 bucket where file is contained.
    ires : tuple of int
        Resolution of split images. Really just a sanity check to make sure you
        correctly splitting.
    frames : int
        Number of frames in input videos.
    rows : int
        Number of rows of split images (from kn.split).
    cols : int
        Number of columns of split images (from kn.split).

    '''

    import os
    import boto3
    import diff_classifier.aws as aws
    import diff_classifier.msd as msd
    import diff_classifier.features as ft
    import diff_classifier.utils as ut

    filename = '{}.tif'.format(prefix)
    remote_name = remote_folder+'/'+filename
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)

    s3 = boto3.client('s3')

    # names = []
    # for i in range(0, 4):
    #     for j in range(0, 4):
    #         names.append('{}_{}_{}.tif'.format(prefix, i, j))
    all_objects = s3.list_objects(Bucket=bucket,
                                  Prefix='{}/{}_'.format(remote_folder,
                                                         prefix))
    names = []
    for entry in all_objects['Contents']:
        name = entry['Key'].split('/')[1]
        names.append(name)
        row = int(name.split(prefix)[1].split('.')[0].split('_')[1])
        col = int(name.split(prefix)[1].split('.')[0].split('_')[2])
        if row > rows:
            rows = row
        if col > cols:
            cols = col
    rows = rows + 1
    cols = cols + 1

    counter = 0
    for name in names:
        row = int(name.split(prefix)[1].split('.')[0].split('_')[1])
        col = int(name.split(prefix)[1].split('.')[0].split('_')[2])

        filename = "Traj_{}_{}_{}.csv".format(prefix, row, col)
        aws.download_s3(remote_folder+'/'+filename, filename,
                        bucket_name=bucket)
        local_name = filename

        if counter == 0:
            to_add = ut.csv_to_pd(local_name)
            to_add['X'] = to_add['X'] + ires[0]*col
            to_add['Y'] = ires[1] - to_add['Y'] + ires[1]*(rows-1-row)
            merged = msd.all_msds2(to_add, frames=frames)
        else:

            if merged.shape[0] > 0:
                to_add = ut.csv_to_pd(local_name)
                to_add['X'] = to_add['X'] + ires[0]*col
                to_add['Y'] = ires[1] - to_add['Y'] + ires[1]*(rows-1-row)
                to_add['Track_ID'] = to_add['Track_ID'
                                            ] + max(merged['Track_ID']) + 1
            else:
                to_add = ut.csv_to_pd(local_name)
                to_add['X'] = to_add['X'] + ires[0]*col
                to_add['Y'] = ires[1] - to_add['Y'] + ires[1]*(rows-1-row)
                to_add['Track_ID'] = to_add['Track_ID']

            merged = merged.append(msd.all_msds2(to_add, frames=frames))
            print('Done calculating MSDs for row {} and col {}'.format(row,
                                                                       col))
        counter = counter + 1

    merged.to_csv(msd_file)
    aws.upload_s3(msd_file, remote_folder+'/'+msd_file, bucket_name=bucket)
    merged_ft = ft.calculate_features(merged)
    merged_ft.to_csv(ft_file)
    aws.upload_s3(ft_file, remote_folder+'/'+ft_file, bucket_name=bucket)

    os.remove(ft_file)
    os.remove(msd_file)
    for name in names:
        outfile = 'Traj_' + name.split('.')[0] + '.csv'
        os.remove(outfile)


def split_track_msds(prefix, remote_folder, bucket, tparams,
                     rows=4, cols=4, ores=(2048, 2048), ires=(512, 512),
                     to_split=False, regress_f='regress.obj', frames=651):
    '''Splits images, track particles, and calculates MSDs

    A composite function designed to work with Cloudknot to split images,
    track particles, and calculate MSDs.

    Parameters
    ----------
    prefix : string
        Prefix (everything except file extension and folder name) of image file
        to be tracked. Must be available on S3.
    remote_folder : string
        Folder name where file is contained on S3 in the bucket specified by
        'bucket'.
    bucket : string
        S3 bucket where file is contained.
    rows : int
        Number of rows to split image into.
    cols : int
        Number of columns to split image into.
    ores : tuple of int
        Original resolution of input image.
    ires : tuple of int
        Resolution of split images. Really just a sanity check to make sure you
        correctly splitting.
    to_split : bool
        If True, will perform image splitting.
    regress_f : string
        Name of regress object used to predict quality parameter.
    frames : int
        Number of frames in input videos.
    tparams : dict
        Dictionary containing tracking parameters to Trackmate analysis.

    '''

    if to_split:
        split(prefix=prefix, remote_folder=remote_folder, bucket=bucket,
              rows=rows, cols=cols, ores=ores, ires=ires)

    pref = []
    for row in range(0, rows):
        for col in range(0, cols):
            pref.append("{}_{}_{}".format(prefix, row, col))

    for subprefix in pref:
        tracking(subprefix=subprefix, remote_folder=remote_folder, bucket=bucket,
                 regress_f=regress_f, rows=rows, cols=cols, ires=ires,
                 tparams=tparams)

    assemble_msds(prefix=prefix, remote_folder=remote_folder, bucket=bucket,
                  ires=ires, frames=frames)


# def sensitivity_it(counter):
#     '''Performs sensitivity analysis on single input image
#
#     An example function (not designed for re-use) of a sensitivity analysis that
#     demonstrates the impact of input tracking parameters on output MSDs and
#     features.
#
#     '''
#
#     import matplotlib as mpl
#     mpl.use('Agg')
#     import matplotlib.pyplot as plt
#     import diff_classifier.aws as aws
#     import diff_classifier.utils as ut
#     import diff_classifier.msd as msd
#     import diff_classifier.features as ft
#     import diff_classifier.imagej as ij
#     import diff_classifier.heatmaps as hm
#
#     from scipy.spatial import Voronoi
#     import scipy.stats as stats
#     from shapely.geometry import Point
#     from shapely.geometry.polygon import Polygon
#     import matplotlib.cm as cm
#     import os
#     import os.path as op
#     import numpy as np
#     import numpy.ma as ma
#     import pandas as pd
#     import boto3
#     import itertools
#
#     # Sweep parameters
#     # ----------------------------------
#     radius = [4.5, 6.0, 7.0]
#     do_median_filtering = [True, False]
#     quality = [1.5, 4.5, 8.5]
#     linking_max_distance = [6.0, 10.0, 15.0]
#     gap_closing_max_distance = [6.0, 10.0, 15.0]
#     max_frame_gap = [1, 2, 5]
#     track_displacement = [0.0, 10.0, 20.0]
#
#     sweep = [radius, do_median_filtering, quality, linking_max_distance,
#              gap_closing_max_distance, max_frame_gap, track_displacement]
#     all_params = list(itertools.product(*sweep))
#
#     # Variable prep
#     # ----------------------------------
#     s3 = boto3.client('s3')
#
#     folder = '01_18_Experiment'
#     s_folder = '{}/sensitivity'.format(folder)
#     local_folder = '.'
#     prefix = "P1_S1_R_0001_2_2"
#     name = "{}.tif".format(prefix)
#     local_im = op.join(local_folder, name)
#     aws.download_s3('{}/{}/{}.tif'.format(folder, prefix.split('_')[0], prefix),
#                     '{}.tif'.format(prefix))
#
#     outputs = np.zeros((len(all_params), len(all_params[0])+2))
#
#     # Tracking and calculations
#     # ------------------------------------
#     params = all_params[counter]
#     outfile = 'Traj_{}_{}.csv'.format(name.split('.')[0], counter)
#     msd_file = 'msd_{}_{}.csv'.format(name.split('.')[0], counter)
#     geo_file = 'geomean_{}_{}.csv'.format(name.split('.')[0], counter)
#     geoS_file = 'geoSEM_{}_{}.csv'.format(name.split('.')[0], counter)
#     msd_image = 'msds_{}_{}.png'.format(name.split('.')[0], counter)
#     iter_name = "{}_{}".format(prefix, counter)
#
#     ij.track(local_im, outfile, template=None, fiji_bin=None, radius=params[0], threshold=0.,
#              do_median_filtering=params[1], quality=params[2], x=511, y=511, ylo=1, median_intensity=300.0, snr=0.0,
#              linking_max_distance=params[3], gap_closing_max_distance=params[4], max_frame_gap=params[5],
#              track_displacement=params[6])
#
#     traj = ut.csv_to_pd(outfile)
#     msds = msd.all_msds2(traj, frames=651)
#     msds.to_csv(msd_file)
#     gmean1, gSEM1 = hm.plot_individual_msds(iter_name, alpha=0.05)
#     np.savetxt(geo_file, gmean1, delimiter=",")
#     np.savetxt(geoS_file, gSEM1, delimiter=",")
#
#     aws.upload_s3(outfile, '{}/{}'.format(s_folder, outfile))
#     aws.upload_s3(msd_file, '{}/{}'.format(s_folder, msd_file))
#     aws.upload_s3(geo_file, '{}/{}'.format(s_folder, geo_file))
#     aws.upload_s3(geoS_file, '{}/{}'.format(s_folder, geoS_file))
#     aws.upload_s3(msd_image, '{}/{}'.format(s_folder, msd_image))
#
#     print('Successful parameter calculations for {}'.format(iter_name))
