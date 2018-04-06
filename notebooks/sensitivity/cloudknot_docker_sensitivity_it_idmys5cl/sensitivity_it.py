import boto3
import cloudpickle
import os
import pickle
from argparse import ArgumentParser
from functools import wraps


def pickle_to_s3(server_side_encryption=None, array_job=True):
    def real_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            s3 = boto3.client("s3")
            bucket = os.environ.get("CLOUDKNOT_JOBS_S3_BUCKET")

            if array_job:
                array_index = os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX")
            else:
                array_index = '0'

            jobid = os.environ.get("AWS_BATCH_JOB_ID")

            if array_job:
                jobid = jobid.split(':')[0]

            key = '/'.join([
                'cloudknot.jobs',
                os.environ.get("CLOUDKNOT_S3_JOBDEF_KEY"),
                jobid,
                array_index,
                '{0:03d}'.format(int(os.environ.get("AWS_BATCH_JOB_ATTEMPT"))),
                'output.pickle'
            ])

            result = f(*args, **kwargs)

            # Only pickle output and write to S3 if it is not None
            if result is not None:
                pickled_result = cloudpickle.dumps(result)
                if server_side_encryption is None:
                    s3.put_object(Bucket=bucket, Body=pickled_result, Key=key)
                else:
                    s3.put_object(Bucket=bucket, Body=pickled_result, Key=key,
                                  ServerSideEncryption=server_side_encryption)

        return wrapper
    return real_decorator


def sensitivity_it(counter):
    
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import diff_classifier.aws as aws
    import diff_classifier.utils as ut
    import diff_classifier.msd as msd
    import diff_classifier.features as ft
    import diff_classifier.imagej as ij
    import diff_classifier.heatmaps as hm

    from scipy.spatial import Voronoi
    import scipy.stats as stats
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import matplotlib.cm as cm
    import os
    import os.path as op
    import numpy as np
    import numpy.ma as ma
    import pandas as pd
    import boto3
    import itertools
    
    #Sweep parameters
    #----------------------------------
    radius = [4.5, 6.0, 7.0]
    do_median_filtering = [True, False]
    quality = [1.5, 4.5, 8.5]
    linking_max_distance = [6.0, 10.0, 15.0]
    gap_closing_max_distance = [6.0, 10.0, 15.0]
    max_frame_gap = [1, 2, 5]
    track_displacement = [0.0, 10.0, 20.0]

    sweep = [radius, do_median_filtering, quality, linking_max_distance, gap_closing_max_distance, max_frame_gap,
             track_displacement]
    all_params = list(itertools.product(*sweep))

    #Variable prep
    #----------------------------------
    s3 = boto3.client('s3')

    folder = '01_18_Experiment'
    s_folder = '{}/sensitivity'.format(folder)
    local_folder = '.'
    prefix = "P1_S1_R_0001_2_2"
    name = "{}.tif".format(prefix)
    local_im = op.join(local_folder, name)
    aws.download_s3('{}/{}/{}.tif'.format(folder, prefix.split('_')[0], prefix), '{}.tif'.format(prefix))

    outputs = np.zeros((len(all_params), len(all_params[0])+2))

    #Tracking and calculations
    #------------------------------------
    params = all_params[counter]
    outfile = 'Traj_{}_{}.csv'.format(name.split('.')[0], counter)
    msd_file = 'msd_{}_{}.csv'.format(name.split('.')[0], counter)
    geo_file = 'geomean_{}_{}.csv'.format(name.split('.')[0], counter)
    geoS_file = 'geoSEM_{}_{}.csv'.format(name.split('.')[0], counter)
    msd_image = 'msds_{}_{}.png'.format(name.split('.')[0], counter)
    iter_name = "{}_{}".format(prefix, counter)

    ij.track(local_im, outfile, template=None, fiji_bin=None, radius=params[0], threshold=0., 
             do_median_filtering=params[1], quality=params[2], x=511, y=511, ylo=1, median_intensity=300.0, snr=0.0, 
             linking_max_distance=params[3], gap_closing_max_distance=params[4], max_frame_gap=params[5],
             track_displacement=params[6])

    traj = ut.csv_to_pd(outfile)
    msds = msd.all_msds2(traj, frames=651)
    msds.to_csv(msd_file)
    gmean1, gSEM1 = hm.plot_individual_msds(iter_name, alpha=0.05)
    np.savetxt(geo_file, gmean1, delimiter=",")
    np.savetxt(geoS_file, gSEM1, delimiter=",")

    aws.upload_s3(outfile, '{}/{}'.format(s_folder, outfile))
    aws.upload_s3(msd_file, '{}/{}'.format(s_folder, msd_file))
    aws.upload_s3(geo_file, '{}/{}'.format(s_folder, geo_file))
    aws.upload_s3(geoS_file, '{}/{}'.format(s_folder, geoS_file))
    aws.upload_s3(msd_image, '{}/{}'.format(s_folder, msd_image))

    print('Successful parameter calculations for {}'.format(iter_name))


if __name__ == "__main__":
    description = ('Download input from an S3 bucket and provide that input '
                   'to our function. On return put output in an S3 bucket.')

    parser = ArgumentParser(description=description)

    parser.add_argument(
        'bucket', metavar='bucket', type=str,
        help='The S3 bucket for pulling input and pushing output.'
    )

    parser.add_argument(
        '--starmap', action='store_true',
        help='Assume input has already been grouped into a single tuple.'
    )

    parser.add_argument(
        '--arrayjob', action='store_true',
        help='If True, this is an array job and it should reference the '
             'AWS_BATCH_JOB_ARRAY_INDEX environment variable.'
    )

    parser.add_argument(
        '--sse', dest='sse', action='store',
        choices=['AES256', 'aws:kms'], default=None,
        help='Server side encryption algorithm used when storing objects '
             'in S3.'
    )

    args = parser.parse_args()

    s3 = boto3.client('s3')
    bucket = args.bucket

    jobid = os.environ.get("AWS_BATCH_JOB_ID")

    if args.arrayjob:
        jobid = jobid.split(':')[0]

    key = '/'.join([
        'cloudknot.jobs',
        os.environ.get("CLOUDKNOT_S3_JOBDEF_KEY"),
        jobid,
        'input.pickle'
    ])

    response = s3.get_object(Bucket=bucket, Key=key)
    input_ = pickle.loads(response.get('Body').read())

    if args.arrayjob:
        array_index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
        input_ = input_[array_index]

    if args.starmap:
        pickle_to_s3(args.sse, args.arrayjob)(sensitivity_it)(*input_)
    else:
        pickle_to_s3(args.sse, args.arrayjob)(sensitivity_it)(input_)
