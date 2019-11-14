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


def assemble_msds(prefix, remote_folder, bucket='nancelab.publicfiles',
                  ires=(512, 512), frames=651, rows=4, cols=4):
    '''Calculates MSDs and features from input trajectory files

    A function based on msd.all_msds2 and features.calculate_features, creates
    msd and feature csv files from input trajectory files and uploads to S3.

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
        Number of rows to split image into.
    cols : int
        Number of columns to split image into.

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

    names = []
    for i in range(0, 4):
        for j in range(0, 4):
            names.append('{}_{}_{}.tif'.format(prefix, i, j))

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
        pickle_to_s3(args.sse, args.arrayjob)(assemble_msds)(*input_)
    else:
        pickle_to_s3(args.sse, args.arrayjob)(assemble_msds)(input_)
