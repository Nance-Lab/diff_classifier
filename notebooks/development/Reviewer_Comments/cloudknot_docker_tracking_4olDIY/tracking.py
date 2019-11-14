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


def tracking(subprefix, remote_folder, bucket='nancelab.publicfiles',
             regress_f='regress.obj', rows=4, cols=4, ires=(512, 512),
             tparams={'radius': 3.0, 'threshold': 0.0,
                      'do_median_filtering': False, 'quality': 15.0,
                      'xdims': (0, 511), 'ydims': (1, 511),
                      'median_intensity': 300.0, 'snr': 0.0,
                      'linking_max_distance': 6.0,
                      'gap_closing_max_distance': 10.0, 'max_frame_gap': 3,
                      'track_duration': 20.0}):
    '''Tracks particles in input image using Trackmate.

    A function based on imagej.track that downloads the image from S3, tracks
    particles using Trackmate, and uploads the resulting trajectory file to S3.

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

    try:
        aws.download_s3(remote_folder+'/'+outfile, outfile, bucket_name=bucket)
    except:
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
        pickle_to_s3(args.sse, args.arrayjob)(tracking)(*input_)
    else:
        pickle_to_s3(args.sse, args.arrayjob)(tracking)(input_)
