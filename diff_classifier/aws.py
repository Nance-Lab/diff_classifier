import os
import diff_classifier.imagej as ij
import boto3
import os.path as op


def download_s3(remote_fname, local_fname, bucket_name="ccurtis7.pup"):
    """
    Download a file from S3 to our local file-system
    """
    if not os.path.exists(local_fname):
        s3 = boto3.resource('s3')
        b = s3.Bucket(bucket_name)
        b.download_file(remote_fname, local_fname)


def upload_s3(local_fname, remote_fname, bucket_name="ccurtis7.pup"):
    
    s3 = boto3.resource('s3')
    b = s3.Bucket(bucket_name)
    b.upload_file(local_fname, remote_fname) 
    
    
def partition_and_store(remote_fname, local_dir, bucket_name="ccurtis7.pup"):
    
    remote_dir, remote_file = op.split(remote_fname)
    download_s3(remote_fname, op.join(local_dir, remote_file))
    names = ij.partition_im(op.join(local_dir, remote_file))
    
    remote_names = []
    for file in names:
        local_file = op.split(file)[1]
        upload_s3(op.join(local_dir, local_file), op.join(remote_dir, local_file))
        remote_names.append(op.join(remote_dir, local_file))
    return remote_names