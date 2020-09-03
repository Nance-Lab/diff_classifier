"""
IO functions for downloading and uploading files from AWS S3 buckets.

The diff_classifier module was built to be used with conjunction with AWS
services. With the exception of Cloudknot parallelization capabilities, most
functions can be used separate from AWS. These functions faciliate interaction
with files stores in S3 buckets. Users must have appropriate credentials to
access desired S3 buckets.

"""
import os
import os.path as op
import boto3


def get_s3_keys(bucket_name, remote_folder, keywords=''):
    """
    Gets all s3 keys for bucket_name/remote_folder. Uses a list 
    convention to go through keywords (i.e): ['a', 'b', 'c OR d
    OR e'] will find all files containing 'a' and 'b' and either
    'c', 'd', or 'e'. Using '' will return every file key in
    
    folder.
    
    Parameters
    ----------
    remote_folder: string
        Desired name to be stored in S3 bucket.
    bucket_name: string
        Bucket name on S3.
    keywords: string or [strings]
        Keyword or list of keywords to search for. ex:
            ['a', 'b', 'c OR d OR e'] will 
            find all files containing 'a' and 'b' and either 
            'c', 'd', or 'e'. Using '' will return every 
            file key in folder.
    Returns
    -------
    obj_list : [s3.Object()]
        list containing the desitred s3 objects
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    obj_list = []
    keywords = [i.split('OR') for i in list(keywords)]
    keywords = [list(map(lambda x:x.strip(), i)) for i in keywords]
    for object in bucket.objects.all():
        filename = object.key.split("/")[-1]
        kwds_in = all(any(k in filename for k in ([keyword]*isinstance(keyword, str) or keyword)) for keyword in keywords)
        if remote_folder in object.key and kwds_in:
            obj_list.append(s3.Object(object.bucket_name, object.key))
    return obj_list


def glacier_restore(bucket_name, remote_folder, keywords='', days=5, tier='Standard'):
    """
    Get s3 keys in s3 bucket based on keywords. Checks for files
    under keyword if they are stored in S3 glacier. Restores for
    days days using tier restore tier. Uses a list convention to
    go through keywords (i.e): ['a', 'b', 'c OR d OR e'] will
    find all files containing 'a' and 'b' and either 'c', 'd',
    or 'e'. Using '' will return every file key in folder. Tier
    options are Expedited, Standard, and Bulk.
    
    Parameters
    ----------
    remote_folder : string
        Desired name to be stored in S3 bucket.
    bucket_name : string
        Bucket name on S3.
    keywords : string or [strings]
        Keyword or list of keywords to search for. ex:
            ['a', 'b', 'c OR d OR e'] will 
            find all files containing 'a' and 'b' and either 
            'c', 'd', or 'e'. Using '' will return every 
            file key in folder.
    days : int
        Number of days to restore
    tier : string
        Tier to restoer with. Acceptable tiers are 'Expedited', 'Standard', 'Bulk'
    Returns
    -------
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    restore_requestDict = {'Days': days,
                           'GlacierJobParameters': {
                               'Tier': tier},}
    obj_list = get_s3_keys(bucket_name, remote_folder, keywords)
    if len(obj_list) > 10:
        press = input(f'Large amount of items. Ok to restore {len(obj_list)} items? (Y/N): ')
        if press != 'Y':
            print('Canceling restore')
            return
    for object in obj_list:
        if object.storage_class == "GLACIER":
            if object.restore is None:
                print(f"Requesting restore for --> {object.key}")
                object.restore_object(Bucket=object.bucket_name,
                                      Key=object.key,
                                      RestoreRequest=restore_requestDict)
            elif 'ongoing-request="true"' in object.restore:
                print(f"Already in restore queue --> {object.key}")
            elif 'ongoing-request="false"' in object.restore:
                print(f"Already restored --> {object.key}")
        else:
            print(f"File doesn't need restore --> {object.key}")
    print('---REQUEST DONE---')
    
# Takes in a String, "bucket1", a string, "folder1", a string,
# "bucket2", a string "folder2", and a list of strings or a 
# single string, "keywords". Will check for similar files 
# between bucket1/folder1 and bucket2/folder2 and output 
# them as a set of tuples containing the location of these
# files. if bucket2 or folder2 are not inputed, will check
# the same bucket1/folder1 for duplicated files. Uses a list convention
# to go through keywords (i.e): ['a', 'b', 'c OR d OR e'] will 
# find all files containing 'a' and 'b' and either 'c', 'd', or 'e'.
# Using '' will return every file key in folder.
def check_duplicated_files(bucket1, folder1, bucket2=None, folder2=None, keywords=''):
    """
    Checks for duplicated files within s3. Will check for similar files
    between bucket1/folder1 and bucket2/folder2 and output them as a set
    of tuples containing the location of these files. if bucket2 or
    folder2 are not inputed, will check the same bucket1/folder1 for
    duplicated files. Uses a list convention to go through keywords
    (i.e): ['a', 'b', 'c OR d OR e'] will find all files containing
    'a' and 'b' and either 'c', 'd', or 'e'. Using '' will return every
    file key in folder.
    
    Parameters
    ----------
    bucket1 : string
        Bucket name on S3.
    folder1 : string
        Bucket name on S3.
    bucket2 : string
        Bucket name on S3.
    folder2 : string
        Bucket name on S3.
    keywords : string or [strings]
        Keyword or list of keywords to search for. ex:
            ['a', 'b', 'c OR d OR e'] will 
            find all files containing 'a' and 'b' and either 
            'c', 'd', or 'e'. Using '' will return every 
            file key in folder.
    Returns
    -------
    duplicate_list : [string]
        List of s3 objects in two buckets that are duplicate
    """
    if (bucket2 == None):
        bucket2 = bucket1
    if (folder2 == None):
        folder2 = folder1
    s3 = boto3.client('s3')
    obj_list1 = get_s3_keys(bucket1, folder1, keywords)
    obj_list2 = get_s3_keys(bucket2, folder2, keywords)
    duplicate_list = set()
    for object1 in obj_list1:
        duplicated_objects = [f"{object1.bucket_name}/{object1.key}"]
        etag1 = s3.head_object(Bucket=object1.bucket_name,Key=object1.key)['ResponseMetadata']['HTTPHeaders']['etag']
        for object2 in obj_list2:
            etag2 = s3.head_object(Bucket=object2.bucket_name,Key=object2.key)['ResponseMetadata']['HTTPHeaders']['etag']
            if (etag1 == etag2):
                if (f"{object2.bucket_name}/{object2.key}" not in duplicated_objects):
                    duplicated_objects.append(f"{object2.bucket_name}/{object2.key}")
        if len(duplicated_objects) > 1:
            duplicate_list.add(tuple(sorted(duplicated_objects, key=lambda sent: len(sent))))
    return duplicate_list


def download_s3(bucket_name, remote_folder, local_folder, keywords=''):
    """
    Download a file from s3 to local file system. Downloads all files
    in bucket_name/remote_folder from s3 that match keywords into
    remote_folder. into a local folder Uses a list convention to go
    through keywords (i.e): ['a', 'b', 'c OR d OR e'] will find all
    files containing 'a' and 'b' and either 'c', 'd', or 'e'. Using
    '' will return every file key in folder.
    
    Parameters
    ----------
    local_folder : string
        Name of local file stored on computer.
    remote_folder : string
        Desired name to be stored in S3 bucket.
    bucket_name : string
        Bucket name on S3.
    keywords : string or [strings]
        Keyword or list of keywords to search for. ex:
            ['a', 'b', 'c OR d OR e'] will 
            find all files containing 'a' and 'b' and either 
            'c', 'd', or 'e'. Using '' will return every 
            file key in folder.
    Returns
    -------
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for object in get_s3_keys(bucket_name, remote_folder, keywords):
        local_path = '/'.join([local_folder, object.key.split('/')[-1]])
        if not os.path.exists(local_path):
            bucket.download_file(object.key, local_path)


def upload_s3(local_fname, remote_fname, bucket_name="ccurtis.data"):
    """
    Upload a file from local file-system to S3.

    Parameters
    ----------
    local_fname : string
        Name of local file stored on computer.
    remote_fname : string
        Desired name to be stored in S3 bucket.
    bucket_name : string
        Bucket name on S3.
    Returns
    -------
    """

    sthree = boto3.resource('s3')
    buckt = sthree.Bucket(bucket_name)
    buckt.upload_file(local_fname, remote_fname)


# def partition_and_store(remote_fname, local_dir, bucket_name="ccurtis7.pup"):
#     """
#     Download image from S3, partition, and upload partitions to S3.

#     Parameters
#     ----------
#     remote_fname: string
#         Target filename in S3 bucket.
#     local_dir: string
#         Local directory to store downloaded file.
#     bucket_name: string
#         Bucket name on S3.

#     Returns
#     -------
#     remote_names: list of strings.
#         Names of partitioned images in S3.
#     """
#     remote_dir, remote_file = op.split(remote_fname)
#     download_s3(remote_fname, op.join(local_dir, remote_file))
#     names = ij.partition_im(op.join(local_dir, remote_file))

#     remote_names = []
#     for file in names:
#         local_file = op.split(file)[1]
#         upload_s3(op.join(local_dir, local_file), op.join(remote_dir, local_file))
#         remote_names.append(op.join(remote_dir, local_file))
#     return remote_names
