.. _interacting-with-s3-label:

Uploading and download with s3
==============================

The two basic functions found in the aws package for interacting with s3 are
download_s3 and upload_s3.  Users must select a bucket by changing the variable
bucket_name.  Users must also have the correct permissions on their computer
to download from or upload to the bucket of interest.

.. code-block:: python
  download_s3(remote_fname, local_fname, bucket_name="ccurtis7.pup")