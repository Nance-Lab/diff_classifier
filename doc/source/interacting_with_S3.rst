.. _interacting-with-s3-label:

Uploading and download with s3
==============================

The two basic functions found in the aws package for interacting with s3 are
download_s3 and upload_s3.  Users must select a bucket by changing the variable
bucket_name.  Users must also have an AWS account and the correct permissions on
their computer to download from or upload to the bucket of interest.

.. code-block:: python

  download_s3(remote_fname, local_fname, bucket_name="nancelab.publicfiles")

Example tif images for tracking analysis are available in a publicly
accessible bucket nancelab.publicfiles. Despite the public access, users must
have an AWS account in order to access the files.

For information on using the AWS Command Line Interface, check out the
documentation
`here <https://docs.aws.amazon.com/cli/latest/userguide/installing.html>`_.
