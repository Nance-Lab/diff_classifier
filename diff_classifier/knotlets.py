def download_and_split(filename):
    
    import diff_classifier.imagej as ij
    import diff_classifier.aws as aws
    import os.path as op
    
    local_name = op.split(filename)[1]
    DIR = op.split(filename)[0]
    try1 = filename.split('.')[0] + '_0_0.tif'
    try2 = filename.split('.')[0] + '_3_3.tif'
    
    s3 = boto3.client('s3')
    try:
        obj = s3.head_object(Bucket='ccurtis7.pup', Key=try1)
    except:
        try:
            obj = s3.head_object(Bucket='ccurtis7.pup', Key=try2)
        except:
            aws.download_s3(filename, local_name)
            names = ij.partition_im(local_name)
            for name in names:
                aws.upload_s3(name, op.join(op.split(filename)[0], name))
    print("Done with splitting.  Should output file of name {}".format(op.join(op.split(filename)[0], name)))


def download_and_track(filename):
    
    import diff_classifier.imagej as ij
    import diff_classifier.utils as ut
    import diff_classifier.aws as aws
    import os.path as op
    import pandas as pd
    
    aws.download_s3(filename, op.split(filename)[1])
    
    outfile = 'Traj_' + op.split(filename)[1].split('.')[0] + '.csv'
    local_im = op.join(os.getcwd(), op.split(filename)[1])
    if not op.isfile(outfile):
        ij.track(local_im, outfile, template=None, fiji_bin=None, radius=4.5, threshold=0., 
              do_median_filtering=True, quality=4.5, median_intensity=300.0, snr=0.0, 
              linking_max_distance=8.0, gap_closing_max_distance=10.0, max_frame_gap=2,
              track_displacement=10.0)

        aws.upload_s3(outfile, op.join(op.split(filename)[0], outfile))
    print("Done with tracking.  Should output file of name {}".format(op.join(op.split(filename)[0], outfile)))