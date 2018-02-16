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


def download_and_calc_MSDs(prefix):
    
    import diff_classifier.aws as aws
    import diff_classifier.utils as ut
    import diff_classifier.msd as msd
    import diff_classifier.features as ft
    import os
    import os.path as op
    import numpy as np
    import numpy.ma as ma
    import pandas as pd
    
    remote_folder = "01_18_Experiment/{}".format(prefix.split('_')[0])
    local_folder = os.getcwd()
    ires = 512

    for row in range(0, 4):
        for col in range(0, 4):
            filename = "Traj_{}_{}_{}.csv".format(prefix, row, col)
            to_download = op.join(remote_folder, filename)
            local_name = op.join(local_folder, filename)
            aws.download_s3(to_download, local_name)
            if row==0 and col==0:
                merged = msd.all_msds(ut.csv_to_pd(local_name))
            else:
                to_add = ut.csv_to_pd(local_name)
                to_add['X'] = to_add['X'] + ires*row
                to_add['Y'] = to_add['Y'] + ires*col
                to_add['Track_ID'] = to_add['Track_ID'] + max(merged['Track_ID'])
                merged.append(msd.all_msds(to_add))
            print('Successfully downloaded and calculated MSDs for {}_{}_{}'.format(prefix, row, col))
    
    merged.to_csv('MSD_{}.csv'.format(prefix))
    print('Saved MSDs as MSD_{}.csv'.format(prefix))
    merged_ft = ft.calculate_features(merged)
    merged_ft.to_csv('features_{}.csv'.format(prefix))
    print('Saved features as features_{}.csv'.format(prefix))


def download_split_track_msds(prefix):
    
    import diff_classifier.aws as aws
    import diff_classifier.utils as ut
    import diff_classifier.msd as msd
    import diff_classifier.features as ft
    import diff_classifier.imagej as ij
    import os
    import os.path as op
    import numpy as np
    import numpy.ma as ma
    import pandas as pd
    import boto3
    
    #Splitting section
    ###############################################################################################
    remote_folder = "01_18_Experiment/{}".format(prefix.split('_')[0])
    local_folder = os.getcwd()
    ires = 512
    frames = 651
    filename = '{}.tif'.format(prefix)
    remote_name = op.join(remote_folder, filename)
    local_name = op.join(local_folder, filename)
    
    #local_name = op.split(filename)[1]
    #DIR = op.split(filename)[0]
    try1 = prefix + '_0_0.tif'
    try2 = prefix + '_3_3.tif'
    
    s3 = boto3.client('s3')
    try:
        obj = s3.head_object(Bucket='ccurtis7.pup', Key=op.join(remote_folder, try1))
    except:
        try:
            obj = s3.head_object(Bucket='ccurtis7.pup', Key=op.join(remote_folder, try2))
        except:
            aws.download_s3(remote_name, local_name)
            names = ij.partition_im(local_name)
            for name in names:
                aws.upload_s3(name, op.join(remote_folder, name))
    print("Done with splitting.  Should output file of name {}".format(op.join(remote_folder, name)))
    
    
    #Tracking section
    ################################################################################################
    for name in names:
        outfile = 'Traj_' + name.split('.')[0] + '.csv'
        local_im = op.join(local_folder, name)
        if not op.isfile(outfile):
            ij.track(local_im, outfile, template=None, fiji_bin=None, radius=4.5, threshold=0., 
                  do_median_filtering=True, quality=4.5, median_intensity=300.0, snr=0.0, 
                  linking_max_distance=8.0, gap_closing_max_distance=10.0, max_frame_gap=2,
                  track_displacement=10.0)

            aws.upload_s3(outfile, op.join(remote_folder, outfile))
        print("Done with tracking.  Should output file of name {}".format(op.join(remote_folder, outfile)))
    
    
    #MSD and features section
    #################################################################################################
    counter = 0
    for name in names:
        row = int(name.split('.')[0].split('_')[4])
        col = int(name.split('.')[0].split('_')[5])
        
        filename = "Traj_{}_{}_{}.csv".format(prefix, row, col)
        local_name = op.join(local_folder, filename)

        if counter == 0:
            merged = msd.all_msds2(ut.csv_to_pd(local_name), frames=frames)
        else: 
            to_add = ut.csv_to_pd(local_name)
            to_add['X'] = to_add['X'] + ires*row
            to_add['Y'] = to_add['Y'] + ires*col
            to_add['Track_ID'] = to_add['Track_ID'] + max(merged['Track_ID'])
            merged = merged.append(msd.all_msds2(to_add, frames=frames))
        counter = counter + 1
        
        msd_file = 'msd_{}.csv'.format(prefix)
        merged.to_csv(msd_file)
        aws.upload_s3(msd_file, op.join(remote_folder, msd_file))
        merged_ft = ft.calculate_features(merged)
        ft_file = 'features_{}.csv'.format(prefix)
        merged_ft.to_csv(ft_file)
        aws.upload_s3(ft_file, op.join(remote_folder, ft_file))