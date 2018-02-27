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
                aws.upload_s3(name, op.split(filename)[0]+'/'+name)
    print("Done with splitting.  Should output file of name {}".format(op.split(filename)[0]+'/'+name))


def download_and_track(filename):
    
    import diff_classifier.imagej as ij
    import diff_classifier.utils as ut
    import diff_classifier.aws as aws
    import os.path as op
    import pandas as pd
    
    aws.download_s3(filename, op.split(filename)[1])
    
    outfile = 'Traj_' + op.split(filename)[1].split('.')[0] + '.csv'
    local_im = op.split(filename)[1]
    if not op.isfile(outfile):
        ij.track(local_im, outfile, template=None, fiji_bin=None, radius=4.5, threshold=0., 
              do_median_filtering=True, quality=4.5, median_intensity=300.0, snr=0.0, 
              linking_max_distance=8.0, gap_closing_max_distance=10.0, max_frame_gap=2,
              track_displacement=10.0)

        aws.upload_s3(outfile, op.split(filename)[0]+'/'+outfile)
    print("Done with tracking.  Should output file of name {}".format(op.split(filename)[0]+'/'+outfile))


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
            to_download = remote_folder+'/'+filename
            local_name = local_folder+'/'+filename
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
    """
    1. Checks to see if features file exists.
    2. If not, checks to see if image partitioning has occured.
    3. If yes, checks to see if tracking has occured.
    4. Regardless, tracks, calculates MSDs and features. 
    """
    
    import diff_classifier.aws as aws
    import diff_classifier.utils as ut
    import diff_classifier.msd as msd
    import diff_classifier.features as ft
    import diff_classifier.imagej as ij
    import diff_classifier.heatmaps as hm
    import matplotlib.pyplot as plt
    import scipy.stats as stats
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
    remote_name = remote_folder+'/'+filename
    local_name = local_folder+'/'+filename
    
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)
    
    s3 = boto3.client('s3')
      
    names = []
    for i in range(0, 4):
        for j in range(0, 4):
            names.append('{}_{}_{}.tif'.format(prefix, i, j))
    
    try:
        obj = s3.head_object(Bucket='ccurtis7.pup', Key=remote_name+'/'+ft_file)
    except:

        try:
            for name in names:
                aws.download_s3(remote_folder+'/'+name, name)
        except:
            aws.download_s3(remote_name, local_name)
            names = ij.partition_im(local_name)
            for name in names:
                aws.upload_s3(name, remote_folder+'/'+name)
                print("Done with splitting.  Should output file of name {}".format(remote_folder+'/'+name))

        #Tracking section
        ################################################################################################
        for name in names:
            outfile = 'Traj_' + name.split('.')[0] + '.csv'
            local_im = name

            try:
                aws.download_s3(remote_folder+'/'+outfile, outfile)
            except:
                test_intensity = ij.mean_intensity(local_im)
                if test_intensity > 500:
                    quality = 245
                else:
                    quality = 4.5
                
                if row==3:
                    y = 500
                else:
                    y = 511

                ij.track(local_im, outfile, template=None, fiji_bin=None, radius=4.5, threshold=0., 
                         do_median_filtering=True, quality=quality, x=511, y=y, ylo=1, median_intensity=300.0, snr=0.0, 
                         linking_max_distance=8.0, gap_closing_max_distance=10.0, max_frame_gap=2,
                         track_displacement=10.0) 

                aws.upload_s3(outfile, remote_folder+'/'+outfile)
            print("Done with tracking.  Should output file of name {}".format(remote_folder+'/'+outfile))


        #MSD and features section
        #################################################################################################
        files_to_big = False
        size_limit = 10
        
        for name in names:
            outfile = 'Traj_' + name.split('.')[0] + '.csv'
            local_im = name
            file_size_MB = op.getsize(local_im)/1000000
            if file_size_MB > size_limit:
                file_to_big = True

        if files_to_big:
            print('One or more of the {} trajectory files exceeds {}MB in size.  Will not continue with MSD calculations.'.format(
                  prefix, size_limit))
        else:
            counter = 0
            for name in names:
                row = int(name.split('.')[0].split('_')[4])
                col = int(name.split('.')[0].split('_')[5])

                filename = "Traj_{}_{}_{}.csv".format(prefix, row, col)
                local_name = local_folder+'/'+filename

                if counter == 0:
                    to_add = ut.csv_to_pd(local_name)
                    to_add['X'] = to_add['X'] + ires*col
                    to_add['Y'] = ires - to_add['Y'] + ires*(3-row)
                    merged = msd.all_msds2(to_add, frames=frames)
                else: 

                    if merged.shape[0] > 0:
                        to_add = ut.csv_to_pd(local_name)
                        to_add['X'] = to_add['X'] + ires*col
                        to_add['Y'] = ires - to_add['Y'] + ires*(3-row)
                        to_add['Track_ID'] = to_add['Track_ID'] + max(merged['Track_ID']) + 1
                    else:
                        to_add = ut.csv_to_pd(local_name)
                        to_add['X'] = to_add['X'] + ires*col
                        to_add['Y'] = ires - to_add['Y'] + ires*(3-row)
                        to_add['Track_ID'] = to_add['Track_ID']

                    merged = merged.append(msd.all_msds2(to_add, frames=frames))
                counter = counter + 1

            merged.to_csv(msd_file)
            aws.upload_s3(msd_file, remote_folder+'/'+msd_file)
            merged_ft = ft.calculate_features(merged)
            merged_ft.to_csv(ft_file)
         
            aws.upload_s3(ft_file, remote_folder+'/'+ft_file)
            
            #Plots          
            features = ('AR', 'D_fit', 'alpha', 'MSD_ratio', 'Track_ID', 'X', 'Y', 'asymmetry1', 'asymmetry2', 'asymmetry3',
                        'boundedness', 'efficiency', 'elongation', 'fractal_dim', 'frames', 'kurtosis', 'straightness', 'trappedness')
            vmin = (1.36, 0.015, 0.72, -0.09, 0, 0, 0, 0.5, 0.049, 0.089, 0.0069, 0.65, 0.26, 1.28, 0, 1.66, 0.087, -0.225)
            vmax = (3.98, 2.6, 2.3, 0.015, max(merged_ft['Track_ID']), 2048, 2048, 0.99, 0.415, 0.53,
                    0.062, 3.44, 0.75, 1.79, 650, 3.33, 0.52, -0.208)
            die = {'features': features,
                   'vmin': vmin,
                   'vmax': vmax}
            di = pd.DataFrame(data=die) 
            for i in range(0, di.shape[0]):
                hm.plot_heatmap(prefix, feature=di['features'][i], vmin=di['vmin'][i], vmax=di['vmax'][i])
                hm.plot_scatterplot(prefix, feature=di['features'][i], vmin=di['vmin'][i], vmax=di['vmax'][i])
            
            hm.plot_trajectories(prefix)
            hm.plot_histogram(prefix)
            hm.plot_particles_in_frame(prefix)
            gmean1, gSEM1 = hm.plot_individual_msds(prefix, alpha=0.05)