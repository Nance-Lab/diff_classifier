import os
import numpy as np
import diff_classifier.msd as msd
import diff_classifier.features as ft
import diff_classifier.heatmaps as hm


def test_voronoi_finite_polygons_2d():
    print()


def test_plot_heatmap():
    prefix = 'test'
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)

    dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
    msds = msd.all_msds2(dataf, frames=100)
    msds.to_csv(msd_file)
    feat = ft.calculate_features(msds)
    feat.to_csv(ft_file)
    
    hm.plot_heatmap(prefix, resolution=520, rows=1, cols=1, figsize=(6,5), upload=False)
    assert os.path.isfile('hm_asymmetry1_{}.png'.format(prefix))
    

def test_plot_scatterplot():
    prefix = 'test'
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)

    dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
    msds = msd.all_msds2(dataf, frames=100)
    msds.to_csv(msd_file)
    feat = ft.calculate_features(msds)
    feat.to_csv(ft_file)

    hm.plot_scatterplot(prefix, resolution=400, rows=1, cols=1, dotsize=120, upload=False)
    assert os.path.isfile('scatter_asymmetry1_{}.png'.format(prefix))

    
def test_plot_trajectories():
    prefix = 'test'
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)

    dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
    msds = msd.all_msds2(dataf, frames=100)
    msds.to_csv(msd_file)
    feat = ft.calculate_features(msds)
    feat.to_csv(ft_file)

    hm.plot_trajectories(prefix, resolution=520, rows=1, cols=1, upload=False)
    assert os.path.isfile('traj_{}.png'.format(prefix))
    

def test_plot_histogram():
    print()
    
    
def test_plot_individual_msds():
    print()


def test_plot_particles_in_frame():
    prefix = 'test'
    msd_file = 'msd_{}.csv'.format(prefix)
    ft_file = 'features_{}.csv'.format(prefix)

    dataf = msd.random_traj_dataset(nparts=10, ndist=(1, 1), seed=3)
    msds = msd.all_msds2(dataf, frames=100)
    msds.to_csv(msd_file)
    feat = ft.calculate_features(msds)
    feat.to_csv(ft_file)
    
    hm.plot_particles_in_frame(prefix, x_range=100, y_range=20, upload=False)
    assert os.path.isfile('in_frame_{}.png'.format(prefix))