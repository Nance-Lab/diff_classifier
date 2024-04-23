# import os
# import numpy as np
# import numpy.testing as npt
# from scipy.spatial import Voronoi
# import matplotlib as mpl
# mpl.use('Agg')
# import diff_classifier.msd as msd
# import diff_classifier.features as ft
# import diff_classifier.heatmaps as hm


# def test_voronoi_finite_polygons_2d():
#     prefix = 'test'
#     msd_file = 'msd_{}.csv'.format(prefix)
#     ft_file = 'features_{}.csv'.format(prefix)

#     dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
#     msds = msd.all_msds2(dataf, frames=100)
#     msds.to_csv(msd_file)
#     feat = ft.calculate_features(msds)
#     feat.to_csv(ft_file)

#     xs = feat['X'].astype(int)
#     ys = feat['Y'].astype(int)
#     points = np.zeros((xs.shape[0], 2))
#     points[:, 0] = xs
#     points[:, 1] = ys

#     vor = Voronoi(points)
#     regions, vertices = hm.voronoi_finite_polygons_2d(vor)

#     npt.assert_equal(243.8, np.round(np.mean(vertices), 1))


# def test_plot_heatmap():
#     prefix = 'test'
#     msd_file = 'msd_{}.csv'.format(prefix)
#     ft_file = 'features_{}.csv'.format(prefix)

#     dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
#     msds = msd.all_msds2(dataf, frames=100)
#     msds.to_csv(msd_file)
#     feat = ft.calculate_features(msds)
#     feat.to_csv(ft_file)

#     hm.plot_heatmap(prefix, resolution=520, rows=1, cols=1, figsize=(6,5), upload=False)
#     assert os.path.isfile('hm_asymmetry1_{}.png'.format(prefix))


# def test_plot_scatterplot():
#     prefix = 'test'
#     msd_file = 'msd_{}.csv'.format(prefix)
#     ft_file = 'features_{}.csv'.format(prefix)

#     dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
#     msds = msd.all_msds2(dataf, frames=100)
#     msds.to_csv(msd_file)
#     feat = ft.calculate_features(msds)
#     feat.to_csv(ft_file)

#     hm.plot_scatterplot(prefix, resolution=400, rows=1, cols=1, dotsize=120, upload=False)
#     assert os.path.isfile('scatter_asymmetry1_{}.png'.format(prefix))


# def test_plot_trajectories():
#     prefix = 'test'
#     msd_file = 'msd_{}.csv'.format(prefix)
#     ft_file = 'features_{}.csv'.format(prefix)

#     dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
#     msds = msd.all_msds2(dataf, frames=100)
#     msds.to_csv(msd_file)
#     feat = ft.calculate_features(msds)
#     feat.to_csv(ft_file)

#     hm.plot_trajectories(prefix, resolution=520, rows=1, cols=1, upload=False)
#     assert os.path.isfile('traj_{}.png'.format(prefix))


# def test_plot_histogram():
#     prefix = 'test'
#     msd_file = 'msd_{}.csv'.format(prefix)
#     ft_file = 'features_{}.csv'.format(prefix)

#     dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
#     msds = msd.all_msds2(dataf, frames=100)
#     msds.to_csv(msd_file)
#     feat = ft.calculate_features(msds)
#     feat.to_csv(ft_file)

#     hm.plot_histogram(prefix, fps=1, umppx=1, frames=100, frame_interval=5, frame_range=5, y_range=10, upload=False)
#     assert os.path.isfile('hist_{}.png'.format(prefix))


# #------------------------------------------------------------------------------------------------------------------------
# ## plot individual MSDs has been moved to diff_visualizer

# # def test_plot_individual_msds():
# #     prefix = 'test'
# #     msd_file = 'msd_{}.csv'.format(prefix)
# #     ft_file = 'features_{}.csv'.format(prefix)

# #     dataf = msd.random_traj_dataset(nparts=30, ndist=(1, 1), seed=3)
# #     msds = msd.all_msds2(dataf, frames=100)
# #     msds.to_csv(msd_file)
# #     feat = ft.calculate_features(msds)
# #     feat.to_csv(ft_file)

# #     geomean, gSEM = hm.plot_individual_msds(prefix, umppx=1, fps=1, y_range=400, alpha=0.3, upload=False)
# #     npt.assert_almost_equal(339.9, np.round(np.sum(geomean), 1))
# #     npt.assert_almost_equal(35.3, np.round(np.sum(gSEM), 1))


# def test_plot_particles_in_frame():
#     prefix = 'test'
#     msd_file = 'msd_{}.csv'.format(prefix)
#     ft_file = 'features_{}.csv'.format(prefix)

#     dataf = msd.random_traj_dataset(nparts=10, ndist=(1, 1), seed=3)
#     msds = msd.all_msds2(dataf, frames=100)
#     msds.to_csv(msd_file)
#     feat = ft.calculate_features(msds)
#     feat.to_csv(ft_file)

#     hm.plot_particles_in_frame(prefix, x_range=100, y_range=20, upload=False)
#     assert os.path.isfile('in_frame_{}.png'.format(prefix))
