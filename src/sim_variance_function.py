#
# import Transformation as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import CameraAndMarker
# from plot_transformation_covariance import *
#
# # plot how the pose variance of a camera varies with distance
#
# # Simulate a fixed marker with no uncertainty. Figure out the uncertainty in the camera pose given
# # some uncertainty in the image points.
#
# # set some parameters
# num_samples = 217
# std_corners_f_image = 0.4
#
# scenario = 2
#
# if scenario == 1:
#     rpy_world_marker = (0., 0., 0.)
#     xyz_world_marker = (0., 0., 1.)
#     rpy_world_camera = (0., 0., 0.)
#     xyz_world_camera = (-1., 0., -1.5)
#     ident = "looking up"
# elif scenario == 2:
#     rpy_world_marker = (np.pi / 2., 0., -np.pi / 2.)
#     xyz_world_marker = (0., 0., 1.)
#     rpy_world_camera = (-np.pi / 2, 0., -np.pi / 2)
#     xyz_world_camera = (-0.5, 0., 1.0)
#     ident = "looking along x"
#
# num_calcs = 31
# d_sim_range = 3.0
# ds = np.zeros((1, num_calcs))
# stds = np.zeros((6, num_calcs))
# for i_d_sim in range(num_calcs):
#     d_sim = i_d_sim * d_sim_range / (num_calcs - 1)
#     xyz_world_camera_calc = (xyz_world_camera[0] - d_sim, xyz_world_camera[1], xyz_world_camera[2])
#     ds[0, i_d_sim] = xyz_world_camera_calc[0]
#
#     # the fixed/known marker pose and simulated camera pose
#     t_world_marker = tf.Transformation.from_rpy(*rpy_world_marker, translation=np.array(xyz_world_marker))
#     t_world_camera_sim = tf.Transformation.from_rpy(*rpy_world_camera, translation=np.array(xyz_world_camera_calc))
#
#     # get the projected points of this marker.
#     cam = CameraAndMarker.CameraAndMarker(t_world_camera=t_world_camera_sim, t_world_marker=t_world_marker)
#     corners_f_image = cam.project_points()
#
#     # make this into a column vector
#     corners_f_image = corners_f_image.reshape(8, 1)
#
#     # create a group of image points that have a normal distribution
#     corners_f_images = corners_f_image + np.random.normal(0, std_corners_f_image**2, size=[8, num_samples])
#
#     # create a list of t_world_marker. Since there is no uncertainty, the list can contain
#     # references to the same transform.
#     t_world_marker_cvec = t_world_marker.as_cvec()
#     t_world_markers_cvec = np.tile(t_world_marker_cvec, (1, num_samples))
#
#     # calculate the list of t_world_camera.
#     t_world_camera_mu_cvec, t_world_camera_cov, t_world_cameras_cvec = \
#         cam.solve_many_t_world_cameras(corners_f_images, t_world_markers_cvec)
#
#     # save the results
#     std = np.sqrt(np.diag(t_world_camera_cov))
#     stds[:, i_d_sim] = std
#
# plt.plot(ds[0, :], stds[0, :])
# plt.show()
#
# xxxx = 10
