import Transformation as tf
import numpy as np
import CameraAndMarker
import Scenarios
from plot_transformation_covariance import *

# m=np.array([[1, 2, 3, 4], [5, 2,9,4], [2, 4, 6, 8]])
# print(np.cov(m))

# Simulate a fixed marker with no uncertainty. Figure out the uncertainty in the camera pose given
# some uncertainty in the image points.

# set some parameters
num_samples = 217
std_corners_f_image = 0.3

use_camera_distribution = False

plot_camera_data = False

sc = Scenarios.Scenarios(0)

# distribution of corner points
corners_f_image_a, corners_f_images_a = sc.cam_aa.project_points_normal(std_corners_f_image, num_samples)

# create a list of t_world_marker. Since there is no uncertainty, the list can contain
# references to the same transform.
t_world_markers_cvec = sc.cam_aa.t_world_markers_normal(None, num_samples)

# calculate the list of t_world_camera.
t_world_camera_mu_cvec, t_world_camera_cov, t_world_cameras_cvec = \
    sc.cam.solve_many_t_world_cameras(corners_f_images_a, t_world_markers_cvec)

if plot_camera_data:

    # plot the results
    title = "Fixed marker '{}', std_corners={}\nmarker_rpy:{}, marker_xyz:{}\ncamera_rpy{}, camera_xyz{}".format(
        sc.ident, std_corners_f_image,
        sc.rpy_world_marker, sc.xyz_world_marker,
        sc.rpy_world_camera, sc.xyz_world_camera)

    if True:
        plot_transformation_covariance(title, corners_f_image_a,
                                       t_world_camera_mu_cvec, t_world_camera_cov, t_world_cameras_cvec)

    else:
        mu = t_world_camera_mu_cvec
        cov = t_world_camera_cov
        t_world_camera_sim_cvec = np.random.multivariate_normal(mu[:, 0], cov, num_samples).T

        plot_transformation_covariance(title, corners_f_image,
                                       mu, cov, t_world_camera_sim_cvec)

# With the camera location now figure the second marker location
if use_camera_distribution:
    pass
else:
    pass

# distribution of corner points
corners_f_image, corners_f_images = sc.cam_ab.project_points_normal(std_corners_f_image, num_samples)

# # create a list of t_world_marker. Since there is no uncertainty, the list can contain
# # references to the same transform.
# t_world_cameras_cvec = sc.cam.t_world_cameras_normal(None, num_samples)

# calculate the list of t_world_camera.
t_world_marker_mu_cvec, t_world_marker_cov, t_world_markers_cvec = \
    sc.cam_ab.solve_many_t_world_markers(corners_f_images, t_world_cameras_cvec)

# plot the results
title = "two markers '{}', std_corners={}\nmarker_rpy_a:{}, marker_xyz_a:{}\ncamera_rpy{}, camera_xyz{}\nmarker_rpy_b:{}, marker_xyz_b:{}".format(
    sc.ident, std_corners_f_image,
    sc.rpy_world_marker_a, sc.xyz_world_marker_a,
    sc.rpy_world_camera, sc.xyz_world_camera,
    sc.rpy_world_marker_b, sc.xyz_world_marker_b)

if True:
    plot_transformation_covariance(title, corners_f_image,
                                   t_world_marker_mu_cvec, t_world_marker_cov, t_world_markers_cvec)

else:
    mu = t_world_camera_mu_cvec
    cov = t_world_camera_cov
    t_world_camera_sim_cvec = np.random.multivariate_normal(mu[:, 0], cov, num_samples).T

    plot_transformation_covariance(title, corners_f_image,
                                   mu, cov, t_world_camera_sim_cvec)




xxxx = 10
