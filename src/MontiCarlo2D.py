#
# import scipy.stats as ss
# import numpy as np
# import matplotlib.pyplot as plt
#
# # x = np.arange(-10, 11)
# # xU, xL = x + 0.5, x - 0.5
# # prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
# # prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
# # nums = np.random.choice(x, size=10000, p=prob)
# # plt.hist(nums, bins=len(x))
# # plt.show()
#
# # np.random.normal(0, 0.1, 1)
#
# # angle theta = 0 points along the x axis. Positive angles are toward the y axis.
#
#
# class CameraInfo:
#     def __init__(self, f, half_width):
#         self.f = f
#         self.half_width = half_width
#
#     def image_points(self, camera_xyts_world, point_xys_world):
#         i_points = []
#
#         # Return the y coordinate of the point in the image plane of the camera.
#         # The camera points along the x axis of the camera pose.
#         for camera_xyt_world, point_xy_world in zip(camera_xyts_world, point_xys_world):
#
#             # First figure the position of the point in the camera coordinate system.
#             point_xy_offset = point_xy_world - camera_xyt_world[0:2]
#             theta = camera_xyt_world[2]
#             r = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
#             point_xy_camera = r @ point_xy_offset
#
#             # The x coordinate is the distance of the point from the focal point
#             # The y coordinate is the distance of the point from the camera axis
#
#             # If the point is closer to the camera than the image plane, f, the it can't be imaged
#             if point_xy_camera[0] <= self.f:
#                 continue
#
#             # use the pin-hole camera model to calculate the image_point
#             i_point = point_xy_camera[1] * self.f / point_xy_camera[0]
#
#             # If the calculated i_point is outside of the image plane then it can't be imaged
#             if np.abs(i_point) > self.half_width:
#                 continue
#             i_points.append(i_point)
#
#         return i_points
#
#
# class Pose2DWithCovariance:
#     def __init__(self, mu, cov):
#         self.mu = mu
#         self.cov = np.array(cov)  # 3x3 array, order: (x, y, theta)
#
#
# # return a numpy array of poses. Each row is a pose.
# # The columns are x, y, theta
# def generate_normal_poses(pose_with_covariance, size):
#     xyt = np.random.multivariate_normal(pose_with_covariance.mu, pose_with_covariance.cov, size)
#     return xyt
#
#
# num_samples = 100000
# target_width = .25
#
# camera_cov = [[1, 0, 0],
#               [0, 1, 0],
#               [0, 0, .0]]
# camera_mean = [0, 0, 0]
#
# camera_pwc_world = Pose2DWithCovariance(camera_mean, camera_cov)
# camera_xyts_world = generate_normal_poses(camera_pwc_world, num_samples)
#
# point_a_cov = [[1, 0, 0],
#                [0, 1, 0],
#                [0, 0, .1]]
# point_a_mean = [5, 0, 0]
#
# point_a_pwc_world = Pose2DWithCovariance(point_a_mean, point_a_cov)
# point_a_xyts_world = generate_normal_poses(point_a_pwc_world, num_samples)
# point_a_xys_world = point_a_xyts_world[:, 0:2]
#
# camera_info = CameraInfo(0.1, 0.1)
# ys_image = camera_info.image_points(camera_xyts_world, point_a_xys_world)
#
# plt.hist(ys_image, bins=41)
# plt.show()
#
# # plt.plot(xyts[:, 0], xyts[:, 1], xyts[:, 2], '.')
# # plt.axis('equal')
# # plt.show()
