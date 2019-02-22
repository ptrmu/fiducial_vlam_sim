#
# import cv2
# import Transformation
# import numpy as np
# import quaternion
# import matplotlib.pyplot as plt
#
#
# class CameraInfo:
#     def __init__(self, f, half_width):
#         self.f = f
#         self.half_width = half_width
#         self.K = np.array([
#             [self.f, 0, 0],
#             [0, self.f, 0],
#             [0, 0, 1]], dtype=np.float32)
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
# # FOCAL_LENGTH = 0.1
# #
# # K = np.array([
# #     [FOCAL_LENGTH, 0, 0],
# #     [0, FOCAL_LENGTH, 0],
# #     [0, 0, 1]], dtype=np.float32)
# #
# #
# # ret, rvecs, tvecs = cv2.solvePnP(
# #     self.pattern.object_points, corners,
# #     K, np.zeros(5))
#
# # http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
# # https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
# # https://github.com/matthew-brett/transforms3d
#
#
# class Marker:
#     def __init__(self, pose_f_world, marker_len=0.162717998):
#         self.pose_f_world = pose_f_world  # a Transformation object
#         self.marker_len = marker_len
#
#     def corners_world(self):
#         corners_marker = self.corners_marker()
#         return self.pose_f_world.transform_vectors(corners_marker)
#
#     def corners_marker(self):
#         m2 = self.marker_len / 2.0
#         return np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T
#
# # qa = quaternion.from_euler_angles(0., 0., -np.pi/4)
#
# # qa = rpy_to_quaternion(0., 0.2, 0.1)
# # rot = quaternion.as_rotation_matrix(qa)
# # rvec,_ = cv2.Rodrigues(rot)
#
#
# # t = Transformation.from_rpy(2., 1.5, -.5)
# # rvec = t.as_rodrigues()
# # t1 = Transformation.from_rodrigues(rvec)
# # r, p, y = t1.as_rpy()
# # q0 = rpy_to_quaternion(r, p, y)
#
# # For OpenCV:
# # The camera 3D coordinate system has the camera looking out the z axis.
# # If one looks along this z axis from the origin, the x is to the right
# # and the y axis is down.
# # The image 2D coordinate system has x to the right and y down.
#
# t_drone_camera = Transformation.Transformation.from_rpy(-np.pi/2, 0., -np.pi/2)
# v_drone = t_drone_camera.transform_vectors(np.array([[0.], [0.], [3.]]))
#
# t_world_drone = Transformation.Transformation.from_rpy(0., 0., 0., translation=np.array([-1.4, 0., 1.]))
#
# t_world_marker = Transformation.Transformation.from_rpy(np.pi/2., 0., -np.pi/2., translation=np.array([0., 0., 1.]))
#
# t_camera_marker = t_drone_camera.as_inverse() \
#     .as_right_combined(t_world_drone.as_inverse()) \
#     .as_right_combined(t_world_marker)
# t_camera_marker_r, t_camera_marker_p, t_camera_marker_y = t_camera_marker.as_rpy()
#
# rvec_t_camera_marker, tvec_t_camera_marker = t_camera_marker.as_rvec_tvec()
#
# real_rvec_t_camera_marker = np.array([1.3375433, -1.26168659, 0.98199645])
# real_tvec_t_camera_marker = np.array([0.41788314, 0.65751337, 1.43586492])
# real_t_camera_marker = Transformation.Transformation.from_rodrigues(real_rvec_t_camera_marker, translation=real_tvec_t_camera_marker)
#
# real_t_world_camera = t_world_marker.as_right_combined(real_t_camera_marker.as_inverse())
# real_t_world_drone = real_t_world_camera.as_right_combined(t_drone_camera.as_inverse())
# real_t_world_drone_r, real_t_world_drone_p, real_t_world_drone_y = real_t_world_drone.as_rpy()
#
# marker_f_world = Marker(t_world_marker)
#
# rvec, tvec = t_camera_marker.as_rvec_tvec()
#
# # FOCAL_LENGTH = 500
# # K = np.array([
# #     [FOCAL_LENGTH, 0, 0],
# #     [0, FOCAL_LENGTH, 0],
# #     [0, 0, 1]], dtype=np.float32)
# camera_matrix = np.array([
#     [921.17070200000001, 0., 459.90435400000001],
#     [0., 919.01837699999999, 351.23830099999998],
#     [459.90435400000001, 0., 1.]
# ])
# dist_coeffs = np.array([-0.033458000000000002, 0.105152, 0.001256, -0.0066470000000000001, 0.])
#
# # p = np.float32([[1, 0, .5], [0, 1, .5], [0, 0, 2], [1, 0, 2], [0, 1, 2]])
# # print(p.dtype)
# p = marker_f_world.corners_world().T
#
# try:
#     imgpts, _ = cv2.projectPoints(p, rvec, tvec, camera_matrix, dist_coeffs)
# except cv2.error:
#     print(cv2.error or cv2.error.shape())
#
# imgpts2 = imgpts[:, 0, :]
#
# ret, rvecs, tvecs = cv2.solvePnP(
#     p, imgpts2,
#     camera_matrix, dist_coeffs)
#
# corners_f_map = np.array([
#     [0., 0.081358999013900757, 1.0813589990139008],
#     [0., -0.081358999013900757, 1.0813589990139008],
#     [0., -0.081358999013900757, 0.91864100098609924],
#     [0., 0.081358999013900757, 0.91864100098609924]
# ])
# corners_f_image = np.array([
#     [642.357605, 62.8794136],
#     [762.332214, 51.8663559],
#     [765.440979, 172.22583],
#     [647.891907, 181.043488]
# ])
#
# ret1, rvecs1, tvecs1 = cv2.solvePnP(
#     corners_f_map, corners_f_image,
#     camera_matrix, dist_coeffs)
#
# xxx = 10
