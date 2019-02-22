# import cv2
# import Transformation as tf
# import numpy as np
#
#
# class DroneAndMarker:
#     def __init__(self, t_world_drone=None, t_world_camera=None, t_world_marker=None, marker_len=0.162717998):
#         self.marker_len = marker_len
#
#         # Adjustable transforms
#         self.t_world_camera = None
#         self.t_world_marker = None
#
#         # Fixed transform
#         self.t_drone_camera = tf.Transformation.from_rpy(-np.pi / 2, 0., -np.pi / 2)
#         self.t_camera_drone = self.t_drone_camera.as_inverse()
#
#         # Initialize t_world_camera
#         if t_world_camera is not None:
#             self.t_world_camera = t_world_camera
#         elif t_world_drone is not None:
#             self.set_t_world_drone(t_world_drone)
#         else:
#             t_world_drone = tf.Transformation.from_rpy(0.0, 0.0, 0.0,
#                                                        translation=np.array([-1.5, 0., 1.]))
#             self.set_t_world_drone(t_world_drone)
#
#         # Initialize t_world_marker
#         if t_world_marker is not None:
#             self.t_world_marker = t_world_marker
#         else:
#             self.t_world_marker = tf.Transformation.from_rpy(np.pi / 2., 0., -np.pi / 2.,
#                                                              translation=np.array([0., 0., 1.]))
#
#         # Pick some camera calibration values
#         self.camera_matrix = np.array([
#             [921.17070200000001, 0., 459.90435400000001],
#             [0., 919.01837699999999, 351.23830099999998],
#             [0., 0., 1.]
#         ])
#         self.dist_coeffs = np.array([-0.033458000000000002, 0.105152, 0.001256, -0.0066470000000000001, 0.])
#
#     def set_t_world_camera(self, t_world_camera):
#         self.t_world_camera = t_world_camera
#
#     def set_t_world_marker(self, t_world_marker):
#         self.t_world_marker = t_world_marker
#
#     def set_t_world_drone(self, t_world_drone):
#         self.t_world_camera = t_world_drone.as_right_combined(self.t_drone_camera)
#
#     def get_t_world_drone(self):
#         return self.t_world_camera.as_right_combined(self.t_drone_camera.as_inverse())
#
#     def _calc_marker_corners_f_marker(self):
#         m2 = self.marker_len / 2.0
#         marker_corners_f_marker = np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T
#         return marker_corners_f_marker;
#
#     def project_points(self):
#         # calculate marker_corners_f_world
#         marker_corners_f_marker = self._calc_marker_corners_f_marker().T
#
#         # calculate rvec, tvec from t_camera_marker
#         t_camera_marker = self.t_world_camera.as_inverse() \
#             .as_right_combined(self.t_world_marker)
#         rvec, tvec = t_camera_marker.as_rvec_tvec()
#
#         # project the points using t_camera_marker
#         marker_corners_f_image, _ = cv2.projectPoints(marker_corners_f_marker,
#                                                       rvec, tvec,
#                                                       self.camera_matrix, self.dist_coeffs)
#         return marker_corners_f_image
#
#     def solve_pnp(self, marker_corners_f_image):
#         # calculate marker_corners_f_world
#         marker_corners_f_marker = self._calc_marker_corners_f_marker().T
#
#         # Given the location of the corners in the image, find the pose of the marker in the camera frame.
#         ret, rvecs, tvecs = cv2.solvePnP(marker_corners_f_marker, marker_corners_f_image,
#                                          self.camera_matrix, self.dist_coeffs)
#
#         t_camera_marker = tf.Transformation.from_rodrigues(rvecs[:, 0], translation=tvecs[:, 0])
#         self.t_world_camera = self.t_world_marker \
#             .as_right_combined(t_camera_marker.as_inverse())
