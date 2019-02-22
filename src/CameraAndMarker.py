# import cv2
# import Transformation as tf
# import numpy as np
#
#
# class CameraAndMarker:
#     def __init__(self, t_world_camera=None, t_world_marker=None, marker_len=0.162717998):
#         self.marker_len = marker_len
#
#         # Initialize t_world_camera
#         self.t_world_camera = t_world_camera if t_world_camera is not None else \
#             tf.Transformation.from_rpy(-np.pi / 2, 0., -np.pi / 2, translation=np.array([-1.5, 0., 1.]))
#
#         # Initialize t_world_marker
#         self.t_world_marker = t_world_marker if t_world_marker is not None else \
#             tf.Transformation.from_rpy(np.pi / 2., 0., -np.pi / 2., translation=np.array([0., 0., 1.]))
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
#     def get_t_world_camera(self):
#         return self.t_world_camera
#
#     def get_t_world_marker(self):
#         return self.t_world_marker
#
#     def _get_corners_f_marker(self):
#         m2 = self.marker_len / 2.0
#         marker_corners_f_marker = np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T
#         return marker_corners_f_marker
#
#     def project_points(self):
#
#         # get corners in the marker frame
#         corners_f_marker = self._get_corners_f_marker().T
#
#         # calculate rvec, tvec from t_camera_marker
#         t_camera_marker = self.t_world_camera.as_inverse() \
#             .as_right_combined(self.t_world_marker)
#         rvec, tvec = t_camera_marker.as_rvec_tvec()
#
#         # project the points using t_camera_marker
#         corners_f_image, _ = cv2.projectPoints(corners_f_marker,
#                                                       rvec, tvec,
#                                                       self.camera_matrix, self.dist_coeffs)
#         return corners_f_image
#
#     def project_points_normal(self, std, num_samples):
#         corners_f_image = self.project_points().reshape(8, 1)
#
#         # create a group of image points that have a normal distribution
#         corners_f_images = corners_f_image + np.random.normal(0, std ** 2, size=[8, num_samples])
#
#         return corners_f_image, corners_f_images
#
#     def t_world_xxxs_normal(self, cov, num_samples, t_world_xxx_cvec):
#         if cov is None:
#             return np.tile(t_world_xxx_cvec, (1, num_samples))
#
#         return np.random.multivariate_normal(t_world_xxx_cvec[:, 0], cov, num_samples).T
#
#     def t_world_markers_normal(self, cov, num_samples):
#         return self.t_world_xxxs_normal(cov, num_samples, self.t_world_marker.as_cvec())
#
#     def t_world_cameras_normal(self, cov, num_samples):
#         return self.t_world_xxxs_normal(cov, num_samples, self.t_world_camera.as_cvec())
#
#     def solve_pnp(self, marker_corners_f_image):
#         # get corners in the marker frame
#         corners_f_marker = self._get_corners_f_marker().T
#
#         # Given the location of the corners in the image, find the pose of the marker in the camera frame.
#         ret, rvecs, tvecs = cv2.solvePnP(corners_f_marker, marker_corners_f_image,
#                                          self.camera_matrix, self.dist_coeffs)
#
#         t_camera_marker = tf.Transformation.from_rodrigues(rvecs[:, 0], translation=tvecs[:, 0])
#         self.t_world_camera = self.t_world_marker \
#             .as_right_combined(t_camera_marker.as_inverse())
#
#     def solve_many_t_world_cameras(self, corners_f_images, t_world_markers_cvec):
#         # Return a n t_world_cameras in cvec format (6xn) given:
#         #   1) an array of corners_f_images in a 8xn array
#         #   2) an array of t_world_markers in cvec format (6xn)
#
#         t_world_cameras_cvec = np.zeros(t_world_markers_cvec.shape)
#
#         # get corners in the marker frame
#         corners_f_marker = self._get_corners_f_marker().T
#
#         for i in range(t_world_markers_cvec.shape[1]):
#             # Given the location of the corners in the image, find the pose of the marker in the camera frame.
#             ret, rvecs, tvecs = cv2.solvePnP(corners_f_marker, corners_f_images[:, i].reshape(4, 2),
#                                              self.camera_matrix, self.dist_coeffs)
#
#             t_camera_marker = tf.Transformation.from_rodrigues(rvecs[:, 0], translation=tvecs[:, 0])
#             t_world_marker = tf.Transformation.from_cvec(t_world_markers_cvec[:, i])
#             t_world_camera = t_world_marker \
#                 .as_right_combined(t_camera_marker.as_inverse())
#
#             t_world_cameras_cvec[:, i] = t_world_camera.as_cvec().T
#
#         t_world_camera_mu_cvec = np.mean(t_world_cameras_cvec, axis=1).reshape(6, 1)
#         t_world_camera_cov = np.cov(t_world_cameras_cvec)
#
#         return t_world_camera_mu_cvec, t_world_camera_cov, t_world_cameras_cvec
#
#     def solve_many_t_world_markers(self, corners_f_images, t_world_cameras_cvec):
#         # Return a list of n t_world_marker given:
#         #   1) an array of corners_f_images in a 4x2xn array
#         #   2) an n element long list of t_world_camera
#
#         t_world_markers_cvec = np.zeros(t_world_cameras_cvec.shape)
#
#         # get corners in the marker frame
#         corners_f_marker = self._get_corners_f_marker().T
#
#         for i in range(t_world_cameras_cvec.shape[1]):
#             # Given the location of the corners in the image, find the pose of the marker in the camera frame.
#             ret, rvecs, tvecs = cv2.solvePnP(corners_f_marker, corners_f_images[:, i].reshape(4, 2),
#                                              self.camera_matrix, self.dist_coeffs)
#
#             t_camera_marker = tf.Transformation.from_rodrigues(rvecs[:, 0], translation=tvecs[:, 0])
#             t_world_marker = tf.Transformation.from_cvec(t_world_cameras_cvec[:, i]) \
#                 .as_right_combined(t_camera_marker)
#
#             t_world_markers_cvec[:, i] = t_world_marker.as_cvec().T
#
#         t_world_marker_mu_cvec = np.mean(t_world_markers_cvec, axis=1).reshape(6, 1)
#         t_world_marker_cov = np.cov(t_world_markers_cvec)
#
#         return t_world_marker_mu_cvec, t_world_marker_cov, t_world_markers_cvec
#
