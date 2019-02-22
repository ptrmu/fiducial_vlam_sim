
# import cv2
# import numpy as np
# import transformations as tf
#
#
# class Marker:
#     def __init__(self, pose_world, marker_len=0.162717998):
#         self.pose_world = pose_world  # a tf matrix that transforms from Marker to World frames
#         self.marker_len = marker_len
#
#     def corners_world(self):
#         corners_marker = self.corners_marker()
#         return self.pose_world @ corners_marker
#
#     def corners_marker(self):
#         m2 = self.marker_len / 2.0
#         return np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T
#
#
# marker_pose_world = tf.rotation_matrix(np.pi/2., [0, 0, 1]) @ \
#                     tf.rotation_matrix(-np.pi/2., [1, 0, 0]) + \
#                     tf.translation_matrix([0., 0., 1.])
# marker_world = Marker(marker_pose_world)
#
# print(marker_world.corners_world())
