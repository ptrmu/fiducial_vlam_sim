import cv2
import Transformation
import numpy as np

# This sample code demonstrates how to use the output of projectPoints as input to solvePnp.
# It does nothing useful except demonstrate how to call the functions and set up the transformations.

# The world frame axes are oriented in the ROS convention with x->east, y->north, z->up

# Pick some camera calibration values
camera_matrix = np.array([
    [921.17070200000001, 0., 459.90435400000001],
    [0., 919.01837699999999, 351.23830099999998],
    [0., 0., 1.]
])
dist_coeffs = np.array([-0.033458000000000002, 0.105152, 0.001256, -0.0066470000000000001, 0.])


# The marker class calculates corner points in the world frame.
class Marker:
    def __init__(self, pose_f_world, marker_len=0.162717998):
        self.pose_f_world = pose_f_world  # a Transformation object
        self.marker_len = marker_len

    # def corners_world(self):
    #     corners_marker = self.corners_marker()
    #     return self.pose_f_world.transform_vectors(corners_marker)

    def corners_marker(self):
        m2 = self.marker_len / 2.0
        return np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T


# Position the marker in the world frame. Here we choose an orientation with the marker in the world y,z plane
# facing in the world -x direction. The axes map: marker x = world -y, marker y = world z, marker z = world -x
t_world_marker = Transformation.Transformation.from_rpy(np.pi / 2., 0., -np.pi / 2., translation=np.array([0., 0., 1.]))
v_f_world = t_world_marker.transform_vectors(np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]).T)
marker_f_world = Marker(t_world_marker)

# Position the camera on the drone. The drone's x axis points in the forward direction, its y axis points to
# the left side and the z axis points up. The camera's z axis points in the direction the camera is looking,
# its x axis points to the right in the image and the y axis points down in the image.
t_drone_camera = Transformation.Transformation.from_rpy(-np.pi / 2, 0., -np.pi / 2)
v_f_drone = t_drone_camera.transform_vectors(np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]).T)

# Pick a location for the drone in the world frame. It needs to be looking at the marker for the calculations
# to work but this is easy if the drone is positioned on the -x axis and it is looking in the +x direction.
# These are the parameters to change to simulate the drone moving around.
# t_world_drone_sim = Transformation.Transformation.from_rpy(0.0, 0.0, 0.0, translation=np.array([-1.5, 0.0, 1.0]))
t_world_drone_sim = Transformation.Transformation.from_rpy(0., 0., 0., translation=np.array([-1.5, 0.2, 1]))

# The camera to world transformation is what we will find from solvePnp. It should match the following transformation.
t_world_camera_sim = t_world_drone_sim \
    .as_right_combined(t_drone_camera)

# This is the transformation that projectPoints will use when figuring out where the marker corners are in the image.
t_camera_marker_sim = t_drone_camera.as_inverse() \
    .as_right_combined(t_world_drone_sim.as_inverse()) \
    .as_right_combined(t_world_marker)

# The 3D coordinates of the marker corners in the marker frame
marker_corners_f_marker = marker_f_world.corners_marker().T

# Project those corners into the image
rvec_sim, tvec_sim = t_camera_marker_sim.as_rvec_tvec()
imgpts, _ = cv2.projectPoints(marker_corners_f_marker,
                              rvec_sim, tvec_sim,
                              camera_matrix, dist_coeffs)

# Given the location of the corners in the image, find the pose of the marker in the camera frame.
ret, rvecs, tvecs = cv2.solvePnP(marker_corners_f_marker, imgpts, camera_matrix, dist_coeffs)
t_camera_marker = Transformation.Transformation.from_rodrigues(rvecs[:, 0], translation=tvecs[:, 0])

# Figure out the pose of the camera in the world frame.
t_world_camera = t_world_marker \
    .as_right_combined(t_camera_marker.as_inverse())

# The calculated results should be the same as the simulated results for different drone poses.
camera_xyz_world = t_world_camera.transform_vectors(np.array([[0.], [0.], [0.]]))
camera_xyz_world_sim = t_world_camera_sim.transform_vectors(np.array([[0.], [0.], [0.]]))
camera_rpy_world = t_world_camera.as_rpy()
camera_rpy_world_sim = t_world_camera_sim.as_rpy()

# Figure out the pose of the drone in the world frame.
t_world_drone = t_world_marker \
    .as_right_combined(t_camera_marker.as_inverse()) \
    .as_right_combined(t_drone_camera.as_inverse())

drone_xyz_world = t_world_drone.transform_vectors(np.array([[0.], [0.], [0.]]))
drone_xyz_world_sim = t_world_drone_sim.transform_vectors(np.array([[0.], [0.], [0.]]))
drone_rpy_world = t_world_drone.as_rpy()
drone_rpy_world_sim = t_world_drone_sim.as_rpy()


xxx = 10
