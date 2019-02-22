import Transformation as tf
import numpy as np
import CameraAndMarker

cam = CameraAndMarker.CameraAndMarker()

# marker_corners_f_image = cam.project_points()
#
# cam.solve_pnp(marker_corners_f_image)
#
# t_world_camera = cam.get_t_world_camera()
# camera_rpy_f_world = t_world_camera.as_rpy()
# camera_xyz_f_world = t_world_camera.translation


angle_count = 7
angle_bound = 0.3
dist_count = 7
dist_bound = 0.3
for ir in range(angle_count):
    r = ir * (2.0 * angle_bound) / (angle_count - 1) - angle_bound
    r -= np.pi / 2
    for ip in range(angle_count):
        p = ip * (2.0 * angle_bound) / (angle_count - 1) - angle_bound
        for iyaw in range(angle_count):
            yaw = iyaw * (2.0 * angle_bound) / (angle_count - 1) - angle_bound
            yaw -= np.pi / 2
            for idist in range(dist_count):
                dist = idist * (2.0 * dist_bound) / (dist_count - 1) - dist_bound
                x = -1.5 + dist
                y = dist
                z = 1.0 + dist
                cam.set_t_world_camera(tf.Transformation.from_rpy(r, p, yaw, translation=np.array([x, y, z])))
                corners_f_image = cam.project_points()
                cam.solve_pnp(corners_f_image)
                t_world_camera = cam.get_t_world_camera()
                r_test, p_test, yaw_test = t_world_camera.as_rpy()
                x_test, y_test, z_test = t_world_camera.as_translation()

                if not np.allclose([r, p, yaw, x, y, z], [r_test, p_test, yaw_test, x_test, y_test, z_test],
                                   atol=1.e-6):
                    xxxx = 10
                yyy = 10

xxx= 10