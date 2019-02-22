import Transformation as tf
import numpy as np
import DroneAndMarker

dam = DroneAndMarker.DroneAndMarker()

# dam.set_t_world_drone(Transformation.Transformation.from_rpy(0.0, 0.2, 0.2,
#                                                              translation=np.array([-1.5, 0.2, 1.1])))
#
# marker_corners_f_image = dam.project_points()
#
# dam.solve_pnp(marker_corners_f_image)
#
# t_world_drone = dam.get_t_world_drone()
# drone_rpy_f_world = t_world_drone.as_rpy()
# drone_xyz_f_world = t_world_drone.translation

angle_count = 11
angle_bound = 0.5
dist_count = 3
dist_bound = 0.25
for ir in range(angle_count):
    r = ir * (2.0 * angle_bound) / (angle_count - 1) - angle_bound
    for ip in range(angle_count):
        p = ip * (2.0 * angle_bound) / (angle_count - 1) - angle_bound
        for iyaw in range(angle_count):
            yaw = iyaw * (2.0 * angle_bound) / (angle_count - 1) - angle_bound
            for id in range(dist_count):
                d = id * (2.0 * dist_bound) / (dist_count - 1) - dist_bound
                x = -1.5 + d
                y = d
                z = 1.0 + d
                dam.set_t_world_drone(tf.Transformation.from_rpy(r, p, yaw,  translation=np.array([x, y, z])))
                marker_corners_f_image = dam.project_points()
                dam.solve_pnp(marker_corners_f_image)
                t_world_drone = dam.get_t_world_drone()
                r_test, p_test, yaw_test = t_world_drone.as_rpy()
                x_test, y_test, z_test = t_world_drone.as_translation()

                if not np.allclose([r, p, yaw, x, y, z], [r_test, p_test, yaw_test, x_test, y_test, z_test],
                                   atol=1.e-6):
                    xxxx = 10
                yyy = 10
