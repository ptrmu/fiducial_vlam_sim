import numpy as np
from timeit import default_timer as timer
import MonteCarloSimulation
import TransformMath
import VisionMath
import Scenarios

tm = None
vm = None


def pool_initializer():
    global tm
    global vm
    mcs = MonteCarloSimulation.MonteCarloSimulation(1)
    tm = TransformMath.TransformMath(mcs)

    marker_len = 0.162717998
    camera_matrix = np.array([
        [921.17070200000001, 0., 459.90435400000001],
        [0., 919.01837699999999, 351.23830099999998],
        [0., 0., 1.]
    ])
    dist_coeffs = np.array([-0.033458000000000002, 0.105152, 0.001256, -0.0066470000000000001, 0.])

    vm = VisionMath.VisionMath(tm, marker_len, camera_matrix, dist_coeffs)


if __name__ == '__main__':
    def test():
        pool_initializer()

        sc = Scenarios.Scenarios(0)

        t_map_marker = tm.transformation_from_xyz_rpy(sc.xyz_world_marker_a, sc.rpy_world_marker_a)
        t_map_camera = tm.transformation_from_xyz_rpy(sc.xyz_world_camera_a, sc.rpy_world_camera_a)

        t_camera_map = tm.invert_transformation(t_map_camera)
        t_camera_marker = tm.transform_transformation(t_camera_map, t_map_marker)

        corners_f_image = vm.project_points(t_camera_marker)

        t_camera_marker_1 = vm.solve_pnp(corners_f_image, None)

        t = 5


    test()
