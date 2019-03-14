
from CameraOrMarker import *


class Scenarios:
    def __init__(self, scenario):
        self.num_generated_samples = 217
        self.std_corners_f_image = 0.3

        self.marker_len = 0.162717998

        # generate_solve_pnp_inputs options
        self.use_sigma_points = False
        self.use_dist_params = True

        # Pick some camera calibration values
        self.camera_matrix = np.array([
            [921.17070200000001, 0., 459.90435400000001],
            [0., 919.01837699999999, 351.23830099999998],
            [0., 0., 1.]
        ])
        self.dist_coeffs = np.array([-0.033458000000000002, 0.105152, 0.001256, -0.0066470000000000001, 0.])

        self.rpy_world_marker_a = (np.pi / 2., 0., -np.pi / 2.)
        self.xyz_world_marker_a = (0., -.5, 1.)
        self.rpy_world_marker_b = (np.pi / 2., 0., -np.pi / 2.)
        self.xyz_world_marker_b = (0., .5, 1.)
        self.rpy_world_marker = self.rpy_world_marker_a
        self.xyz_world_marker = self.xyz_world_marker_a
        self.rpy_world_camera_a = (-np.pi / 2, 0., -np.pi / 2)
        self.xyz_world_camera_a = (-2.5, -.0, 0.5)
        self.rpy_world_camera_b = (-np.pi / 2, 0., -np.pi / 2)
        self.xyz_world_camera_b = (-2.5, 0.5, .5)
        self.rpy_world_camera = self.rpy_world_camera_a
        self.xyz_world_camera = self.xyz_world_camera_a
        self.ident = "looking along x"
        if scenario == 1:
            self.rpy_world_marker_a = (0., 0., 0.)
            self.xyz_world_marker_a = (0., 0., 1.)
            self.rpy_world_marker_b = (0., 0., 0.)
            self.xyz_world_marker_b = (0., 0., 1.)
            self.rpy_world_camera = (0., 0., 0.)
            self.xyz_world_camera = (-1., 0., -1.5)
            self.ident = "looking up"

    @property
    def marker_a(self):
        return CameraOrMarker.marker_from_rpy(self.rpy_world_marker_a, self.xyz_world_marker_a)

    @property
    def marker_b(self):
        return CameraOrMarker.marker_from_rpy(self.rpy_world_marker_b, self.xyz_world_marker_b)

    @property
    def camera_a(self):
        return CameraOrMarker.camera_from_rpy(self.rpy_world_camera_a, self.xyz_world_camera_a)

    @property
    def camera_b(self):
        return CameraOrMarker.camera_from_rpy(self.rpy_world_camera_b, self.xyz_world_camera_b)

    def as_param_str(self):
        return "std_corners={} dist_params={} sigma_points={}".format(
            self.std_corners_f_image, self.use_dist_params, self.use_sigma_points)
