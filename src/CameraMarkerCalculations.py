import cv2
from CameraOrMarker import *
from scipy.linalg import sqrtm


class SolvePnpInputs:
    def __init__(self, camera_not_marker, corners_f_images, t_world_xxx_cvecs):
        self.camera_not_marker = camera_not_marker  # t_world_xxx_cvecs are for a camera
        self.corners_f_images = corners_f_images
        self.t_world_xxx_cvecs = t_world_xxx_cvecs


class CameraMarkerCalculations:
    def __init__(self, scenario):
        self.sc = scenario

        # Constants for Sigma Point generation
        self.n0 = 8  # 8 coordinates of the corner points
        self.n1 = 14  # 8 + 6

        self.alpha = 1.0
        self.betta = 2.0
        self.k0 = 3.0 - self.n0
        self.k1 = 3.0 - self.n1

        self.lam0 = self.alpha ** 2 * (self.n0 + self.k0) - self.n0
        self.lam1 = self.alpha ** 2 * (self.n1 + self.k1) - self.n1
        self.gamma0 = np.sqrt(self.n0 + self.lam0)
        self.gamma1 = np.sqrt(self.n1 + self.lam1)

    def _get_corners_f_marker(self):
        m2 = self.sc.marker_len / 2.0
        marker_corners_f_marker = np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T
        return marker_corners_f_marker

    def _project_points(self, camera, marker):
        # get corners in the marker frame
        corners_f_marker = self._get_corners_f_marker().T

        # get the transforms
        t_world_camera = camera.t_world_xxx
        t_world_marker = marker.t_world_xxx

        # calculate rvec, tvec from t_camera_marker
        t_camera_marker = t_world_camera.as_inverse() \
            .as_right_combined(t_world_marker)
        rvec, tvec = t_camera_marker.as_rvec_tvec()

        # project the points using t_camera_marker
        corners_f_image, _ = cv2.projectPoints(corners_f_marker,
                                               rvec, tvec,
                                               self.sc.camera_matrix, self.sc.dist_coeffs)
        return corners_f_image.reshape(8, 1)

    def generate_corners_f_image(self, camera, marker):
        assert camera.camera_not_marker
        assert not marker.camera_not_marker

        return self._project_points(camera, marker)

    def _generate_sigma_points(self, com, corners_f_image):
        # TODO figure out real sigma points.
        com_cvec = com.t_world_xxx.as_cvec()

        gamma = self.gamma0 if com.simulated_not_derived else self.gamma1

        corners_f_images = np.zeros([8, 17] if com.simulated_not_derived else [8, 29])
        t_world_xxx_cvecs = np.zeros([6, 17] if com.simulated_not_derived else [6, 29])

        var = self.sc.std_corners_f_image
        f = corners_f_image * np.ones([8, 17])
        f[:, 1:9] += np.eye(8) * var * gamma
        f[:, 9:17] -= np.eye(8) * var * gamma

        corners_f_images[:, :17] = f
        t_world_xxx_cvecs[:, :17] = com_cvec * np.ones([6, 17])

        if not com.simulated_not_derived:
            s = sqrtm(com.cov)
            # var = np.sqrt(np.diag(com.cov).reshape(6, 1))
            # var = np.diag(com.cov).reshape(6, 1)
            f = com_cvec * np.ones([6, 12])
            f[:, :6] += s * gamma
            f[:, 6:12] -= s * gamma

            corners_f_images[:, 17:29] = corners_f_image * np.ones([8, 12])
            t_world_xxx_cvecs[:, 17:29] = f

        return SolvePnpInputs(com.camera_not_marker, corners_f_images, t_world_xxx_cvecs)

    def generate_solve_pnp_inputs(self, camera_not_marker, camera, marker):
        assert camera.camera_not_marker
        assert not marker.camera_not_marker

        com = camera if camera_not_marker else marker

        # Get corners using the poses of the camera and marker
        corners_f_image = self._project_points(camera, marker)

        if self.sc.use_sigma_points:
            return self._generate_sigma_points(com, corners_f_image)

        # Make many of these corner bundles
        corners_f_images = corners_f_image + \
                           np.random.normal(0.,
                                            self.sc.std_corners_f_image ** 2,
                                            size=[8, self.sc.num_generated_samples])

        # Now get many cvecs
        if com.simulated_not_derived:
            t_world_xxx_cvec = com.t_world_xxx.as_cvec()
            t_world_xxx_cvecs = np.tile(t_world_xxx_cvec, (1, self.sc.num_generated_samples))
        elif self.sc.use_dist_params:
            t_world_xxx_cvecs = np.random.multivariate_normal(com.mu[:, 0], com.cov, self.sc.num_generated_samples).T
        else:
            assert com.samples is not None
            t_world_xxx_cvecs = com.samples

        return SolvePnpInputs(camera_not_marker, corners_f_images, t_world_xxx_cvecs)

    def solve_pnp(self, inputs):
        t_world_xxx_cvecs = np.zeros(inputs.t_world_xxx_cvecs.shape)

        # get corners in the marker frame
        corners_f_marker = self._get_corners_f_marker().T

        for i in range(t_world_xxx_cvecs.shape[1]):
            # Given the location of the corners in the image, find the pose of the marker in the camera frame.
            ret, rvecs, tvecs = cv2.solvePnP(corners_f_marker, inputs.corners_f_images[:, i].reshape(4, 2),
                                             self.sc.camera_matrix, self.sc.dist_coeffs)

            t_camera_marker = tf.Transformation.from_rodrigues(rvecs[:, 0], translation=tvecs[:, 0])
            input_t_world_xxx = tf.Transformation.from_cvec(inputs.t_world_xxx_cvecs[:, i])
            t_camera_marker_factor = t_camera_marker
            if not inputs.camera_not_marker:
                t_camera_marker_factor = t_camera_marker_factor.as_inverse()
            output_t_world_xxx = input_t_world_xxx.as_right_combined(t_camera_marker_factor)

            t_world_xxx_cvecs[:, i] = output_t_world_xxx.as_cvec().T

        mu = np.mean(t_world_xxx_cvecs, axis=1).reshape(6, 1)
        cov = np.cov(t_world_xxx_cvecs)

        return CameraOrMarker.from_mu(not inputs.camera_not_marker, mu, cov, t_world_xxx_cvecs)
