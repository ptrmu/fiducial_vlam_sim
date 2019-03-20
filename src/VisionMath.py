import numpy as np
import cv2


class VisionMath:
    def __init__(self, tm, marker_len, camera_matrix, dist_coeffs):
        self.tm = tm
        self.marker_len = marker_len
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def transformation_from_rvec_tvec(self, rvec, tvec):
        rmat, _ = cv2.Rodrigues(rvec)
        rpy = self.tm.rpy_from_mat(rmat)
        return self.tm.transformation_from_xyz_rpy(tvec, rpy)

    def transformation_as_rvec_tvec(self, t1):
        rmat = self.tm.mat_from_rpy(t1[3:6])
        rvec, _ = cv2.Rodrigues(rmat)
        return rvec, t1[0:3]

    def _get_corners_f_marker(self):
        m2 = self.marker_len / 2.0
        marker_corners_f_marker = np.array([[-m2, m2, 0.], [m2, m2, 0.], [m2, -m2, 0.], [-m2, -m2, 0.]]).T
        return marker_corners_f_marker

    def project_points(self, t_camera_marker):
        # get corners in the marker frame
        corners_f_marker = self._get_corners_f_marker().T

        # calculate rvec, tvec from t_camera_marker
        rvec, tvec = self.transformation_as_rvec_tvec(t_camera_marker)

        # project the points using t_camera_marker
        corners_f_image, _ = cv2.projectPoints(corners_f_marker,
                                               rvec, tvec,
                                               self.camera_matrix, self.dist_coeffs)
        return corners_f_image.reshape(8)

    def _multi_solve_pnp(self, corners_f_images):
        # corners_f_images is an nx8 array
        # t_camera_markers is an nx6 array
        n = corners_f_images.shape[0]
        t_camera_markers = np.zeros((n, 6))

        # get corners in the marker frame
        corners_f_marker = self._get_corners_f_marker().T

        for i in range(n):
            ret, rvec, tvec = cv2.solvePnP(corners_f_marker, corners_f_images[i, :].reshape(4, 2),
                                           self.camera_matrix, self.dist_coeffs)
            t_camera_markers[i, :] = self.transformation_from_rvec_tvec(rvec[:, 0], tvec[:, 0])

        return t_camera_markers

    def solve_pnp(self, corners_f_image, std_corners_f_image):
        assert (len(corners_f_image) == 8)

        if std_corners_f_image is not None:
            cov1 = np.diag(np.ones(8) * std_corners_f_image ** 2)
            u, cov = self.tm.do_simulation(corners_f_image, cov1,
                                            lambda a1: self._multi_solve_pnp(a1))

        else:
            cov = None
            u = self._multi_solve_pnp(np.array([corners_f_image]))

        return self.tm.pack(u, cov)
