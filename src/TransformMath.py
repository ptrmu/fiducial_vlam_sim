import numpy as np


# elements are stored as 1D arrays or rows in 2D arrays
# 3 -> translation p (xyz)
# 6 -> translation covariance pc (xyz)x(xyz)
# 9 -> translation with covariance pwc
# 6 -> transform t (xyzrpy)
# 21 -> covariance vector tc (xyzrpy)x(xyzrpy)
# 	    L of Cholesky factorization
# 27 -> transform with covariance vector twc

class TransformMath:
    def __init__(self, sim):
        self.sim = sim

    def transformation_from_xyz_rpy(self, xyz, rpy):
        return np.hstack((xyz, rpy))

    def unpack(self, with_covariance):
        # The length of the input determines what it represents and
        # how to unpack it. The input is a 1D vector.
        n = len(with_covariance)
        # a position or transform with no covariance
        if n == 3 or n == 6:
            return with_covariance, None
        # a position or transform with covariance
        if n == 9:
            nt = 3
        elif n == 27:
            nt = 6
        else:
            assert False

        u = with_covariance[0:nt]

        # The diagonal values are stored first
        i = 2 * nt
        cov = np.diag(with_covariance[nt:i])
        cov[1, 0] = with_covariance[i]
        cov[0, 1] = with_covariance[i]
        cov[2, 0:2] = with_covariance[(i + 1):(i + 3)]
        cov[0:2, 2] = with_covariance[(i + 1):(i + 3)]
        if nt == 6:
            cov[3, 0:3] = with_covariance[15:18]
            cov[0:3, 3] = with_covariance[15:18]
            cov[4, 0:4] = with_covariance[18:22]
            cov[0:4, 4] = with_covariance[18:22]
            cov[5, 0:5] = with_covariance[22:27]
            cov[0:5, 5] = with_covariance[22:27]

        return u, cov

    def pack(self, u, cov):
        if cov is None:
            return u

        n = u.shape[0]
        assert len(cov.shape) == 2 and cov.shape[0] == n and cov.shape[1] == n

        u_new = np.zeros(9 if n == 3 else 27)

        # Add the mean values
        u_new[0:n] = u

        # Append the cov with the diagonals first
        i = 2 * n
        u_new[n:i] = np.diag(cov)
        u_new[i] = cov[1, 0]
        u_new[(i + 1):(i + 3)] = cov[2, 0:2]
        if n == 6:
            u_new[15:18] = cov[3, 0:3]
            u_new[18:22] = cov[4, 0:4]
            u_new[22:27] = cov[5, 0:5]

        return u_new

    def _mats_from_rpys(self, rpys):
        n = rpys.shape[0]
        si, sj, sk = np.sin(rpys[:, 0]), np.sin(rpys[:, 1]), np.sin(rpys[:, 2])
        ci, cj, ck = np.cos(rpys[:, 0]), np.cos(rpys[:, 1]), np.cos(rpys[:, 2])
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        m = np.ones((n, 3, 3))
        m[:, 0, 0] = cj * ck
        m[:, 0, 1] = sj * sc - cs
        m[:, 0, 2] = sj * cc + ss
        m[:, 1, 0] = cj * sk
        m[:, 1, 1] = sj * ss + cc
        m[:, 1, 2] = sj * cs - sc
        m[:, 2, 0] = -sj
        m[:, 2, 1] = cj * si
        m[:, 2, 2] = cj * ci
        return m

    def _rpys_from_mats(self, mats):
        n = mats.shape[0]
        rs = np.zeros([n])
        ps = np.zeros([n])
        ys = np.zeros([n])

        sy = np.sqrt(mats[:, 0, 0] * mats[:, 0, 0] + mats[:, 1, 0] * mats[:, 1, 0])

        sy_ind = sy >= 1e-6
        rs[sy_ind] = np.arctan2(mats[sy_ind, 2, 1], mats[sy_ind, 2, 2])
        ps[sy_ind] = np.arctan2(-mats[sy_ind, 2, 0], sy[sy_ind])
        ys[sy_ind] = np.arctan2(mats[sy_ind, 1, 0], mats[sy_ind, 0, 0])

        sy_ind = sy < 1e-6
        rs[sy_ind] = np.arctan2(-mats[sy_ind, 1, 2], mats[sy_ind, 1, 1])
        ps[sy_ind] = np.arctan2(-mats[sy_ind, 2, 0], sy[sy_ind])
        ys[sy_ind] = 0

        return np.vstack((rs, ps, ys)).transpose()

    def mat_from_rpy(self, rpy):
        return self._mats_from_rpys(np.array([rpy]))[0]

    def rpy_from_mat(self, mat):
        return self._rpys_from_mats(np.array([mat]))[0]

    def _multi_transform_transformation(self, u1, u2):
        n = u1.shape[0]

        # u1 and u2 are (n, 6) arrays
        xyz1 = u1[:, 0:3]
        rpy1 = u1[:, 3:6]
        xyz2 = u2[:, 0:3]
        rpy2 = u2[:, 3:6]

        # r1, r2, r are (n, 3, 3) arrays
        r1 = self._mats_from_rpys(rpy1)
        r2 = self._mats_from_rpys(rpy2)

        # r = r1 * r2
        r = np.sum(np.transpose(r1, (0, 2, 1)).reshape(n, 3, 3, 1) * r2.reshape(n, 3, 1, 3), -3)
        # r_t = np.array([r1[i, :, :] @ r2[i, :, :] for i in range(r1.shape[0])])
        # assert (np.allclose(r, r_t))

        # xyz = r1 * xyz2 + xyz1
        xyz = np.sum(r1 * xyz2.reshape(n, 1, 3), -1) + xyz1
        # xyz_t = np.array([r1[i, :, :] @ xyz2[i, :] + xyz1 for i in range(r1.shape[0])])
        # assert (np.allclose(xyz, xyz_t))

        rpy = self._rpys_from_mats(r)
        u = np.hstack((xyz, rpy))
        return u

    def _multi_transform_position(self, u1, u2):
        n = u1.shape[0]

        # u1 and u2 are (n, 6) arrays
        xyz1 = u1[:, 0:3]
        rpy1 = u1[:, 3:6]
        xyz2 = u2

        # r1 is (n, 3, 3) array
        r1 = self._mats_from_rpys(rpy1)

        # xyz = r1 * xyz2
        xyz = np.sum(r1 * xyz2.reshape(n, 1, 3), -1) + xyz1
        # xyz_t = np.array([r1[i, :, :] @ xyz2[i, :] + xyz1 for i in range(r1.shape[0])])
        # assert (np.allclose(xyz, xyz_t))

        return xyz

    def _multi_invert(self, u1):
        n = u1.shape[0]

        # u1 is (n, 6) array
        xyz1 = u1[:, 0:3]
        rpy1 = u1[:, 3:6]

        # r1, r are (n, 3, 3) arrays
        r1 = self._mats_from_rpys(rpy1)

        # r = inverse(r1) => transpose(r1)
        r = np.transpose(r1, (0, 2, 1))
        # r_t = np.array([np.linalg.inv(r1[i, :, :]) for i in range(r1.shape[0])])
        # assert (np.allclose(r, r_t))

        # xyz = r * -xyz1
        xyz = np.sum(r * (-xyz1.reshape(n, 1, 3)), -1)
        # xyz_t = np.array([r[i, :, :] @ -xyz1[i, :] for i in range(r1.shape[0])])
        # assert (np.allclose(xyz, xyz_t))

        rpy = self._rpys_from_mats(r)
        u = np.hstack((xyz, rpy))
        return u

    def transform_transformation(self, t1, t2):
        assert (len(t1) == 6 or len(t1) == 27)
        assert (len(t2) == 6 or len(t2) == 27)

        u1, cov1 = self.unpack(t1)
        u2, cov2 = self.unpack(t2)

        if cov1 is not None and cov2 is not None:
            u, cov = self.sim.do_simulation_2(u1, cov1, u2, cov2,
                                              lambda a1, a2: self._multi_transform_transformation(a1, a2))

        elif cov1 is not None:
            u, cov = self.sim.do_simulation(u1, cov1,
                                            lambda a1: self._multi_transform_transformation(a1, u2))

        elif cov2 is not None:
            u, cov = self.sim.do_simulation(u2, cov2,
                                            lambda a1: self._multi_transform_transformation(u1, a1))

        else:
            cov = None
            u = self._multi_transform_transformation(np.array([u1]), np.array([u2]))[0]

        t = self.pack(u, cov)
        return t

    def transform_position(self, t1, x2):
        assert (len(t1) == 6 or len(t1) == 27)
        assert (len(x2) == 3 or len(x2) == 9)

        u1, cov1 = self.unpack(t1)
        u2, cov2 = self.unpack(x2)

        if cov1 is not None and cov2 is not None:
            u, cov = self.sim.do_simulation_2(u1, cov1, u2, cov2,
                                              lambda a1, a2: self._multi_transform_position(a1, a2))

        elif cov1 is not None:
            u, cov = self.sim.do_simulation(u1, cov1,
                                            lambda a1: self._multi_transform_position(a1, u2))

        elif cov2 is not None:
            u, cov = self.sim.do_simulation(u2, cov2,
                                            lambda a1: self._multi_transform_position(u1, a1))

        else:
            cov = None
            u = self._multi_transform_position(np.array([u1]), np.array([u2]))[0]

        t = self.pack(u, cov)
        return t

    def invert_transformation(self, t1):
        assert (len(t1) == 6 or len(t1) == 27)

        u1, cov1 = self.unpack(t1)

        if cov1 is not None:
            u, cov = self.sim.do_simulation(u1, cov1,
                                            lambda a1: self._multi_invert(a1))

        else:
            cov = None
            u = self._multi_invert(np.array([u1]))[0]

        t = self.pack(u, cov)
        return t
