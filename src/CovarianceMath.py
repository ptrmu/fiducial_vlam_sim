import numpy as np
import quaternion  # install numpy-quaternion
import transformations
import Transformation


# elements are stored as 1D arrays or rows in 2D arrays
# 3 -> translation p (xyz)
# 6 -> translation covariance pc (xyz)x(xyz)
# 9 -> translation with covariance pwc
# 3 -> rotation r (rpy)
# 6 -> rotation covariance rc (rpy)x(rpy)
# 9 -> rotation with covariance rwc
# 6 -> transform t (xyzrpy)
# 21 -> covariance vector tc (xyzrpy)x(xyzrpy)
# 	    L of Cholesky factorization
# 27 -> transform with covariance vector twc

class MonteCarloSimulation:
    def __init__(self, number_of_samples):
        self.number_of_samples = number_of_samples

    def do_simulation(self, u1, cov1, func):
        # Evaluate many times from distribution (u1, cov1), return (u, cov)
        # func -> f(u1)
        y = func(u1)
        return y, None

    def do_simulation_2(self, u1, cov1, u2, cov2, func):
        # Evaluate many times from distribution (u1, cov1) and (u2, cov2). return (u, cov)
        # func -> f(u1, u2)
        y = func(u1, u2)
        return y, None


class TransformationCalculator:
    def __init__(self, simulation):
        self.sim = simulation

    def _unpack(self, twc):
        i = 6;
        u = twc[0:i]
        if len(twc) <= 6:
            return u, None

        cov = np.zeros([6, 6])
        for r in range(6):
            for c in range(r + 1):
                cov[r, c] = twc[i]
                if r != c:
                    cov[c, r] = twc[i]
                i = i + 1
        return u, cov

    def _pack(self, u, cov):
        return u

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

    def _multi_transform(self, u1, u2):
        n = u1.shape[0]

        # u1 and u2 are (n, 6) arrays
        xyz1 = u1[:, 0:3]
        rpy1 = u1[:, 3:6]
        xyz2 = u2[:, 0:3]
        rpy2 = u2[:, 3:6]

        # r1, r2, r and (n, 3, 3) arrays
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

    def _multi_invert(self, u1):
        n = u1.shape[0]

        # u1 is (n, 6) array
        xyz1 = u1[:, 0:3]
        rpy1 = u1[:, 3:6]

        # r1, r and (n, 3, 3) arrays
        r1 = self._mats_from_rpys(rpy1)

        # r = inverse(r1) => transpose(r1)
        r = np.transpose(r1, (0, 2, 1))
        r_t = np.array([np.linalg.inv(r1[i, :, :]) for i in range(r1.shape[0])])
        assert (np.allclose(r, r_t))

        # xyz = r * -xyz1
        xyz = np.sum(r * (-xyz1.reshape(n, 1, 3)), -1)
        xyz_t = np.array([r[i, :, :] @ -xyz1[i, :] for i in range(r1.shape[0])])
        assert (np.allclose(xyz, xyz_t))

        rpy = self._rpys_from_mats(r)
        u = np.hstack((xyz, rpy))
        return u

    def transform_transformation(self, t1, t2):
        assert (len(t1) == 6 or len(t1) == 27)
        assert (len(t2) == 6 or len(t2) == 27)

        u1, cov1 = self._unpack(t1)
        u2, cov2 = self._unpack(t2)

        if cov1 is not None and cov2 is not None:
            u, cov = self.sim.do_simulation_2(u1, cov1, u2, cov2,
                                              lambda a1, a2: self._multi_transform(a1, a2))

        elif cov1 is not None:
            u, cov = self.sim.do_simulation(u1, cov1,
                                            lambda a1: self._multi_transform(a1, u2))

        elif cov2 is not None:
            u, cov = self.sim.do_simulation(u2, cov2,
                                            lambda a1: self._multi_transform(u1, a1))

        else:
            cov = None
            u = self._multi_transform(np.array([u1]), np.array([u2]))

        t = self._pack(u, cov)
        return t

    def invert_transformation(self, t1):
        assert (len(t1) == 6 or len(t1) == 27)

        u1, cov1 = self._unpack(t1)

        if cov1 is not None:
            u, cov = self.sim.do_simulation(u1, cov1,
                                            lambda a1: self._multi_invert(a1))

        else:
            cov = None
            u = self._multi_invert(np.array([u1]))

        t = self._pack(u, cov)
        return t


# class MultiTransformMath:
#     def __init__(self):
#         pass
#
#
# class TransformOperations:
#     def __init__(self):
#         pass
#
#     def matrix_from_rpy(self, rpy):
#         pass
#
#     def rpy_from_matrix(self, mat):
#         pass
#
#
# class TransformMath:
#     def __init__(self, transform_operations):
#         self.to = transform_operations
#
#     def translate_translation(self, translation, translation_1):
#         return self.to.add(translation, translation_1)
#
#     def translate_transformation(self, translation, transform_1):
#         return self.to.new_transformation(
#             self.to.get_transformation_rotation(transform_1),
#             self.translate_translation(translation, self.to.get_transformation_translation(transform_1)))
#
#     def rotate_rotation(self, rotation, rotation_1):
#         return self.to.rpy_from_matrix(self.to.dot(
#             self.to.matrix_from_rpy(rotation),
#             self.to.matrix_from_rpy(rotation_1)))
#
#     def rotate_transform(self, rotation, transform_1):
#         pass
#
#     def transform_translation(self, transformation, translation_1):
#         pass
#
#     def transform_rotation(self, transformation, rotation_1):
#         pass
#
#     def transform_transformation(self, transformation, transformation_1):
#         pass
#
#     def invert_translation(self, translation):
#         pass
#
#     def invert_rotation(selfself, rotation):
#         pass
#
#     def invert_transformation(self, transformation):
#         pass
#

class CovarianceMath:
    def __init__(self, sim):
        self.sim = sim

    @staticmethod
    def assert_arg_size(var1, var2, size):
        assert (len(var1.shape) != 1 or var1.shape[0] == size)
        assert (len(var1.shape) != 2 or var1.shape[1] == size)
        assert (len(var2.shape) != 1 or var2.shape[0] == size)
        assert (len(var2.shape) != 2 or var2.shape[1] == size)

    @staticmethod
    def new_translation(x, y, z):
        return np.array([x, y, z])

    @staticmethod
    def new_rotation(r, p, y):
        return np.array([r, p, y])

    def new_transformation(self, rotation,
                           translation):
        CovarianceMath.assert_arg_size(rotation, translation, 3)
        return np.hstack((rotation, translation))

    def translate_position(self, translate, translation):
        print(translate.shape, translation.shape)
        sum = translate + translation
        t = 4

    def rotation_matrices_from_rpys(self, rpys):
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

    def rpys_from_rotation_matrices(self, mats):
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

    def transform_transformation(self, transform, transformation):
        pass


if __name__ == "__main__":

    def new_Transformation(t):
        t1 = np.zeros(6)
        t1[0:3] = t[3:6]
        t1[3:6] = t[0:3]
        return Transformation.Transformation.from_cvec(t1)


    def from_Transformation(trans):
        t1 = trans.as_cvec()
        t = np.zeros(6)
        t[0:3] = t1.reshape(6)[3:6]
        t[3:6] = t1.reshape(6)[0:3]
        return t


    def one_gen(gen_args):
        start, delta, number = gen_args
        if number < 2:
            interval = delta
        else:
            interval = delta / (number - 1)
        for i in range(number):
            yield start + interval * i


    def tri_gen(gen_args):
        for a in one_gen(gen_args):
            for b in one_gen(gen_args):
                for c in one_gen(gen_args):
                    yield a, b, c


    def transform_gen(gen_args_ang, gen_args_lin):
        for rpy1 in tri_gen(gen_args_ang):
            for xyz1 in tri_gen(gen_args_lin):
                yield np.hstack((xyz1, rpy1))


    def transform_2_gen(gen_args_ang, gen_args_lin):
        for t1 in transform_gen(gen_args_ang, gen_args_lin):
            for t2 in transform_gen(gen_args_ang, gen_args_lin):
                yield (t1, t2)


    def transformsclose(t, tt):
        if t[0, 3] >= np.pi:
            t[0, 3] -= np.pi * 2
        elif t[0, 3] < -np.pi:
            t[0, 3] += np.pi * 2
        if t[0, 4] >= np.pi:
            t[0, 4] -= np.pi * 2
        elif t[0, 4] < -np.pi:
            t[0, 4] += np.pi * 2
        if t[0, 5] >= np.pi:
            t[0, 5] -= np.pi * 2
        elif t[0, 5] > np.pi:
            t[0, 5] = np.pi * 2
        if tt[3] >= np.pi:
            tt[3] -= np.pi * 2
        elif tt[3] < -np.pi:
            tt[3] += np.pi * 2
        if tt[4] >= np.pi:
            tt[4] -= np.pi * 2
        elif tt[4] < -np.pi:
            tt[4] += np.pi * 2
        if tt[5] >= np.pi:
            tt[5] -= np.pi * 2
        elif tt[5] < -np.pi:
            tt[5] += np.pi * 2
        return np.allclose(t, tt)


    def test_one_element_transform(tc, t1, t2):
        t = tc.transform_transformation(t1, t2)
        tt1 = new_Transformation(t1)
        tt2 = new_Transformation(t2)
        tt = from_Transformation(tt1.as_right_combined(tt2))
        if not transformsclose(t, tt):
            assert np.allclose(t, tt)


    def test_inverse(tc, t1):
        t = tc.invert_transformation(t1)
        tt1 = new_Transformation(t1)
        tt = from_Transformation(tt1.as_inverse())
        if not transformsclose(t, tt):
            assert np.allclose(t, tt)


    def test():
        mcs = MonteCarloSimulation(1)
        tc = TransformationCalculator(mcs)

        test_one_element_transform(tc,
                                   (-1., -1., -1., 0., -np.pi / 2, 0.),
                                   (-1., -1., -1., np.pi, -np.pi / 2, 0.))

        for t1, t2 in transform_2_gen((-np.pi / 2, np.pi, 3), (-1., 2., 3)):
            test_one_element_transform(tc, t1, t2)

        # for t1 in transform_gen((-np.pi / 2, np.pi, 3), (-1., 2., 3)):
        #     test_inverse(tc, t1)

        # cm = CovarianceMath(None)
        #
        # r1 = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
        # r2 = np.array(r1) + 3
        # r1 = np.array([r1])
        # r2 = np.array([r2])
        # r1a = np.transpose(r1, (0, 2, 1))
        # r1b = r1a.reshape(1, 3, 3, 1)
        # r2b = r2.reshape(1, 3, 1, 3)
        # r2c = r1b * r2b
        # r = np.sum(r2c, -3)
        # t1 = r1[0, :, :] @ r2[0, :, :]
        #
        # v = [3, 4]
        # v = np.array([v])
        # vb = v.reshape(1, 1, 2)
        # vc = r1 * vb
        # vd = np.sum(vc, -1)
        # t2 = np.dot(r1[0, :, :], v[0, :])
        #
        #
        # one_a_pos = np.array([1, 2, 3])
        # one_b_pos = np.array([9, 10, 11])
        #
        # two__pos = np.array([[10, 11, 12], [23, 24, 25]])
        #
        # print(cm.new_transformation(one_a_pos, one_b_pos))
        # print(cm.translate_position(two__pos, one_b_pos))
        #
        # increment = np.pi / 8
        #
        # rpys_1 = np.array([[1., 1., 2.], [2.356194490192345, 1.5707963267948966, -2.748893571891069]])
        # rots_1 = cm.rotation_matrices_from_rpys(rpys_1)
        # rots_2 = transformations.euler_matrix(rpys_1[0, 0], rpys_1[0, 1], rpys_1[0, 2])
        # rpys_3 = cm.rpys_from_rotation_matrices(rots_1)
        #
        # rot_3 = transformations.euler_matrix(rpys_1[1, 0], rpys_1[1, 1], rpys_1[1, 2])
        # rpy_4 = transformations.euler_from_matrix(rot_3)
        #
        # rpys = []
        # for r in np.arange(-np.pi, np.pi, increment):
        #     for p in np.arange(-np.pi, np.pi, increment):
        #         for y in np.arange(-np.pi, np.pi, increment):
        #             rpys.append([r, p, y])
        #
        # rpys = np.array(rpys)
        # Rs = cm.rotation_matrices_from_rpys(rpys)
        # rpys_out = cm.rpys_from_rotation_matrices(Rs)
        #
        # for i in range(Rs.shape[0]):
        #     rpy = rpys[i, :]
        #     R1 = Rs[i, :, :]
        #     rpy_out = rpys_out[i, :]
        #
        #     R2 = transformations.euler_matrix(rpy[0], rpy[1], rpy[2])
        #     R2 = R2[0:3, 0:3]
        #     rpy_test = transformations.euler_from_matrix(R2)
        #
        #     if not np.allclose(R1, R2):
        #         t = 2;
        #     if not np.allclose(rpy_out, rpy_test):
        #         t = 4


    test()
