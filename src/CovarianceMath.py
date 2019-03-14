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
        i = 6
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


class CovarianceMath:
    def __init__(self, sim):
        self.sim = sim


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
        pi2 = np.pi * 2.0
        if not (np.isclose(t[0, 3], tt[3])
                or np.isclose(t[0, 3] + pi2, tt[3])
                or np.isclose(t[0, 3], tt[3] + pi2)):
            return False
        if not (np.isclose(t[0, 4], tt[4])
                or np.isclose(t[0, 4] + pi2, tt[4])
                or np.isclose(t[0, 4], tt[4] + pi2)):
            return False
        if not (np.isclose(t[0, 5], tt[5])
                or np.isclose(t[0, 5] + pi2, tt[5])
                or np.isclose(t[0, 5], tt[5] + pi2)):
            return False
        return np.allclose(t[0, 0:3], tt[0:3])


    def test_one_element_transform(args):
        tc, t1, t2 = args
        t = tc.transform_transformation(t1, t2)
        tt1 = new_Transformation(t1)
        tt2 = new_Transformation(t2)
        tt = from_Transformation(tt1.as_right_combined(tt2))
        if not transformsclose(t, tt):
            assert transformsclose(t, tt)


    def test_one_element_inverse(args):
        tc, t1 = args
        t = tc.invert_transformation(t1)
        tt1 = new_Transformation(t1)
        tt = from_Transformation(tt1.as_inverse())
        if not transformsclose(t, tt):
            assert transformsclose(t, tt)


    def test():
        mcs = MonteCarloSimulation(1)
        tc = TransformationCalculator(mcs)

        tweak = 1.0e-4
        gen_args_ang = (-np.pi + tweak, 2. * (np.pi - tweak), 7)
        gen_args_lin = (-1., 2., 1)

        test_2t_cases = [(tc, t1, t2) for t1, t2 in transform_2_gen(gen_args_ang, gen_args_lin)]
        for test_case in test_2t_cases:
            test_one_element_transform(test_case)

        test_1t_cases = [(tc, t1) for t1 in transform_gen(gen_args_ang, gen_args_lin)]
        for test_case in test_1t_cases:
            test_one_element_inverse(test_case)


    test()
