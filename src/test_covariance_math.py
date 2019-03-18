from CovarianceMath import *
import Transformation


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


# def test():
#     mcs = MonteCarloSimulation(1)
#     # tc = TransformationCalculator(mcs)
#
#     tweak = 1.0e-4
#     gen_args_ang = (-np.pi + tweak, 2. * (np.pi - tweak), 5)
#     gen_args_lin = (-1., 2., 3)
#
#     # test_2t_cases = [(tc, t1, t2) for t1, t2 in transform_2_gen(gen_args_ang, gen_args_lin)]
#     # for test_case in test_2t_cases:
#     #     test_one_element_transform(test_case)
#
#     test_x_cases = [(tc, t1, x2)
#                     for t1 in transform_gen(gen_args_ang, gen_args_lin)
#                     for x2 in tri_gen(gen_args_lin)]
#     for test_case in test_x_cases:
#         test_one_element_transform_position(test_case)
#
#     # test_1t_cases = [(tc, t1) for t1 in transform_gen(gen_args_ang, gen_args_lin)]
#     # for test_case in test_1t_cases:
#     #     test_one_element_inverse(test_case)


import multiprocessing

cm = None


def pool_initializer():
    mcs = MonteCarloSimulation(1)
    global cm
    cm = CovarianceMath(mcs)


def pool_test_one_element_transform(args):
    t1, t2 = args
    t = cm.transform_transformation(t1, t2)
    tt1 = new_Transformation(t1)
    tt2 = new_Transformation(t2)
    tt = from_Transformation(tt1.as_right_combined(tt2))
    if not transformsclose(t, tt):
        assert transformsclose(t, tt)


def pool_test_one_element_transform_position(args):
    t1, x2 = args
    x = cm.transform_position(t1, x2)
    tt1 = new_Transformation(t1)
    xx = tt1.transform_vectors(np.array(x2).reshape(3, 1)).reshape(3)
    if not np.allclose(x, xx):
        assert np.allclose(x, xx)


def pool_test_one_element_inverse(args):
    t1 = args
    t = cm.invert_transformation(t1)
    tt1 = new_Transformation(t1)
    tt = from_Transformation(tt1.as_inverse())
    if not transformsclose(t, tt):
        assert transformsclose(t, tt)


if __name__ == '__main__':
    def test():
        pool = multiprocessing.Pool(6, initializer=pool_initializer, initargs=())

        tweak = 1.0e-4
        gen_args_ang = (-np.pi + tweak, 2. * (np.pi - tweak), 11)
        gen_args_lin = (-1., 2., 7)

        # test_2t_cases = [(t1, t2) for t1, t2 in transform_2_gen(gen_args_ang, gen_args_lin)]
        # res = pool.map(pool_test_one_element_transform, test_2t_cases)

        # test_x_cases = [(t1, x2)
        #                 for t1 in transform_gen(gen_args_ang, gen_args_lin)
        #                 for x2 in tri_gen(gen_args_lin)]
        # res = pool.map(pool_test_one_element_transform_position, test_x_cases)


        test_1t_cases = [(t1) for t1 in transform_gen(gen_args_ang, gen_args_lin)]
        res = pool.map(pool_test_one_element_inverse, test_1t_cases)

    test()
