
import numpy as np
import Transformation as tf
import transformations as tf_aff

r = np.pi/4.
p = 0.
y = 0.
r = 0.
p = np.pi/4.
y = 0.
r = 0.
p = 0.
y = np.pi/4.
r = np.pi/2.
p = np.pi/2.
y = 0.
r = -np.pi
p = -np.pi
y = -np.pi
r = np.pi/4.
p = np.pi/4.
y = 0.

t = 0.
# t = 1.

t_rpy = tf.Transformation.from_rpy(r, p, y, translation=[t, t, t])
t_rpy_inverse = t_rpy.as_inverse()
t2_rpy = tf.Transformation.from_rpy(-np.pi/2., 0., np.pi/4., translation=[0., 0., 3.])
tc_rpy = t2_rpy.as_right_combined(t_rpy)

tr_euler_mat = tf_aff.translation_matrix([t, t, t]) @ tf_aff.euler_matrix(r, p, y)
tr_euler_mat_inverse = np.linalg.inv(tr_euler_mat)
tr2_euler_mat = tf_aff.translation_matrix([0., 0., 3.]) @ tf_aff.euler_matrix(-np.pi / 2., 0., np.pi / 4.)
trc_euler_mat = tr2_euler_mat @ tr_euler_mat

# vs = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).T
vs = np.array([[-1., 2., 0.], [1., 2., 0.], [1., -2., 0.], [-1., -2., 0.]]).T
vs1 = np.vstack((vs, np.ones((1, vs.shape[1]))))

tvs = t_rpy.transform_vectors(vs)
vsi = t_rpy_inverse.transform_vectors(tvs)
tvsc = tc_rpy.transform_vectors(vs)

tvs1 = tr_euler_mat @ vs1
vs1i = tr_euler_mat_inverse @ tvs1
tvs1c = trc_euler_mat @ vs1

rt, pt, yt = t_rpy.as_rpy()

increment = np.pi/8
for r in np.arange(-np.pi, np.pi, increment):
    for p in np.arange(-np.pi, np.pi, increment):
        for y in np.arange(-np.pi, np.pi, increment):
            for t in np.arange(-1., 2., 1.):
                # compare our transform with Gohlke's
                t_rpy = tf.Transformation.from_rpy(r, p, y, translation=[t, t, t])
                tr_euler_mat = tf_aff.translation_matrix([t, t, t]) @ tf_aff.euler_matrix(r, p, y)
                tvs = t_rpy.transform_vectors(vs)
                tvs1 = tr_euler_mat @ vs1
                assert(np.allclose(tvs, tvs1[:3, :]))

                # test out the inverse transformation
                t_rpy_inverse = t_rpy.as_inverse()
                vsi = t_rpy_inverse.transform_vectors(tvs)
                assert(np.allclose(vs, vsi))

                # Get the rpy from the transform and ensure they create the same new transform
                rt, pt, yt = t_rpy.as_rpy()
                tt_rpy = tf.Transformation.from_rpy(rt, pt, yt, translation=t_rpy.as_translation())
                ttvs = tt_rpy.transform_vectors(vs)
                assert(np.allclose(ttvs, tvs))

pass
