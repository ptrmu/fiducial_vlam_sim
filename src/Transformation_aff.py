import cv2
import numpy as np
import transformations as tf_aff


class Transformation:
    def __init__(self, matrix_aff=None, transformation=None):
        if matrix_aff is not None:
            self.matrix_aff = np.copy(matrix_aff)
        elif transformation is not None:
            self.matrix_aff = np.copy(transformation.matrix_aff)
        else:
            self.matrix.aff = tf_aff.identity_matrix()

    @staticmethod
    def from_rodrigues(rvec, translation=None):
        m = tf_aff.identity_matrix() if translation is None else tf_aff.translation_matrix(translation)
        rmat, _ = cv2.Rodrigues(rvec)
        m[:3, :3] = rmat
        return Transformation(matrix_aff=m)

    # following ROS standards:
    # r, p, y are rotations around the external x, y, z axes and are
    # applied in the r, p, y order.
    # r, p, y angles use the right handed rotation convention.
    # If you look down the rotation axis, then positive rotations
    # are counter clockwise around the axis.
    #
    # To create a transform t_d_s which maps vectors from s frame to d frame:
    # Start with coordinate axes in the d frame. Rotate those axes r radians
    # about d's x axis and then p radians about d's y axis and then y radians
    # about d's x axis to make the axes line up with the s frame. Figure out
    # when r, p, y need to be and use them in the from_rpy() method.
    @staticmethod
    def from_rpy(r, p, y, translation=None):
        t = tf_aff.identity_matrix() if translation is None else tf_aff.translation_matrix(translation)
        r = tf_aff.euler_matrix(r, p, y)
        m = t @ r
        return Transformation(matrix_aff=m)

    def as_rodrigues(self):
        rmat = self.matrix_aff[:3, :3]
        rvec, _ = cv2.Rodrigues(rmat)
        return rvec

    def as_rvec_tvec(self):
        return self.as_rodrigues(), self.matrix_aff[:3, 3]

    def as_rpy(self):
        r, p, y = tf_aff.euler_from_matrix(self.matrix_aff)
        return r, p, y

    def as_translation(self):
        return self.matrix_aff[:3, 3]

    def transform_vectors(self, vectors_local):
        vl = np.vstack((vectors_local, np.ones((1, vectors_local.shape[1]))))
        vt = np.dot(self.matrix_aff, vl)
        return vt[:3, :]

    def as_inverse(self):
        return Transformation(matrix_aff=np.linalg.inv(self.matrix_aff))

    def as_right_combined(self, right_side):
        return Transformation(matrix_aff=self.matrix_aff @ right_side.matrix_aff)
