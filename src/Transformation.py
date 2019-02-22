import cv2
import numpy as np
import quaternion  # install numpy-quaternion


class Transformation:
    def __init__(self, rotation=None, translation=None, transformation=None):
        if transformation is not None:
            rotation = transformation.rotate if rotation is None else rotation
            translation = transformation.translate if translation is None else translation
        self.rotation = quaternion.one if rotation is None else np.copy(rotation)
        self.translation = (np.zeros(3) if translation is None else np.copy(translation)).reshape(3, 1)

    @staticmethod
    def from_rodrigues(rvec, translation=None):
        rmat, _ = cv2.Rodrigues(rvec)
        q = quaternion.from_rotation_matrix(rmat)
        return Transformation(rotation=q, translation=translation)

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
        qr = quaternion.from_rotation_vector([r, 0., 0.])
        qp = quaternion.from_rotation_vector([0., p, 0.])
        qy = quaternion.from_rotation_vector([0., 0., y])
        q = qy * qp * qr
        return Transformation(rotation=q, translation=translation)

    @staticmethod
    def from_cvec(cvec):
        return Transformation.from_rpy(cvec[0], cvec[1], cvec[2], translation=cvec[3:6])

    def as_rodrigues(self):
        rmat = quaternion.as_rotation_matrix(self.rotation)
        rvec, _ = cv2.Rodrigues(rmat)
        return rvec

    def as_rvec_tvec(self):
        return self.as_rodrigues(), self.translation

    def as_rpy(self):
        rmat = quaternion.as_rotation_matrix(self.rotation)
        sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
        if sy > 1e-6:
            roll = np.arctan2(rmat[2, 1], rmat[2, 2])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
            pitch = np.arctan2(-rmat[2, 0], sy)
            yaw = 0

        return roll, pitch, yaw

    def as_translation(self):
        return self.translation

    def as_cvec(self):
        rpy = self.as_rpy()
        xyz = self.as_translation()
        return np.array([rpy[0], rpy[1], rpy[2], xyz[0], xyz[1], xyz[2]]).reshape(6, 1)

    def transform_vectors(self, vectors_local):
        points_rotated = quaternion.rotate_vectors(self.rotation, vectors_local, axis=0)
        points_translated = points_rotated + self.translation
        return points_translated

    def as_inverse(self):
        # There is probably a faster way to do this, but I can't find it now.
        # q_inverse = quaternion.from_rotation_matrix(np.linalg.inv(quaternion.as_rotation_matrix(self.rotation)))
        q_inverse = self.rotation.conj()
        t_inverse = quaternion.rotate_vectors(q_inverse, -self.translation, axis=0)
        return Transformation(rotation=q_inverse, translation=t_inverse)

    def as_right_combined(self, right_side):
        q_combined = self.rotation * right_side.rotation
        t_combined = quaternion.rotate_vectors(self.rotation, right_side.translation, axis=0) + self.translation
        return Transformation(rotation=q_combined, translation=t_combined)
