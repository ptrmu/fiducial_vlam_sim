
import numpy as np
import Transformation as tf


class CameraOrMarker:
    def __init__(self, camera_not_marker, rpy=None, xyz=None, mu=None, cov=None, samples=None):
        self._camera_not_marker = camera_not_marker
        sim = True if rpy is not None else False
        self._simulated_not_derived = sim
        self._sim_rpy = rpy if sim else None
        self._sim_xyz = xyz if sim else None
        self._mu = np.zeros(6) if sim else mu
        self._cov = np.zeros([6, 6]) if sim else cov
        self._samples = None if sim else samples

    @staticmethod
    def from_rpy(camera_not_marker, rpy, xyz):
        return CameraOrMarker(camera_not_marker, rpy=rpy, xyz=xyz)

    @staticmethod
    def from_mu(camera_not_marker, mu, cov, samples):
        return CameraOrMarker(camera_not_marker, mu=mu, cov=cov, samples=samples)

    @staticmethod
    def camera_from_rpy(rpy, xyz):
        return CameraOrMarker(True, rpy=rpy, xyz=xyz)

    @staticmethod
    def camera_from_mu(mu, cov, samples):
        return CameraOrMarker(True, mu=mu, cov=cov, samples=samples)

    @staticmethod
    def marker_from_rpy(rpy, xyz):
        return CameraOrMarker(False, rpy=rpy, xyz=xyz)

    @staticmethod
    def marker_from_mu(mu, cov, samples):
        return CameraOrMarker(False, mu=mu, cov=cov, samples=samples)

    @property
    def t_world_xxx(self):
        return tf.Transformation.from_rpy(*self._sim_rpy,
                                          translation=np.array(self._sim_xyz)) if self._simulated_not_derived else \
            tf.Transformation.from_rpy(self._mu[0], self._mu[1], self._mu[2],
                                       translation=self._mu[3:6])

    @property
    def simulated_not_derived(self):
        return self._simulated_not_derived

    @property
    def camera_not_marker(self):
        return self._camera_not_marker

    @property
    def sim_rpy(self):
        assert self._simulated_not_derived
        return self._sim_rpy

    @property
    def sim_xyz(self):
        assert self._simulated_not_derived
        return self._sim_xyz

    @property
    def mu(self):
        assert not self._simulated_not_derived
        return self._mu

    @property
    def cov(self):
        assert not self._simulated_not_derived
        return self._cov

    @property
    def samples(self):
        assert not self._simulated_not_derived
        return self._samples

    def as_rpy_str(self):
        if self._simulated_not_derived:
            return "sim_rpy:({:.3f}, {:.3f}, {:.3f}) sim_xyz:({:.2f}, {:.2f}, {:.2f})".format(
                self.sim_rpy[0], self.sim_rpy[1], self.sim_rpy[2],
                self.sim_xyz[0], self.sim_xyz[1], self.sim_xyz[2])

        return "mu_rpy:({:.3f}, {:.3f}, {:.3f}) mu_xyz:({:.2f}, {:.2f}, {:.2f})".format(
            float(self.mu[0]), float(self.mu[1]), float(self.mu[2]),
            float(self.mu[3]), float(self.mu[4]), float(self.mu[5]))

    def combine(self, other):
        assert self.camera_not_marker == other.camera_not_marker
        r = self.cov @ np.linalg.inv(self.cov + other.cov)
        mu = self.mu + r @ (other.mu - self.mu)
        cov = self.cov - r @ self.cov
        return CameraOrMarker(self.camera_not_marker, mu=mu, cov=cov)
