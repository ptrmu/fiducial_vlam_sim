import numpy as np


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
