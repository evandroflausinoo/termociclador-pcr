from dataclasses import dataclass
import random

@dataclass
class ThermalParams:
    Tamb: float = 25.0
    alpha: float = 0.02
    beta: float = 3.0
    dt: float = 1.0
    noise_std: float = 0.05

class ThermalModel:
    def __init__(self, params: ThermalParams, t0: float = 25.0):
        self.p = params
        self.t = t0

    def reset(self, t0: float = 25.0) -> float:
        self.t = t0
        return self.t

    def step(self, u: int) -> float:
        u = float(u)

        loss = (self.t - self.p.Tamb) * self.p.alpha
        gain = self.p.beta * u
        noise = random.gauss(0.0, self.p.noise_std)

        t_next = self.t + self.p.dt * (gain - loss) + noise
        self.t = t_next
        return self.t
