import numpy as np
from dataclasses import dataclass


@dataclass
class ThermalParams:
    Tamb: float = 25.0       # Temperatura ambiente (°C)
    alpha: float = 0.0035    # Perda de calor para o ambiente (cúpula fechada)
    beta: float = 0.48       # Ganho térmico do bloco de alumínio
    dt: float = 1.0          # Passo de tempo (1 segundo por iteração)
    noise_std: float = 0.05  # Ruído leve do DS18B20


class ThermalModel:
    def __init__(self, params: ThermalParams, t0: float = 25.0):
        self.p = params
        self.t = t0
        self.history: list[float] = [t0]

    def reset(self, t0: float = 25.0) -> float:
        """Reinicia o modelo entre episódios de treino."""
        self.t = t0
        self.history = [t0]
        return self.t

    def step(self, u: float) -> float:
        """
        Avança a simulação em um passo de tempo (dt).
        T(k+1) = T(k) + dt * (beta*u - alpha*(T(k) - Tamb)) + ruído
        """
        loss = (self.t - self.p.Tamb) * self.p.alpha   # Perda para o ambiente
        gain = self.p.beta * u                           # Energia da Peltier/Resistor
        noise = np.random.normal(0.0, self.p.noise_std)

        self.t = self.t + self.p.dt * (gain - loss) + noise
        self.history.append(self.t)
        return self.t