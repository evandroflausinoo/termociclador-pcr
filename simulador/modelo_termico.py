import numpy as np
from dataclasses import dataclass, field


@dataclass
class ThermalParams:
    Tamb: float = 25.0       # Temperatura ambiente (°C)
    alpha: float = 0.02      # Taxa de perda de calor pro ambiente
    beta: float = 3.0        # Ganho térmico (quanto o controle aquece/resfria)
    dt: float = 1.0          # Passo de tempo (1 segundo por iteração)
    noise_std: float = 0.05  # Intensidade do ruído do sensor


class ThermalModel:
    def __init__(self, params: ThermalParams, t0: float = 25.0):
        self.p = params
        self.t = t0
        self.history: list[float] = [t0]  # MUDANÇA 4: histórico interno

    def reset(self, t0: float = 25.0) -> float:
        """Reinicia o modelo — usado principalmente pelo agente PPO entre episódios."""
        self.t = t0
        self.history = [t0]  # limpa o histórico ao reiniciar
        return self.t

    def step(self, u: float) -> float:  # MUDANÇA 1: tipo correto (float, não int)
        """
        Avança a simulação em um passo de tempo (dt).

        Equação térmica discreta:
            T(k+1) = T(k) + dt * (beta*u - alpha*(T(k) - Tamb)) + ruído

        Args:
            u: sinal de controle no intervalo [-1.0, 1.0]
               positivo = aquece, negativo = resfria

        Returns:
            Nova temperatura após o passo.
        """
        loss = (self.t - self.p.Tamb) * self.p.alpha   # perda de calor pro ambiente
        gain = self.p.beta * u                           # energia injetada pelo controle
        noise = np.random.normal(0.0, self.p.noise_std) # MUDANÇA 2: numpy para reprodutibilidade

        t_next = self.t + self.p.dt * (gain - loss) + noise

        # MUDANÇA 3: clamp físico — temperatura não cai abaixo de Tamb - 5°C
        self.t = max(self.p.Tamb - 5.0, t_next)

        self.history.append(self.t)  # MUDANÇA 4: salva no histórico
        return self.t