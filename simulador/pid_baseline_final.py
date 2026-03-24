import numpy as np
import matplotlib.pyplot as plt

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr


class PIDController:
    """
    Controlador PID com anti-windup e reset de integral por troca de fase.

    O sinal de controle é calculado como:
        u = Kp*e + Ki*∫e dt + Kd*(de/dt)

    O termo derivativo é baseado na variação da temperatura (não do erro),
    evitando o "tranco derivativo" que ocorre em mudanças bruscas de setpoint.

    Anti-windup: quando o controle satura em u_min ou u_max, a integral
    só é atualizada se o erro estiver na direção oposta à saturação,
    impedindo acúmulo indefinido.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        u_min: float = -1.0,
        u_max: float = 1.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max

        self._erro_acumulado: float = 0.0
        self._sp_anterior: float | None = None
        self._temp_anterior: float | None = None

    def reset(self) -> None:
        """Reinicia o estado interno do controlador."""
        self._erro_acumulado = 0.0
        self._sp_anterior = None
        self._temp_anterior = None

    def compute(self, sp: float, temp_atual: float, dt: float) -> float:
        """
        Calcula o sinal de controle para um passo de tempo.

        Args:
            sp:         setpoint (temperatura alvo em °C)
            temp_atual: temperatura atual medida (°C)
            dt:         passo de tempo em segundos

        Returns:
            Sinal de controle u no intervalo [u_min, u_max].
        """
        # reset da integral em cada troca de fase (novo setpoint)
        if self._sp_anterior is not None and sp != self._sp_anterior:
            self._erro_acumulado = 0.0
        self._sp_anterior = sp

        if self._temp_anterior is None:
            self._temp_anterior = temp_atual

        erro = sp - temp_atual

        P = self.kp * erro

        I_tentativa = self._erro_acumulado + (erro * dt)

        # derivativo baseado na variação da temperatura (evita tranco)
        velocidade_temp = (temp_atual - self._temp_anterior) / dt
        D = -self.kd * velocidade_temp

        u_raw = P + (self.ki * I_tentativa) + D

        # anti-windup: só atualiza integral se não piorar a saturação
        if u_raw > self.u_max:
            u = self.u_max
            if erro < 0:
                self._erro_acumulado = I_tentativa
        elif u_raw < self.u_min:
            u = self.u_min
            if erro > 0:
                self._erro_acumulado = I_tentativa
        else:
            u = u_raw
            self._erro_acumulado = I_tentativa

        self._temp_anterior = temp_atual
        return u


def simulate(
    model: ThermalModel,
    pid: PIDController,
    setpoints: np.ndarray,
    verbose: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Executa a simulação do controle PID sobre o modelo térmico.

    Args:
        model:     modelo térmico já instanciado
        pid:       controlador PID já instanciado
        setpoints: array de setpoints ao longo do tempo
        verbose:   se True, imprime tabela passo a passo no terminal

    Returns:
        Tupla (temperaturas, controles) com os valores registrados a cada passo.
    """
    model.reset(t0=25.0)
    pid.reset()

    temps: list[float] = []
    controls: list[float] = []

    if verbose:
        print(f"{'Tempo':<6} {'Alvo':<8} {'Temp':<8} {'u':<8}")

    for t, sp in enumerate(setpoints):
        u = pid.compute(sp=sp, temp_atual=model.t, dt=model.p.dt)
        model.step(u)

        temps.append(model.t)
        controls.append(u)

        if verbose:
            print(f"{t:<6} {sp:<8.1f} {model.t:<8.2f} {u:<8.3f}")

    return temps, controls


def plot_resultado(setpoints: np.ndarray, temps: list[float]) -> None:
    """Plota a comparação entre setpoint e temperatura real ao longo do tempo."""
    tempo = np.arange(len(setpoints))

    plt.figure(figsize=(12, 5))
    plt.plot(tempo, setpoints, "k--", label="Setpoint (PCR)", linewidth=1.5)
    plt.plot(tempo, temps, "b-", label="PID Clássico (sem jitter)", linewidth=1.5)
    plt.xlabel("Tempo (segundos)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Controle PID — Termociclador PCR (sem efeitos de rede)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficos/pid_baseline.png", dpi=150)
    plt.show()


def main() -> None:
    np.random.seed(42)  # garante reprodutibilidade do ruído

    params = ThermalParams(Tamb=25.0, alpha=0.02, beta=3.0, dt=1.0, noise_std=0.05)
    model = ThermalModel(params, t0=25.0)
    setpoints = gerar_setpoint_pcr(ciclos=2)

    pid = PIDController(kp=0.2, ki=0.5, kd=0.0)

    temps, controls = simulate(model, pid, setpoints, verbose=True)
    plot_resultado(setpoints, temps)


if __name__ == "__main__":
    main()