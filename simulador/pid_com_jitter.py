import heapq
import random
import numpy as np
import matplotlib.pyplot as plt

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr
from pid_baseline_final import PIDController


class NetworkSimulator:
    """
    Simula os efeitos de uma rede de comunicação entre o PC e o microcontrolador.

    Modela três fenômenos reais de redes:
        - Atraso (latência): pacotes demoram um tempo aleatório para chegar
        - Jitter: a variação no atraso entre pacotes sucessivos
        - Perda de pacotes: alguns pacotes simplesmente não chegam

    Internamente usa duas filas de prioridade (heapq) ordenadas pelo tempo
    de chegada — uma para o canal sensor→PC e outra para o canal PC→atuador.
    """

    def __init__(
        self,
        atraso_min: float = 1.0,
        atraso_max: float = 3.5,
        prob_perda: float = 0.3,
    ):
        self.atraso_min = atraso_min
        self.atraso_max = atraso_max
        self.prob_perda = prob_perda

        self._buffer_sensor: list = []    # canal ESP → PC (temperatura)
        self._buffer_atuador: list = []   # canal PC → ESP (comando u)

    def enviar_sensor(self, t: int, valor: float) -> None:
        """
        Simula o envio da leitura do sensor (ESP → PC).
        O pacote pode ser perdido ou chegar com atraso aleatório.
        """
        if random.random() > self.prob_perda:
            atraso = random.uniform(self.atraso_min, self.atraso_max)
            heapq.heappush(self._buffer_sensor, (t + atraso, valor))

    def receber_sensor(self, t: int, ultimo_valor: float) -> tuple[float, str]:
        """
        Tenta receber o pacote mais recente do sensor.

        Returns:
            Tupla (valor, status) onde status é 'OK' se chegou pacote novo,
            ou 'LAG' se ainda está usando a última leitura conhecida.
        """
        recebeu = False
        while self._buffer_sensor and self._buffer_sensor[0][0] <= t:
            _, valor = heapq.heappop(self._buffer_sensor)
            ultimo_valor = valor
            recebeu = True
        return ultimo_valor, ("OK" if recebeu else "LAG")

    def enviar_atuador(self, t: int, u_cmd: float) -> None:
        """
        Simula o envio do comando de controle (PC → ESP).
        O pacote pode ser perdido ou chegar com atraso aleatório.
        """
        if random.random() > self.prob_perda:
            atraso = random.uniform(self.atraso_min, self.atraso_max)
            heapq.heappush(self._buffer_atuador, (t + atraso, u_cmd))

    def receber_atuador(self, t: int, ultimo_u: float) -> tuple[float, str]:
        """
        Tenta receber o comando mais recente enviado pelo PC.

        Returns:
            Tupla (u, status) onde status é 'OK' se chegou comando novo,
            ou 'HOLD' se o ESP ainda está aplicando o último comando recebido.
        """
        recebeu = False
        while self._buffer_atuador and self._buffer_atuador[0][0] <= t:
            _, u = heapq.heappop(self._buffer_atuador)
            ultimo_u = u
            recebeu = True
        return ultimo_u, ("OK" if recebeu else "HOLD")


def simulate_com_jitter(
    model: ThermalModel,
    pid: PIDController,
    net: NetworkSimulator,
    setpoints: np.ndarray,
    verbose: bool = False,
) -> tuple[list[float], list[float]]:
    """
    Executa a simulação do controle PID com efeitos de rede (jitter, atraso, perda).

    O controlador roda no PC e enxerga a temperatura com atraso.
    O microcontrolador aplica o último comando recebido enquanto aguarda novos pacotes.

    Args:
        model:     modelo térmico já instanciado
        pid:       controlador PID já instanciado
        net:       simulador de rede já instanciado
        setpoints: array de setpoints ao longo do tempo
        verbose:   se True, imprime tabela passo a passo no terminal

    Returns:
        Tupla (temperaturas, controles) registrados a cada passo.
    """
    model.reset(t0=25.0)
    pid.reset()

    temps: list[float] = []
    controls: list[float] = []

    temp_visualizada = 25.0   # última temperatura conhecida pelo PC
    u_aplicado = 0.0          # último comando aplicado pelo ESP

    if verbose:
        print(
            f"{'Tempo':<6} {'Alvo':<8} {'T_Real':<8} {'T_PC':<8} "
            f"{'u_cmd':<8} {'u_esp':<8} {'StatusY':<8} {'StatusU':<8}"
        )

    for t, sp in enumerate(setpoints):

        # ESP envia leitura do sensor pro PC
        net.enviar_sensor(t, model.t)

        # PC tenta receber a leitura (pode estar em LAG)
        temp_visualizada, status_y = net.receber_sensor(t, temp_visualizada)

        # PC calcula o comando baseado na temperatura que enxerga
        u_cmd = pid.compute(sp=sp, temp_atual=temp_visualizada, dt=model.p.dt)

        # PC envia o comando pro ESP
        net.enviar_atuador(t, u_cmd)

        # ESP tenta receber o comando (pode estar em HOLD)
        u_aplicado, status_u = net.receber_atuador(t, u_aplicado)

        # planta evolui com o comando que o ESP realmente está aplicando
        model.step(u_aplicado)

        temps.append(model.t)
        controls.append(u_aplicado)

        if verbose:
            print(
                f"{t:<6} {sp:<8.1f} {model.t:<8.2f} {temp_visualizada:<8.2f} "
                f"{u_cmd:<8.3f} {u_aplicado:<8.3f} {status_y:<8} {status_u:<8}"
            )

    return temps, controls


def plot_resultado(setpoints: np.ndarray, temps: list[float]) -> None:
    """Plota a comparação entre setpoint e temperatura real com efeitos de rede."""
    tempo = np.arange(len(setpoints))

    plt.figure(figsize=(12, 5))
    plt.plot(tempo, setpoints, "k--", label="Setpoint (PCR)", linewidth=1.5)
    plt.plot(tempo, temps, "r-", label="PID Clássico (com jitter)", linewidth=1.5)
    plt.xlabel("Tempo (segundos)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Controle PID com Efeitos de Rede — Termociclador PCR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graficos/pid_jitter.png", dpi=150)
    plt.show()


def main() -> None:
    np.random.seed(42)

    params = ThermalParams(Tamb=25.0, alpha=0.02, beta=3.0, dt=1.0, noise_std=0.05)
    model = ThermalModel(params, t0=25.0)
    setpoints = gerar_setpoint_pcr(ciclos=2)

    pid = PIDController(kp=0.2, ki=0.5, kd=0.0)
    net = NetworkSimulator(atraso_min=1.0, atraso_max=3.5, prob_perda=0.3)

    temps, controls = simulate_com_jitter(model, pid, net, setpoints, verbose=True)
    plot_resultado(setpoints, temps)


if __name__ == "__main__":
    main()