import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr
from pid_baseline_final import PIDController, simulate
from pid_com_jitter import NetworkSimulator, simulate_com_jitter


# ============================================================
# ARQUITETURA DO SISTEMA SIMULADO
#
# Dois canais de comunicação entre PC e microcontrolador (ESP):
#
#   SENSOR  (ESP → PC): temperatura real viaja com atraso/perda
#                       e chega como temp_visualizada no controlador
#
#   COMANDO (PC → ESP): u_cmd calculado no PC viaja com atraso/perda
#                       e só vira u_aplicado quando o pacote chega
#                       (ESP mantém HOLD enquanto não chega novo comando)
#
# O controlador (PID ou PPO) opera sobre temp_visualizada,
# mas a recompensa/erro é medida sobre a temperatura real da planta.
# ============================================================


class VaporizadorJitterEnv(gym.Env):
    """
    Ambiente Gymnasium que simula o termociclador PCR com efeitos de rede.

    O agente PPO atua como controlador no PC:
        - Observa: histórico de 3 temperaturas + erro atual + setpoint atual
        - Ação: discreta entre {-1.0, 0.0, +1.0} (resfriar, manter, aquecer)
        - Recompensa: negativo do erro absoluto entre setpoint e temperatura real

    A comunicação com o sistema físico (ESP) é simulada com
    atraso, jitter e perda de pacotes nos dois canais.
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

        # configuração da rede simulada
        self.atraso_min = 1.0
        self.atraso_max = 3.5
        self.prob_perda = 0.3

        # flags para testes A/B (ligar/desligar jitter por canal)
        self.usar_jitter_sensor = True
        self.usar_jitter_comando = True

        # logs por passo — preenchidos durante o rollout para análise
        self.last_u_cmd = 0.0
        self.last_u_apl = 0.0
        self.last_temp_real = 25.0
        self.last_temp_vis = 25.0
        self.last_target = 25.0

    def reset(self, seed=None, options=None):
        """Reinicia o ambiente para um novo episódio de treinamento."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.params = ThermalParams()
        self.model = ThermalModel(self.params, t0=25.0)

        self.net = NetworkSimulator(
            atraso_min=self.atraso_min,
            atraso_max=self.atraso_max,
            prob_perda=self.prob_perda,
        )

        self.temp_visualizada = 25.0
        self.u_aplicado_atual = 0.0
        self.memoria_ia = deque([25.0, 25.0, 25.0], maxlen=3)
        self.steps = 0
        self._setpoints = gerar_setpoint_pcr(ciclos=10)  # episódios longos para treinamento

        target = self._setpoints[self.steps]
        self.last_target = target
        self.last_u_cmd = 0.0
        self.last_u_apl = 0.0
        self.last_temp_real = self.model.t
        self.last_temp_vis = self.temp_visualizada

        return self._get_obs(target), {}

    def step(self, action):
        """
        Avança o ambiente em um passo de tempo.

        Fluxo:
            1. Agente decide u_cmd
            2. u_cmd percorre a rede PC→ESP (com atraso/perda)
            3. Planta evolui com u_aplicado (último comando recebido pelo ESP)
            4. Temperatura real percorre a rede ESP→PC (com atraso/perda)
            5. Agente recebe obs com temp_visualizada (possivelmente defasada)
        """
        self.steps += 1

        # ação discreta do agente mapeada para sinal contínuo de controle
        u_cmd = {0: -1.0, 1: 0.0, 2: 1.0}[int(action)]
        self.last_u_cmd = u_cmd

        # canal PC → ESP: envia comando e tenta receber o mais recente
        if self.usar_jitter_comando:
            self.net.enviar_atuador(self.steps, u_cmd)
            self.u_aplicado_atual, _ = self.net.receber_atuador(
                self.steps, self.u_aplicado_atual
            )
        else:
            self.u_aplicado_atual = u_cmd

        self.last_u_apl = self.u_aplicado_atual

        # planta evolui com o comando que o ESP realmente está aplicando
        temp_real = self.model.step(self.u_aplicado_atual)
        self.last_temp_real = temp_real

        # canal ESP → PC: envia leitura do sensor e tenta receber
        if self.usar_jitter_sensor:
            self.net.enviar_sensor(self.steps, temp_real)
            self.temp_visualizada, _ = self.net.receber_sensor(
                self.steps, self.temp_visualizada
            )
        else:
            self.temp_visualizada = temp_real

        self.last_temp_vis = self.temp_visualizada
        self.memoria_ia.append(self.temp_visualizada)

        # setpoint do passo atual
        idx = min(self.steps, len(self._setpoints) - 1)
        target = self._setpoints[idx]
        self.last_target = target

        # recompensa baseada no erro real (não no visualizado)
        reward = -abs(target - temp_real)

        terminated = False
        truncated = self.steps >= 600

        return self._get_obs(target), reward, terminated, truncated, {}

    def _get_obs(self, target: float) -> np.ndarray:
        """
        Constrói o vetor de observação normalizado para o agente.

        Componentes:
            [t_atual, t-1, t-2] — histórico de temperaturas visualizadas (÷100)
            erro_norm           — (setpoint - t_atual) ÷ 100
            alvo_norm           — setpoint ÷ 100
        """
        historico = list(self.memoria_ia)
        return np.array(
            [
                historico[-1] / 100.0,
                historico[-2] / 100.0,
                historico[-3] / 100.0,
                (target - historico[-1]) / 100.0,
                target / 100.0,
            ],
            dtype=np.float32,
        )


def rodar_simulacao_ia(seed: int = 42) -> dict | None:
    """
    Carrega o modelo PPO treinado e executa uma simulação de 600 passos.

    Returns:
        Dicionário com históricos de temperatura, setpoint e comandos,
        ou None se o modelo não for encontrado.
    """
    print("Rodando IA (PPO)...")
    env = VaporizadorJitterEnv()

    try:
        caminho_modelo = os.path.join(os.path.dirname(__file__), "ppo_pcr_jitter_final")
        model = PPO.load(caminho_modelo, env=env)
    except Exception as e:
        print("Modelo PPO não encontrado ('ppo_pcr_jitter_final.zip').")
        print("Detalhe:", e)
        return None

    obs, _ = env.reset(seed=seed)

    temps_real, temps_vis, alvos, u_cmd_hist, u_apl_hist = [], [], [], [], []

    for _ in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)

        temps_real.append(env.last_temp_real)
        temps_vis.append(env.last_temp_vis)
        alvos.append(env.last_target)
        u_cmd_hist.append(env.last_u_cmd)
        u_apl_hist.append(env.last_u_apl)

    return {
        "temps_real": temps_real,
        "temps_vis": temps_vis,
        "alvos": alvos,
        "u_cmd": u_cmd_hist,
        "u_apl": u_apl_hist,
    }


def rodar_simulacao_pid(seed: int = 42) -> dict:
    """
    Executa a simulação do PID com jitter usando os módulos já construídos.

    Returns:
        Dicionário com históricos de temperatura, setpoint e comandos.
    """
    print("Rodando PID...")
    np.random.seed(seed)

    params = ThermalParams()
    model = ThermalModel(params, t0=25.0)
    setpoints = gerar_setpoint_pcr(ciclos=10)[:600]
    pid = PIDController(kp=0.2, ki=0.5, kd=0.0)
    net = NetworkSimulator(atraso_min=1.0, atraso_max=3.5, prob_perda=0.3)

    temps, _ = simulate_com_jitter(model, pid, net, setpoints)

    # reconstrói temp_visualizada e u_apl rodando de novo (para os logs de análise)
    # usa a mesma seed para garantir mesma rede
    np.random.seed(seed)
    model.reset(t0=25.0)
    pid.reset()
    net2 = NetworkSimulator(atraso_min=1.0, atraso_max=3.5, prob_perda=0.3)

    temps_real, temps_vis, alvos, u_cmd_hist, u_apl_hist = [], [], [], [], []
    temp_vis = 25.0
    u_apl = 0.0

    for t, sp in enumerate(setpoints):
        net2.enviar_sensor(t, model.t)
        temp_vis, _ = net2.receber_sensor(t, temp_vis)

        u_cmd = pid.compute(sp=sp, temp_atual=temp_vis, dt=model.p.dt)

        net2.enviar_atuador(t, u_cmd)
        u_apl, _ = net2.receber_atuador(t, u_apl)

        model.step(u_apl)

        temps_real.append(model.t)
        temps_vis.append(temp_vis)
        alvos.append(sp)
        u_cmd_hist.append(u_cmd)
        u_apl_hist.append(u_apl)

    return {
        "temps_real": temps_real,
        "temps_vis": temps_vis,
        "alvos": alvos,
        "u_cmd": u_cmd_hist,
        "u_apl": u_apl_hist,
    }


def plot_comparacao(ia_data: dict, pid_data: dict) -> None:
    """
    Gera 5 gráficos comparativos entre PID e PPO, salvando em graficos/.

    Gráficos gerados:
        01 — Comparativo principal: PID vs PPO vs Setpoint
        02 — PID: temperatura real vs visualizada (efeito da rede no sensor)
        03 — PID: u_cmd vs u_aplicado (efeito da rede no comando)
        04 — PPO: temperatura real vs visualizada
        05 — PPO: u_cmd vs u_aplicado
    """
    def _salvar(nome: str) -> None:
        plt.tight_layout()
        plt.savefig(f"graficos/{nome}", dpi=200, bbox_inches="tight")
        plt.show()
        plt.close()

    # 01 — comparativo principal
    plt.figure(figsize=(12, 6))
    plt.plot(pid_data["alvos"], "k--", label="Setpoint (PCR)", linewidth=2, alpha=0.6)
    plt.plot(pid_data["temps_real"], "r-", label="PID Clássico (com Jitter)", linewidth=1.5, alpha=0.8)
    plt.plot(ia_data["temps_real"], "b-", label='IA (PPO) — "Zen Mode"', linewidth=2.0)
    plt.title("Batalha Final: Controle Clássico (PID) vs Inteligência Artificial (RL)")
    plt.ylabel("Temperatura (°C)")
    plt.xlabel("Tempo (segundos)")
    plt.legend()
    plt.grid(True)
    _salvar("01_comparativo_pid_vs_ppo.png")

    # 02 — PID: sensor real vs visualizado
    plt.figure(figsize=(12, 4))
    plt.plot(pid_data["alvos"], "k--", label="Setpoint", alpha=0.6)
    plt.plot(pid_data["temps_real"], label="PID: temp_real (planta)")
    plt.plot(pid_data["temps_vis"], label="PID: temp_visualizada (PC)")
    plt.title("PID: Efeito da rede no sensor (ESP→PC)")
    plt.ylabel("Temperatura (°C)")
    plt.xlabel("Tempo (segundos)")
    plt.legend()
    plt.grid(True)
    _salvar("02_pid_sensor_real_vs_visualizada.png")

    # 03 — PID: comando decidido vs aplicado
    plt.figure(figsize=(12, 4))
    plt.plot(pid_data["u_cmd"], label="PID: u_cmd (PC decide)")
    plt.plot(pid_data["u_apl"], label="PID: u_aplicado (ESP aplica)")
    plt.title("PID: Efeito da rede no comando (PC→ESP)")
    plt.ylabel("u")
    plt.xlabel("Tempo (segundos)")
    plt.legend()
    plt.grid(True)
    _salvar("03_pid_comando_ucmd_vs_uaplicado.png")

    # 04 — PPO: sensor real vs visualizado
    plt.figure(figsize=(12, 4))
    plt.plot(ia_data["alvos"], "k--", label="Setpoint", alpha=0.6)
    plt.plot(ia_data["temps_real"], label="PPO: temp_real (planta)")
    plt.plot(ia_data["temps_vis"], label="PPO: temp_visualizada (PC)")
    plt.title("PPO: Efeito da rede no sensor (ESP→PC)")
    plt.ylabel("Temperatura (°C)")
    plt.xlabel("Tempo (segundos)")
    plt.legend()
    plt.grid(True)
    _salvar("04_ppo_sensor_real_vs_visualizada.png")

    # 05 — PPO: comando decidido vs aplicado
    plt.figure(figsize=(12, 4))
    plt.plot(ia_data["u_cmd"], label="PPO: u_cmd (PC decide)")
    plt.plot(ia_data["u_apl"], label="PPO: u_aplicado (ESP aplica)")
    plt.title("PPO: Efeito da rede no comando (PC→ESP)")
    plt.ylabel("u")
    plt.xlabel("Tempo (segundos)")
    plt.legend()
    plt.grid(True)
    _salvar("05_ppo_comando_ucmd_vs_uaplicado.png")


def main() -> None:
    os.makedirs("graficos", exist_ok=True)  # cria a pasta de saída se não existir

    pid_data = rodar_simulacao_pid(seed=42)
    ia_data = rodar_simulacao_ia(seed=42)

    if ia_data is None:
        # PPO não disponível — plota apenas o PID com debug de rede
        print("Plotando apenas PID (modelo PPO não encontrado).")
        plt.figure(figsize=(12, 6))
        plt.plot(pid_data["alvos"], "k--", label="Setpoint (PCR)", linewidth=2, alpha=0.6)
        plt.plot(pid_data["temps_real"], "r-", label="PID: temp_real", linewidth=1.5)
        plt.plot(pid_data["temps_vis"], "g-", label="PID: temp_visualizada", linewidth=1.0, alpha=0.8)
        plt.title("PID com rede: temperatura real vs visualizada")
        plt.ylabel("Temperatura (°C)")
        plt.xlabel("Tempo (segundos)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        plot_comparacao(ia_data, pid_data)


if __name__ == "__main__":
    main()