import sys
import os
import time
import numpy as np
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr
from pid_com_jitter import NetworkSimulator


# ============================================================
# PPO v2 — Recompensa Básica Tradicional (Erro Absoluto)
#
# Estrutura v2:
#   - Penalização proporcional ao erro absoluto: -abs(target - temp_real)
#   - Bônus simples para quando está dentro de ±1.0°C do alvo
#   - Sem componente de velocidade ou penalidades dinâmicas
# ============================================================


class TermocicladorRealEnvV2(gym.Env):
    """
    Ambiente v2 — Recompensa tradicional de acompanhamento de setpoint.
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

        self.atraso_min = 1.0
        self.atraso_max = 3.5
        self.prob_perda = 0.3

        self.last_temp_real = 25.0
        self.last_temp_vis = 25.0
        self.last_target = 25.0
        self.last_u_cmd = 0.0
        self.last_u_apl = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.params = ThermalParams(
            Tamb=25.0,
            alpha=0.005,
            beta=0.4,
            dt=1.0,
            noise_std=0.3
        )
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

        ciclos = np.random.randint(3, 8)
        self._setpoints = gerar_setpoint_pcr(
            ciclos=ciclos,
            t_estabilizacao=20,
            t_desnaturacao=120,
            t_anelamento=180,
            t_extensao=120,
        )
        self.max_steps = len(self._setpoints)

        target = self._setpoints[0]
        self.last_target = target
        self.last_temp_real = self.model.t
        self.last_temp_vis = self.temp_visualizada

        return self._get_obs(target), {}

    def step(self, action):
        self.steps += 1

        u_cmd = float(action[0])
        u_cmd = np.clip(u_cmd, -1.0, 1.0)
        self.last_u_cmd = u_cmd

        self.net.enviar_atuador(self.steps, u_cmd)
        self.u_aplicado_atual, _ = self.net.receber_atuador(
            self.steps, self.u_aplicado_atual
        )
        self.last_u_apl = self.u_aplicado_atual

        temp_real = self.model.step(self.u_aplicado_atual)
        self.last_temp_real = temp_real

        self.net.enviar_sensor(self.steps, temp_real)
        self.temp_visualizada, _ = self.net.receber_sensor(
            self.steps, self.temp_visualizada
        )
        self.last_temp_vis = self.temp_visualizada
        self.memoria_ia.append(self.temp_visualizada)

        idx = min(self.steps, len(self._setpoints) - 1)
        target = self._setpoints[idx]
        self.last_target = target

        # ============================================================
        # RECOMPENSA v2 — Estrutura Básica Tradicional
        # ============================================================
        erro_abs = abs(target - temp_real)

        # 1. Recompensa base: penalização direta do erro
        reward = -erro_abs

        # 2. Bônus simples de estabilização
        if erro_abs < 1.0:
            reward += 1.0
        # ============================================================

        terminated = False
        truncated = self.steps >= self.max_steps

        return self._get_obs(target), reward, terminated, truncated, {}

    def _get_obs(self, target: float) -> np.ndarray:
        historico = list(self.memoria_ia)
        return np.array([
            historico[-1] / 100.0,
            historico[-2] / 100.0,
            historico[-3] / 100.0,
            (target - historico[-1]) / 100.0,
            target / 100.0,
        ], dtype=np.float32)


class ProgressCallback(BaseCallback):
    def __init__(self, total_steps: int, verbose=0):
        super().__init__(verbose)
        self.total_steps = total_steps
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"TREINAMENTO PPO v2 — Recompensa Básica Tradicional")
        print(f"Objetivo: Acompanhamento de erro simples com atraso de rede")
        print(f"Total de passos: {self.total_steps:,}")
        print(f"{'='*60}\n")

    def _on_step(self) -> bool:
        if self.num_timesteps % 10_000 == 0:
            elapsed = time.time() - self.start_time
            progresso = self.num_timesteps / self.total_steps
            restante = (elapsed / progresso) - elapsed if progresso > 0 else 0
            print(
                f"Passo: {self.num_timesteps:>8,} / {self.total_steps:,} "
                f"({progresso*100:.1f}%) | "
                f"Tempo: {elapsed/60:.1f}min | "
                f"Restante: {restante/60:.1f}min"
            )
        return True


def main():
    os.makedirs("graficos", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    TOTAL_STEPS = 1_000_000

    env = Monitor(TermocicladorRealEnvV2())

    progress_cb = ProgressCallback(total_steps=TOTAL_STEPS)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path="modelos/",
        name_prefix="ppo_pcr_v2",
        verbose=1
    )

    print("Criando modelo PPO v2...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",
    )

    print("Iniciando treinamento...\n")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[progress_cb, checkpoint_cb],
    )

    # Salva o arquivo na raiz ou na pasta de modelos
    model.save("modelos/ppo_pcr_v2_final")
    model.save("ppo_pcr_v2_final")
    print("\n✅ Modelo salvo como 'ppo_pcr_v2_final.zip'")
    print("✅ Checkpoints salvos em 'modelos/'")


if __name__ == "__main__":
    main()