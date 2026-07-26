import sys
import os
import time
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Ajusta as rotas do Python para encontrar os módulos em qualquer subpasta
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_SIMULADOR = os.path.abspath(os.path.join(DIR_ATUAL, ".."))

sys.path.append(DIR_ATUAL)
sys.path.append(DIR_SIMULADOR)

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr


class TermocicladorRealEnvV8(gym.Env):
    """
    Ambiente v8 — Gêmeo Digital Fiel com Parâmetros Medidos na Bancada.
    Inclui inércia do alumínio e simulação de Jitter/Perda de Pacote.
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)

        # Modelo térmico calibrado com a física do protótipo
        self.params = ThermalParams(
            Tamb=25.0,
            alpha=0.0035,
            beta=0.48,
            dt=1.0,
            noise_std=0.05
        )
        self.model = ThermalModel(self.params)

        self.memoria_ia = deque(maxlen=3)
        self.passo_atual = 0
        self.setpoints = []
        self.max_passos = 600
        self.u_prev = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        temp_init = np.random.uniform(22.0, 26.0)
        self.model.reset(t0=temp_init)

        self.setpoints = gerar_setpoint_pcr()
        self.max_passos = len(self.setpoints)
        self.passo_atual = 0
        self.u_prev = 0.0

        self.memoria_ia.clear()
        for _ in range(3):
            self.memoria_ia.append(temp_init)

        obs = self._get_obs(self.setpoints[0])
        return obs, {}

    def step(self, action):
        u_raw = float(np.clip(action[0], -1.0, 1.0))
        target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]

        # Simulação de canal com 5% de perda de pacote (mantém o último comando em caso de falha)
        if np.random.rand() < 0.05:
            u_apl = self.u_prev
        else:
            u_apl = u_raw
            self.u_prev = u_apl

        # Avança a física no modelo térmico
        nova_temp = self.model.step(u_apl)
        self.memoria_ia.append(nova_temp)

        # Cálculo de Erro e Recompensa
        erro = abs(target - nova_temp)
        reward = -erro

        # Bônus por estabilização na faixa de precisão de PCR (±0.4°C)
        if erro <= 0.4:
            reward += 1.0

        self.passo_atual += 1
        terminated = self.passo_atual >= self.max_passos
        truncated = False

        prox_target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]
        obs = self._get_obs(prox_target)

        return obs, reward, terminated, truncated, {"temp_real": nova_temp, "u_apl": u_apl}

    def _get_obs(self, target: float) -> np.ndarray:
        historico = list(self.memoria_ia)
        return np.array([
            historico[-1] / 100.0,
            historico[-2] / 100.0,
            historico[-3] / 100.0,
            (target - historico[-1]) / 50.0,
            target / 100.0,
        ], dtype=np.float32)


def main():
    os.makedirs("modelos", exist_ok=True)

    TOTAL_STEPS = 1_000_000
    NUM_ENVS = 3

    print(f"\n{'='*65}")
    print(f"🚀 TREINAMENTO PPO v8 (1.000.000 PASSOS) — GÊMEO DIGITAL REALISTA")
    print(f"🎯 Objetivo: Treinar PPO com estabilização e inércia do alumínio")
    print(f"{'='*65}\n")

    env = make_vec_env(
        TermocicladorRealEnvV8,
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000, save_path="modelos/", name_prefix="ppo_pcr_v8_ckpt"
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1
    )

    t_inicio = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_cb)
    duracao = (time.time() - t_inicio) / 60.0

    caminho_final = os.path.join("modelos", "ppo_pcr_v8_final")
    model.save(caminho_final)

    print(f"\n{'='*65}")
    print(f"✅ TREINAMENTO PPO v8 CONCLUÍDO COM SUCESSO!")
    print(f"⏱️  Tempo total: {duracao:.2f} minutos")
    print(f"💾 Modelo salvo em: {caminho_final}.zip")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()