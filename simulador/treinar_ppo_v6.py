import sys
import os
import time
import multiprocessing
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# Garante a importação dos módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr
from pid_com_jitter import NetworkSimulator


class TermocicladorRealEnvV6Final(gym.Env):
    """
    Ambiente v6 Final — Correção de Sim-to-Real e Frenagem Antecipatória.
    Sintonizado com beta=0.15 para eliminar o overshoot em 72°C e 95°C.
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)

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

        # ⚡ CORREÇÃO CRUCIAL SIM-TO-REAL: beta=0.15 reflete a retenção real de calor do alumínio
        self.params = ThermalParams(Tamb=25.0, alpha=0.005, beta=0.15, dt=1.0, noise_std=0.3)
        self.model = ThermalModel(self.params, t0=25.0)

        self.net = NetworkSimulator(
            atraso_min=self.atraso_min, atraso_max=self.atraso_max, prob_perda=self.prob_perda
        )

        self.temp_visualizada = 25.0
        self.u_aplicado_atual = 0.0
        self.memoria_ia = deque([25.0, 25.0, 25.0], maxlen=3)
        self.steps = 0

        ciclos = np.random.randint(3, 8)
        self._setpoints = gerar_setpoint_pcr(
            ciclos=ciclos, t_estabilizacao=20, t_desnaturacao=120, t_anelamento=180, t_extensao=120
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
        self.u_aplicado_atual, _ = self.net.receber_atuador(self.steps, self.u_aplicado_atual)
        self.last_u_apl = self.u_aplicado_atual

        temp_real = self.model.step(self.u_aplicado_atual)
        self.last_temp_real = temp_real

        self.net.enviar_sensor(self.steps, temp_real)
        self.temp_visualizada, _ = self.net.receber_sensor(self.steps, self.temp_visualizada)
        self.last_temp_vis = self.temp_visualizada
        self.memoria_ia.append(self.temp_visualizada)

        idx = min(self.steps, len(self._setpoints) - 1)
        target = self._setpoints[idx]
        self.last_target = target

        # ============================================================
        # RECOMPENSA v6 FINAL — PENALIZAÇÃO ASSIMÉTRICA RIGOROSA
        # ============================================================
        erro_sinal = target - temp_real
        erro_abs = abs(erro_sinal)

        # 1. Recompensa base inversamente proporcional ao erro
        reward = -erro_abs

        # 2. Bônus de Fixação Exata no Patamar
        if erro_abs < 0.5:
            reward += 8.0
        elif erro_abs < 1.5:
            reward += 3.0

        # 3. PENALIDADE MORTAL PARA AQUECIMENTO ACIMA DA META (ANTI-OVERSHOOT)
        # Se T_real > target (erro_sinal < 0) e a IA insistir em u > 0:
        if erro_sinal < -0.2 and u_cmd > 0.0:
            reward -= (u_cmd * 35.0) + (abs(erro_sinal) * 6.0)

        # 4. RECOMPENSA DE FRENAGEM ANTECIPATÓRIA
        # Quando estiver a menos de 2°C de atingir o alvo vindo de baixo,
        # recompensa fortemente a IA por reduzir a potência (u <= 0.10)
        if 0.0 <= erro_sinal <= 2.0 and u_cmd <= 0.10:
            reward += 4.0

        terminated = False
        truncated = self.steps >= self.max_steps

        return self._get_obs(target), reward, terminated, truncated, {}

    def _get_obs(self, target: float) -> np.ndarray:
        historico = list(self.memoria_ia)
        # Erro dividido por 20.0 para multiplicar a sensibilidade do desvio por 5x
        return np.array([
            historico[-1] / 100.0,
            historico[-2] / 100.0,
            historico[-3] / 100.0,
            (target - historico[-1]) / 20.0,
            target / 100.0,
        ], dtype=np.float32)


def main():
    os.makedirs("modelos", exist_ok=True)

    TOTAL_STEPS = 1_000_000
    NUM_ENVS = 3  # Configuração ideal para Intel i3 com 4 threads e 4GB RAM

    print(f"\n{'='*65}")
    print(f"🚀 TREINAMENTO PPO v6 FINAL — CONTROLE RIGOROSO ANTI-OVERSHOOT")
    print(f"🔥 Paralelizando treino em {NUM_ENVS} ambientes virtuais (SubprocVecEnv)")
    print(f"🎯 Foco: Eliminar o overshoot de +5°C em 72°C e 95°C")
    print(f"{'='*65}\n")

    env = make_vec_env(
        TermocicladorRealEnvV6Final,
        n_envs=NUM_ENVS,
        vec_env_cls=SubprocVecEnv
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=30_000, save_path="modelos/", name_prefix="ppo_pcr_v6"
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        device="cpu",
    )

    print("Iniciando aprendizado das 3 instâncias em paralelo...\n")
    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_cb)

    model.save("modelos/ppo_pcr_v6_final")
    print("\n✅ Treinamento concluído! Modelo salvo em 'modelos/ppo_pcr_v6_final.zip'")


if __name__ == "__main__":
    main()