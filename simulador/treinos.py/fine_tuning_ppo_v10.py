import sys
import os
import time
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_SIMULADOR = os.path.abspath(os.path.join(DIR_ATUAL, ".."))

sys.path.append(DIR_ATUAL)
sys.path.append(DIR_SIMULADOR)

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr


class TermocicladorV10OtimizadoEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(6,), dtype=np.float32)

        self.params = ThermalParams(Tamb=25.0, alpha=0.0035, beta=0.48, dt=1.0, noise_std=0.05)
        self.model = ThermalModel(self.params)

        self.memoria_ia = deque(maxlen=4)
        self.passo_atual = 0
        self.setpoints = []
        self.max_passos = 600

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        temp_init = np.random.uniform(22.0, 26.0)
        self.model.reset(t0=temp_init)

        self.setpoints = gerar_setpoint_pcr()
        self.max_passos = len(self.setpoints)
        self.passo_atual = 0

        self.memoria_ia.clear()
        for _ in range(4):
            self.memoria_ia.append(temp_init)

        return self._get_obs(self.setpoints[0]), {}

    def step(self, action):
        u_apl = float(np.clip(action[0], -1.0, 1.0))
        target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]

        nova_temp = self.model.step(u_apl)
        self.memoria_ia.append(nova_temp)

        erro = target - nova_temp
        abs_erro = abs(erro)
        velocidade = self.memoria_ia[-1] - self.memoria_ia[-2]

        reward = -abs_erro

        # 1. SE O ERRO FOR GRANDE: Recompensa velocidade alta para forçar u = 1.0!
        if abs_erro > 3.0:
            if (erro > 0 and u_apl > 0.8) or (erro < 0 and u_apl < -0.8):
                reward += 1.5  # Bônus por aplicar força total durante as rampas

        # 2. SE ESTIVER PERTO DO SETPOINT (<= 1.5°C): Exige redução e frenagem
        elif abs_erro <= 1.5:
            # Penaliza velocidade excessiva no momento do pouso para zerar overshoot
            if abs(velocidade) > 0.3:
                reward -= abs(velocidade) * 2.0

        # Bônus por travamento perfeito (±0.3°C)
        if abs_erro <= 0.3:
            reward += 3.0

        self.passo_atual += 1
        terminated = self.passo_atual >= self.max_passos

        prox_target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]
        obs = self._get_obs(prox_target)

        return obs, reward, terminated, False, {"temp_real": nova_temp, "u_apl": u_apl}

    def _get_obs(self, target: float) -> np.ndarray:
        historico = list(self.memoria_ia)
        t_atual = historico[-1]
        t_anterior = historico[-2]
        velocidade_dT = t_atual - t_anterior

        return np.array([
            t_atual / 100.0,
            t_anterior / 100.0,
            historico[-3] / 100.0,
            (target - t_atual) / 50.0,
            target / 100.0,
            velocidade_dT / 2.0
        ], dtype=np.float32)


def main():
    print(f"\n{'='*65}")
    print(f"🚀 RE-TREINANDO PPO v10 — RECOMPENSA DE POTÊNCIA MÁXIMA NAS RAMPAS")
    print(f"{'='*65}\n")

    env = TermocicladorV10OtimizadoEnv()

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
        verbose=1
    )

    t_inicio = time.time()
    model.learn(total_timesteps=100_000)
    duracao = (time.time() - t_inicio) / 60.0

    pasta_modelos = os.path.join(DIR_SIMULADOR, "modelos")
    caminho_v10 = os.path.join(pasta_modelos, "ppo_pcr_v10_inercia")
    model.save(caminho_v10)

    print(f"\n✅ RE-TREINO CONCLUÍDO COM SUCESSO EM {duracao:.2f} MINUTOS!")


if __name__ == "__main__":
    main()