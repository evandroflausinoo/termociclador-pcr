import os
import sys
import time
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR_ATUAL)

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr


class TermocicladorFineTuningEnv(gym.Env):
    """
    Ambiente v9 — Ajuste Fino com Foco em Zero Overshoot e Pouso Suave.
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # 6 observações: [T_k, T_k-1, T_k-2, Erro, Target, Velocidade_dT]
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(6,), dtype=np.float32)

        self.params = ThermalParams(Tamb=25.0, alpha=0.0035, beta=0.48, dt=1.0, noise_std=0.05)
        self.model = ThermalModel(self.params)

        self.memoria_ia = deque(maxlen=3)
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
        for _ in range(3):
            self.memoria_ia.append(temp_init)

        return self._get_obs(self.setpoints[0]), {}

    def step(self, action):
        u_apl = float(np.clip(action[0], -1.0, 1.0))
        target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]

        nova_temp = self.model.step(u_apl)
        self.memoria_ia.append(nova_temp)

        erro = target - nova_temp  # Positivo se abaixo da meta, negativo se acima
        abs_erro = abs(erro)

        # --- REWARD SHAPING DE ALTA PRECISÃO ---
        # 1. Penalidade base por erro
        reward = -abs_erro

        # 2. Penalidade pesada (quadrática) para OVERSHOOT / UNDERSHOOT
        if (target == 55.0 and nova_temp < 54.5) or (target in [72.0, 95.0] and nova_temp > target + 0.3):
            reward -= (abs_erro ** 2) * 2.0

        # 3. Bônus por travamento perfeito no setpoint (±0.3°C)
        if abs_erro <= 0.3:
            reward += 2.0

        self.passo_atual += 1
        terminated = self.passo_atual >= self.max_passos

        prox_target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]
        obs = self._get_obs(prox_target)

        return obs, reward, terminated, False, {"temp_real": nova_temp, "u_apl": u_apl}

    def _get_obs(self, target: float) -> np.ndarray:
        historico = list(self.memoria_ia)
        t_atual = historico[-1]
        t_anterior = historico[-2]
        velocidade = (t_atual - t_anterior)  # Derivativa / Velocidade de variação

        return np.array([
            t_atual / 100.0,
            t_anterior / 100.0,
            historico[-3] / 100.0,
            (target - t_atual) / 50.0,
            target / 100.0,
            velocidade / 5.0  # Velocidade normalizada
        ], dtype=np.float32)


def main():
    caminho_v8 = os.path.join("modelos", "ppo_pcr_v8_final.zip")
    
    if not os.path.exists(caminho_v8):
        print(f"❌ Modelo base {caminho_v8} não encontrado!")
        return

    print(f"\n{'='*65}")
    print(f"🎯 INICIANDO AJUSTE FINO (FINE-TUNING) — PPO v9 (ZERO OVERSHOOT)")
    print(f"💡 Carregando pesos do PPO v8 e ajustando política por 50.000 passos")
    print(f"{'='*65}\n")

    env = TermocicladorFineTuningEnv()

    # Carrega o modelo v8 com Taxa de Aprendizado menor (1e-4) para ajuste fino
    model = PPO.load(
        caminho_v8,
        env=env,
        learning_rate=1e-4,  # Taxa menor para não desaprender o que já sabe
        custom_objects={"observation_space": env.observation_space}
    )

    t_inicio = time.time()
    # Apenas 50.000 passos (leva entre 3 e 7 minutos)
    model.learn(total_timesteps=50_000)
    duracao = (time.time() - t_inicio) / 60.0

    caminho_v9 = os.path.join("modelos", "ppo_pcr_v9_finetuned")
    model.save(caminho_v9)

    print(f"\n{'='*65}")
    print(f"✅ AJUSTE FINO CONCLUÍDO COM SUCESSO!")
    print(f"⏱️  Tempo total: {duracao:.2f} minutos")
    print(f"💾 Novo Modelo Salvo em: {caminho_v9}.zip")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()