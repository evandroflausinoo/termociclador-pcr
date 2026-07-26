import os
import sys
import time
import sys
import os
import time
import numpy as np
import gymnasium as gym

from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO

# --- REGISTRO DINÂMICO DE CAMINHOS DO PYTHON ---
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))          # Subpasta treinos.py
DIR_SIMULADOR = os.path.abspath(os.path.join(DIR_ATUAL, ".."))   # Pasta pai simulador

sys.path.append(DIR_ATUAL)
sys.path.append(DIR_SIMULADOR)

from modelo_termico import ThermalModel, ThermalParams
from setpoint import gerar_setpoint_pcr


class TermocicladorFineTuningEnv(gym.Env):
    """
    Ambiente v9 — Fine-Tuning com Recompensa Quadrática (100% compatível com PPO v8).
    """

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Mantém 5 observações idênticas ao v8 para permitir o PPO.load()
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)

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

        erro = abs(target - nova_temp)

        # --- REWARD SHAPING RIGOROSO PARA ZERAR OVERSHOOT ---
        reward = -erro

        # Penalidade QUADRÁTICA pesada se houver overshoot/undershoot
        if (target == 55.0 and nova_temp < 54.5) or (target in [72.0, 95.0] and nova_temp > target + 0.3):
            reward -= (erro ** 2) * 3.0

        # Bônus por estabilização na faixa de precisão (±0.3°C)
        if erro <= 0.3:
            reward += 2.0

        self.passo_atual += 1
        terminated = self.passo_atual >= self.max_passos

        prox_target = self.setpoints[min(self.passo_atual, self.max_passos - 1)]
        obs = self._get_obs(prox_target)

        return obs, reward, terminated, False, {"temp_real": nova_temp, "u_apl": u_apl}

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
    caminho_v8 = os.path.join("modelos", "ppo_pcr_v8_final.zip")
    
    if not os.path.exists(caminho_v8):
        print(f"❌ Modelo base {caminho_v8} não encontrado!")
        return

    print(f"\n{'='*65}")
    print(f"🎯 INICIANDO AJUSTE FINO (FINE-TUNING) — PPO v9 (ZERO OVERSHOOT)")
    print(f"💡 Carregando pesos do PPO v8 e refinando por 50.000 passos")
    print(f"{'='*65}\n")

    env = TermocicladorFineTuningEnv()

    # Carrega os pesos do v8 e ajusta com aprendizado fino (1e-4)
    model = PPO.load(
        caminho_v8,
        env=env,
        learning_rate=1e-4
    )

    t_inicio = time.time()
    model.learn(total_timesteps=50_000)
    duracao = (time.time() - t_inicio) / 60.0

    caminho_v9 = os.path.join("modelos", "ppo_pcr_v9_finetuned")
    model.save(caminho_v9)

    print(f"\n{'='*65}")
    print(f"✅ AJUSTE FINO CONCLUÍDO COM SUCESSO!")
    print(f"⏱️  Tempo total: {duracao:.2f} minutos")
    print(f"💾 Modelo v9 salvo em: {caminho_v9}.zip")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()