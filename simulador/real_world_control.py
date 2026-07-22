import gymnasium as gym
from gymnasium import spaces
import numpy as np
import serial
import time
import warnings
import csv
import sys
import os
import datetime
from collections import deque
from stable_baselines3 import PPO
from setpoint import gerar_setpoint_pcr

warnings.filterwarnings("ignore")

MAX_SAFE_TEMP = 105.0


class TermocicladorHardwareEnv(gym.Env):
    def __init__(self, port='COM3', baudrate=115200):
        super(TermocicladorHardwareEnv, self).__init__()

        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)

        # Ação contínua compatível com PPO v3
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

        self._setpoints = gerar_setpoint_pcr(
            ciclos=2,
            t_estabilizacao=20,
            t_desnaturacao=120,
            t_anelamento=180,
            t_extensao=120,
        )
        self._current_step = 0
        self.temp_history = deque([25.0, 25.0, 25.0], maxlen=3)

    def _get_obs(self, temp, sp):
        historico = list(self.temp_history)
        return np.array([
            historico[-1] / 100.0,
            historico[-2] / 100.0,
            historico[-3] / 100.0,
            (sp - historico[-1]) / 100.0,
            sp / 100.0
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self.temp_history = deque([25.0, 25.0, 25.0], maxlen=3)

        self.ser.write(b"0\n")
        self.ser.reset_input_buffer()

        temp_real = 25.0
        for _ in range(5):
            try:
                raw_line = self.ser.readline()
                linha = raw_line.decode('utf-8', errors='ignore').strip()
                if linha:
                    temp_real = float(linha)
                    break
            except ValueError:
                pass

        self.temp_history = deque([temp_real, temp_real, temp_real], maxlen=3)
        sp = self._setpoints[self._current_step]
        return self._get_obs(temp_real, sp), {}

    def verificar_seguranca(self, temp):
        if temp > MAX_SAFE_TEMP:
            self.ser.write(b"0\n")
            self.ser.close()
            print(f"\n[!] EMERGÊNCIA: {temp}°C — sistema desligado!")
            sys.exit(1)

    def step(self, action):
        u = float(action[0])
        u = np.clip(u, -1.0, 1.0)

        if u > 0.05:                       # Aquecer
            pwm = int(u * 255)
            comando = f"H{pwm}\n"
        elif u < -0.05:                    # Esfriar
            pwm = int(abs(u) * 255)
            comando = f"C{pwm}\n"
        else:                              # Desliga
            comando = "0\n"

        self.ser.write(comando.encode())
        self.ser.reset_input_buffer()

        try:
            raw_line = self.ser.readline()
            line = raw_line.decode('utf-8', errors='ignore').strip()
            temp_real = float(line)
        except (UnicodeDecodeError, ValueError):
            temp_real = self.temp_history[-1]

        self.temp_history.append(temp_real)
        self.verificar_seguranca(temp_real)

        sp = self._setpoints[self._current_step]
        obs = self._get_obs(temp_real, sp)
        reward = -abs(sp - temp_real)

        self._current_step += 1
        truncated = self._current_step >= len(self._setpoints)

        return obs, reward, False, truncated, {}


if __name__ == "__main__":
    log_base = datetime.datetime.now().strftime("log_pcr_%Y-%m-%d_%H-%M-%S")
    log_temp = log_base + "_temp.csv"

    env = TermocicladorHardwareEnv(port='COM3')

    try:
        model = PPO.load("ppo_pcr_v3_final", env=env)
        print("--- IA v3 CARREGADA ---")
    except Exception as e:
        print(f"ERRO AO CARREGAR IA: {e}")
        sys.exit(1)

    obs, info = env.reset()

    # O bloco try envolve a escrita completa
    try:
        with open(log_temp, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["passo", "horario", "temp_real", "setpoint", "acao", "recompensa"])

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                temp_atual = env.temp_history[-1]
                setpoint_atual = env._setpoints[env._current_step - 1]
                acao_str = f"{float(action[0]):.3f}"

                writer.writerow([
                    env._current_step,
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    temp_atual,
                    setpoint_atual,
                    acao_str,
                    reward
                ])
                f.flush()

                print(
                    f"Passo: {env._current_step:<4} | "
                    f"{datetime.datetime.now().strftime('%H:%M:%S')} | "
                    f"Real: {temp_atual:>6.2f}°C | "
                    f"Alvo: {setpoint_atual:>5.1f}°C | "
                    f"Ação: {float(action[0]):>+.3f}"
                )

                if truncated:
                    print("\n--- CICLO DE PCR FINALIZADO ---")
                    break

                time.sleep(1)

    except KeyboardInterrupt:
        print("\nExecução interrupted pelo usuário.")

    finally:
        # Desliga atuadores e fecha porta serial
        try:
            env.ser.write(b"0\n")
            env.ser.close()
        except Exception:
            pass

        # RENOMEAR APÓS FECHAR O ARQUIVO CSV (FORA DO WITH)
        log_final = f"{log_base}_{env._current_step}passos.csv"
        
        if os.path.exists(log_temp):
            try:
                os.rename(log_temp, log_final)
                print(f"Log salvo: {log_final}")
            except PermissionError:
                # Caso o SO retenha o handle por alguns milissegundos
                time.sleep(0.5)
                os.rename(log_temp, log_final)
                print(f"Log salvo: {log_final}")

        print("Sistema desligado com segurança.")