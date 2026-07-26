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

MAX_SAFE_TEMP = 100.0  # Trava limite de emergência da Peltier


class TermocicladorHardwareEnv(gym.Env):
    """
    Ambiente de Hardware do Termociclador para execução em bancada real.
    Gerencia comunicação Serial com o ESP32 e aplica o Safety Shield v10.
    """
    def __init__(self, port='COM3', baudrate=115200):
        super(TermocicladorHardwareEnv, self).__init__()

        print(f"🔌 Conectando ao ESP32 na porta {port} ({baudrate} baud)...")
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2.5)  # Aguarda boot do ESP32
        self.ser.reset_input_buffer()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # 6 variáveis: [T_k, T_k-1, T_k-2, Erro, Target, Velocidade_dT]
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(6,), dtype=np.float32
        )
        
        self.memoria_ia = deque(maxlen=4)

        self._setpoints = gerar_setpoint_pcr(
            ciclos=2,
            t_estabilizacao=20,
            t_desnaturacao=120,
            t_anelamento=180,
            t_extensao=120
        )
        self.max_steps = len(self._setpoints)
        self._current_step = 0
        self.last_u_cmd = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._current_step = 0
        self.last_u_cmd = 0.0

        # 1. Lê a temperatura inicial do sensor pela serial
        temp_inicial = self._ler_sensor_serial()

        # 2. Preenche o histórico de 4 posições (necessário para calcular a velocidade no v10)
        self.memoria_ia = deque([temp_inicial, temp_inicial, temp_inicial, temp_inicial], maxlen=4)

        # 3. Define o primeiro setpoint
        target = self._setpoints[0]
        
        return self._get_obs(target), {}

    def step(self, action):
        self._current_step += 1

        # Clip do sinal de controle enviado pela IA (-1.0 a +1.0)
        u_raw = float(np.clip(action[0], -1.0, 1.0))

        # Identifica o setpoint do passo atual
        idx = min(self._current_step, len(self._setpoints) - 1)
        target = self._setpoints[idx]
        temp_atual = self.memoria_ia[-1]

        # ============================================================
        # 🛡️ SAFETY SHIELD v10 — PROTEÇÃO EXTREMA DE HARDWARE
        # ============================================================
        u_cmd = u_raw  # A IA v10 tem prioridade total de decisão

        # 1. Trava de Segurança Térmica Absoluta (Evita queimar a Peltier)
        if temp_atual >= 98.0 and u_raw > 0.0:
            u_cmd = -0.500
            print("⚠️ SAFETY SHIELD: Limite térmico de emergência (>98°C)!")

        # 2. Proteção da Ponte H / Fonte (Evita trancos elétricos instantâneos)
        delta_u = u_cmd - self.last_u_cmd
        if abs(delta_u) > 1.5:
            u_cmd = self.last_u_cmd + np.sign(delta_u) * 1.5

        self.last_u_cmd = u_cmd
        # ============================================================

        # --- APLICAÇÃO DO COMANDO NO HARDWARE ---
        self._enviar_comando_serial(u_cmd)
        time.sleep(0.95)
        temp_real = self._ler_sensor_serial()

        # Atualiza a memória de temperaturas
        self.memoria_ia.append(temp_real)

        # Cálculo de erro para o log
        erro = target - temp_real

        # Verifica se o ciclo de PCR chegou ao fim
        terminated = self._current_step >= len(self._setpoints) - 1
        truncated = False

        # Prepara a observação de 6 variáveis para o próximo passo
        obs = self._get_obs(target)

        info = {
            "temp_real": temp_real,
            "u_apl": u_cmd,
            "erro": erro
        }

        return obs, -abs(erro), terminated, truncated, info

    def _get_obs(self, target: float) -> np.ndarray:
        historico = list(self.memoria_ia)
        t_atual = historico[-1]
        t_anterior = historico[-2]

        # 🚀 6ª VARIÁVEL: Velocidade térmica (°C/s) para anteceder a inércia do alumínio
        velocidade_dT = t_atual - t_anterior

        return np.array([
            t_atual / 100.0,
            t_anterior / 100.0,
            historico[-3] / 100.0,
            (target - t_atual) / 50.0,
            target / 100.0,
            velocidade_dT / 2.0
        ], dtype=np.float32)

    def _enviar_comando_serial(self, u_cmd: float):
        try:
            comando = f"{u_cmd:.3f}\n"
            self.ser.write(comando.encode('utf-8'))
        except Exception as e:
            print(f"❌ Erro de escrita na Serial: {e}")

    def _ler_sensor_serial(self) -> float:
        try:
            if self.ser.in_waiting > 0:
                dados_brutos = self.ser.read_all().decode('utf-8', errors='ignore').strip()
                linhas = dados_brutos.split('\n')
                for linha in reversed(linhas):
                    linha_limpa = linha.strip()
                    if linha_limpa:
                        try:
                            val = float(linha_limpa)
                            if -50.0 < val < 105.0:
                                return val
                        except ValueError:
                            continue
        except Exception:
            pass
        return list(self.temp_history)[-1]


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == "__main__":

    VERSAO_MODELO = "v10"

    DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
    PASTA_LOGS = os.path.join(DIRETORIO_SCRIPT, "logs", VERSAO_MODELO)
    os.makedirs(PASTA_LOGS, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_temp = os.path.join(PASTA_LOGS, f"temp_{timestamp}.csv")

    env = TermocicladorHardwareEnv(port='COM3')

    # AQUI ESTÁ O AJUSTE PRINCIPAL:
    # O arquivo salvo pelo script de fine-tuning foi "ppo_pcr_v9_finetuned.zip"
    VERSAO_MODELO = "v10"

    # Nome exato do arquivo salvo pelo script do v10
    nome_arquivo_modelo = "ppo_pcr_v10_inercia.zip"
    caminho_modelo = os.path.join(DIRETORIO_SCRIPT, "modelos", nome_arquivo_modelo)
    
    if not os.path.exists(caminho_modelo):
        caminho_modelo = os.path.join(os.path.dirname(DIRETORIO_SCRIPT), "modelos", nome_arquivo_modelo)
    try:
        model = PPO.load(caminho_modelo, env=env)
        print(f"\n✅ MODELO PPO {VERSAO_MODELO} CARREGADO COM SUCESSO DE: {caminho_modelo}")
    except Exception as e:
        print(f"\n❌ ERRO AO CARREGAR O MODELO {VERSAO_MODELO}: {e}")
        sys.exit(1)

    obs, info = env.reset()

    print(f"\n🚀 INICIANDO ENSAIO EM HARDWARE REAL (PPO {VERSAO_MODELO} - SAFETY SHIELD v7.6)")
    print(f"Pressione Ctrl+C a qualquer momento para interromper com segurança.\n")

    try:
        with open(log_temp, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["passo", "horario", "temp_real", "setpoint", "acao", "recompensa"])

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                temp_atual = env.memoria_ia[-1]
                idx_sp = min(env._current_step, len(env._setpoints) - 1)
                setpoint_atual = env._setpoints[idx_sp]
                u_cmd = env.last_u_cmd
                horario = datetime.datetime.now().strftime("%H:%M:%S")

                writer.writerow([env._current_step, horario, round(temp_atual, 2), round(setpoint_atual, 2), round(u_cmd, 3), round(reward, 2)])
                f.flush()

                print(f"Passo: {env._current_step:03d} | Horário: {horario} | Temp: {temp_atual:6.2f}°C | Setpoint: {setpoint_atual:5.1f}°C | Ação: {u_cmd:6.3f}")

                if terminated or truncated:
                    print("\n🏁 Ensaio de PCR concluído com sucesso!")
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupção manual detectada (Ctrl+C).")

    finally:
        try:
            env.ser.write(b"0\n")
            env.ser.close()
        except Exception:
            pass

        print("🔌 Sistema desligado com segurança.")

        if os.path.exists(log_temp):
            nome_final = f"log_pcr_{VERSAO_MODELO}_{timestamp}_{env._current_step}passos.csv"
            log_final = os.path.join(PASTA_LOGS, nome_final)

            print("\n" + "="*50)
            resposta = input(f"Deseja salvar o log deste teste ({env._current_step} passos) em logs/{VERSAO_MODELO}/? (s/n): ").strip().lower()
            print("="*50)

            if resposta in ['s', 'sim', 'y', 'yes']:
                try:
                    os.rename(log_temp, log_final)
                    print(f"✅ Log salvo com sucesso em: {log_final}")
                except Exception:
                    time.sleep(0.5)
                    os.rename(log_temp, log_final)
                    print(f"✅ Log salvo com sucesso em: {log_final}")
            else:
                try:
                    os.remove(log_temp)
                    print("🗑️  Log descartado com sucesso.")
                except Exception:
                    pass