import sys
import os
import time
import serial
import pandas as pd
import numpy as np
import gymnasium as gym

from datetime import datetime
from gymnasium import spaces
from collections import deque
from stable_baselines3 import PPO

# Garantir que os arquivos locais sejam encontrados
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIRETORIO_SCRIPT)

VERSAO_MODELO = "v7"
PORTA_SERIAL = "COM3"  # Ajuste se a sua porta for outra (ex: COM4)
BAUD_RATE = 115200
MAX_SAFE_TEMP = 105.0


def gerar_setpoint_pcr_rapido():
    """
    Perfil de Teste Rápido de 240 segundos (4 Minutos).
    Mede a resposta em todos os 3 patamares do PCR em tempo recorde.
    """
    setpoints = []
    setpoints.extend([25.0] * 20)  # 20s em 25°C (Estabilização inicial)
    setpoints.extend([95.0] * 60)  # 60s em 95°C (Desnaturação)
    setpoints.extend([55.0] * 80)  # 80s em 55°C (Resfriamento Ativo e Anelamento)
    setpoints.extend([72.0] * 80)  # 80s em 72°C (Frenagem e Extensão)
    return setpoints


class RealTermocicladorEnvRapido(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)

        self._setpoints = gerar_setpoint_pcr_rapido()
        self.max_steps = len(self._setpoints)  # Exactly 240 steps
        self._current_step = 0

        self.temp_history = deque(maxlen=10)
        self.last_u_cmd = 0.0

        # Histórico de registros para salvar o CSV
        self.log_data = []

        print(f"🔌 Conectando ao ESP32 na porta {PORTA_SERIAL} ({BAUD_RATE} baud)...")
        try:
            self.ser = serial.Serial(PORTA_SERIAL, BAUD_RATE, timeout=1.0)
            time.sleep(2.0)  # Aguarda reset do ESP32
            print("✅ Conexão Serial estabelecida com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao abrir porta serial {PORTA_SERIAL}: {e}")
            sys.exit(1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self.temp_history.clear()
        self.log_data.clear()

        # Faz leitura inicial do sensor
        temp_init = self._ler_sensor_serial()
        for _ in range(3):
            self.temp_history.append(temp_init)

        target = self._setpoints[0]
        return self._get_obs(target), {}

    def step(self, action):
        self._current_step += 1

        u_raw = float(action[0])
        u_raw = np.clip(u_raw, -1.0, 1.0)

        idx = min(self._current_step, len(self._setpoints) - 1)
        target = self._setpoints[idx]
        temp_atual = self.temp_history[-1]

        erro = target - temp_atual

        # ============================================================
        # 🛡️ SAFETY SHIELD INTELIGENTE (FRENAGEM + RESFRIAMENTO ATIVO)
        # ============================================================
        # 1. Se estiver MUITO ACIMA do alvo (descida), LIGA RESFRIAMENTO ATIVO!
        if erro < -1.5:
            u_cmd = -1.000  # Ponte H inverte polaridade (Peltier refrigera)

        # 2. Frenagem Antecipatória: A 1.5°C da meta subindo, CORTA O CALOR!
        elif erro <= 1.5 and u_raw > 0.0:
            u_cmd = 0.000   # Inércia térmica do alumínio termina a subida

        # 3. Caso contrário, segue o sinal normal
        else:
            u_cmd = u_raw

        self.last_u_cmd = u_cmd

        # Envia comando para a placa
        self._enviar_comando_serial(u_cmd)
        time.sleep(0.95)

        # Lê resposta do sensor
        temp_real = self._ler_sensor_serial()

        # Emergência
        if temp_real >= MAX_SAFE_TEMP:
            print(f"\n🚨 EMERGÊNCIA TÉRMICA: Temp de {temp_real}°C atingiu {MAX_SAFE_TEMP}°C!")
            self._enviar_comando_serial(0.0)
            self.ser.close()
            sys.exit(1)

        self.temp_history.append(temp_real)

        erro_sinal = target - temp_real
        reward = -abs(erro_sinal)

        hora_atual = datetime.now().strftime("%H:%M:%S")

        # Guarda linha do registro
        self.log_data.append({
            "passo": self._current_step,
            "horario": hora_atual,
            "temp_real": round(temp_real, 2),
            "setpoint": round(target, 1),
            "acao": round(u_cmd, 3),
            "recompensa": round(reward, 2)
        })

        terminated = False
        truncated = self._current_step >= self.max_steps

        print(
            f"Passo: {self._current_step:03d}/240 | Horário: {hora_atual} | "
            f"Temp: {temp_real:6.2f}°C | Setpoint: {target:5.1f}°C | "
            f"Ação: {u_cmd:6.3f}"
        )

        return self._get_obs(target), reward, terminated, truncated, {}

    def _get_obs(self, target: float) -> np.ndarray:
        t_atual = self.temp_history[-1]
        t_prev1 = self.temp_history[-2] if len(self.temp_history) > 1 else t_atual
        t_prev2 = self.temp_history[-3] if len(self.temp_history) > 2 else t_prev1

        return np.array([
            t_atual / 100.0,
            t_prev1 / 100.0,
            t_prev2 / 100.0,
            (target - t_atual) / 50.0,
            target / 100.0,
        ], dtype=np.float32)

    def _enviar_comando_serial(self, u_cmd: float):
        try:
            msg = f"{u_cmd:.3f}\n"
            self.ser.write(msg.encode("utf-8"))
        except Exception as e:
            print(f"\n❌ Erro ao enviar serial: {e}")

    def _ler_sensor_serial(self) -> float:
        try:
            while self.ser.in_waiting:
                linha = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if linha.startswith("TEMP:"):
                    val_str = linha.replace("TEMP:", "").strip()
                    return float(val_str)
        except Exception:
            pass
        return self.temp_history[-1] if len(self.temp_history) > 0 else 25.0

    def salvar_log_csv(self):
        if not self.log_data:
            print("\n⚠️ Nenhum dado gravado para salvar.")
            return

        pasta_logs = os.path.join(DIRETORIO_SCRIPT, "logs", VERSAO_MODELO)
        os.makedirs(pasta_logs, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nome_arquivo = f"log_pcr_{VERSAO_MODELO}_rapido_{timestamp}_{len(self.log_data)}passos.csv"
        caminho_completo = os.path.join(pasta_logs, nome_arquivo)

        resp = input(f"\n💾 Deseja salvar o log deste teste rápido ({len(self.log_data)} passos) em logs/{VERSAO_MODELO}/? (s/n): ").strip().lower()
        if resp == 's':
            df = pd.DataFrame(self.log_data)
            df.to_csv(caminho_completo, index=False)
            print(f"✅ Log salvo com sucesso em: {caminho_completo}")
        else:
            print("🗑️ Log descartado com sucesso.")

    def close(self):
        print("\n🔌 Desligando sistema com segurança...")
        self._enviar_comando_serial(0.0)
        if hasattr(self, "ser") and self.ser.is_open:
            self.ser.close()
        self.salvar_log_csv()


def main():
    caminho_modelo = os.path.join(DIRETORIO_SCRIPT, "modelos", f"ppo_pcr_{VERSAO_MODELO}_final.zip")

    if not os.path.exists(caminho_modelo):
        print(f"❌ Modelo não encontrado em: {caminho_modelo}")
        sys.exit(1)

    print(f"\n✅ CARREGANDO MODELO PPO {VERSAO_MODELO}...")
    model = PPO.load(caminho_modelo)

    env = RealTermocicladorEnvRapido()
    obs, _ = env.reset()

    print("\n🚀 INICIANDO TESTE RÁPIDO DE 4 MINUTOS (240 PASSOS)...")
    print("Pressione Ctrl+C para interromper com segurança.\n")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    except KeyboardInterrupt:
        print("\n⚠️ Interrupção manual detectada.")
    finally:
        env.close()


if __name__ == "__main__":
    main()