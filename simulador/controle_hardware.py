import gymnasium as gym
from gymnasium import spaces
import numpy as np
import serial
import time
import warnings
import csv
import sys
import datetime
from collections import deque
from stable_baselines3 import PPO

# Importa a sua função de perfil térmico do PCR
from setpoint import gerar_setpoint_pcr

warnings.filterwarnings("ignore")

# Limite de segurança para não derreter a Peltier
MAX_SAFE_TEMP = 105.0 

class TermocicladorHardwareEnv(gym.Env):
    """
    Ambiente Gymnasium para controle do Hardware Físico do Termociclador.
    Substitui a planta simulada pela comunicação serial via ESP32/Arduino.
    """
    def __init__(self, port='COM3', baudrate=115200):
        super(TermocicladorHardwareEnv, self).__init__()
        
        # Conexão Serial (Ajuste 'COM3' para a porta do seu Arduino)
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        
        # Ações contínuas de -1.0 a 1.0 (PWM)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observação idêntica ao ambiente simulado
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)
        
        self._setpoints = gerar_setpoint_pcr()
        self._current_step = 0
        self.temp_history = deque(maxlen=3)

    def normalizar(self, valor):
        """Aplica a mesma normalização (÷100) do treinamento original"""
        return valor / 100.0

    def _get_obs(self, temp, sp):
        t0 = temp
        t1 = self.temp_history[-2] if len(self.temp_history) > 1 else t0
        t2 = self.temp_history[-3] if len(self.temp_history) > 2 else t1
        
        erro = sp - t0

        return np.array([
            self.normalizar(t0),
            self.normalizar(t1),
            self.normalizar(t2),
            self.normalizar(erro),
            self.normalizar(sp)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self.temp_history.clear()
        
        # Desliga a energia por segurança no início
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
                
        for _ in range(3):
            self.temp_history.append(temp_real)
            
        sp = self._setpoints[self._current_step]
        return self._get_obs(temp_real, sp), {}

    def verificar_seguranca(self, temp):
        if temp > MAX_SAFE_TEMP:
            self.ser.write(b"0\n")
            self.ser.close()
            print(f"\n[!] EMERGÊNCIA: Temperatura limite excedida ({temp}°C).")
            sys.exit(1)

    def step(self, action):
        u = float(action) 
        
        # Converte a ação -1 a 1 para comando Serial "Hxxx" ou "Cxxx"
        if u >= 0:
            pwm = int(u * 255)
            comando = f"H{pwm}\n" 
        else:
            pwm = int(abs(u) * 255)
            comando = f"C{pwm}\n" 
            
        self.ser.write(comando.encode())
        self.ser.reset_input_buffer()
        
        try:
            raw_line = self.ser.readline()
            line = raw_line.decode('utf-8', errors='ignore').strip()
            temp_real = float(line)
        except (UnicodeDecodeError, ValueError):
            temp_real = self.temp_history[-1] if len(self.temp_history) > 0 else 25.0
            
        self.temp_history.append(temp_real)
        self.verificar_seguranca(temp_real)
            
        sp = self._setpoints[self._current_step]
        obs = self._get_obs(temp_real, sp)
        reward = -abs(sp - temp_real)
            
        self._current_step += 1
        truncated = self._current_step >= len(self._setpoints)
            
        return obs, reward, False, truncated, {}


# ====================================================
# BLOCO DE EXECUÇÃO (Para quando a IA estiver treinada)
# ====================================================
if __name__ == "__main__":
    env = TermocicladorHardwareEnv(port='COM3') # Mude para a porta do seu Arduino
    log_filename = datetime.datetime.now().strftime("log_hardware_%Y-%m-%d_%H-%M-%S.csv")
    
    try:
        # ATENÇÃO: Carregaremos o NOVO modelo contínuo aqui no futuro
        model = PPO.load("ppo_continuo_hardware", env=env)
        print(f"--- IA CARREGADA | SALVANDO LOG EM: {log_filename} ---")
    except Exception as e:
        print(f"\n[AVISO] Modelo PPO contínuo não encontrado: {e}")
        print("Faça o treinamento do novo modelo antes de rodar o hardware!")
        sys.exit(1)
            
    obs, info = env.reset()
    
    with open(log_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["passo", "temp_real", "setpoint", "acao", "recompensa"])

        try:
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                temp_atual = env.temp_history[-1]
                setpoint_atual = env._setpoints[env._current_step - 1]
                
                writer.writerow([env._current_step, temp_atual, setpoint_atual, action, reward])
                f.flush() 
                
                print(f"Passo: {env._current_step:<4} | Real: {temp_atual:>5.2f}°C | Alvo: {setpoint_atual:>5.1f}°C | Ação: {float(action):>5.2f}")
                
                if truncated:
                    print("\n--- CICLO DE PCR FINALIZADO COM SUCESSO ---")
                    break
                
                time.sleep(1) # Aguarda 1 segundo (ciclo da planta)
                
        except KeyboardInterrupt:
            print("\nExecução interrompida pelo usuário.")
        finally:
            env.ser.write(b"0\n")
            env.ser.close()
            print("Peltier desligada com segurança.")