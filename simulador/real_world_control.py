import gymnasium as gym
from gymnasium import spaces
import numpy as np
import serial
import time
import warnings
from collections import deque
from stable_baselines3 import PPO
# Importando minha função de gerar a curva de temperatura do PCR
from setpoint import gerar_setpoint_pcr

# Silenciando os avisos do Gym/Numpy pra deixar o terminal limpo
warnings.filterwarnings("ignore")

class RealThermalEnv(gym.Env):
    def __init__(self, port='COM3', baudrate=115200):
        super(RealThermalEnv, self).__init__()
        
        # Conexão com o ESP32
        try:
            self.ser = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2) # Tempo pro ESP32 dar o boot
            print(f"--- HARDWARE CONECTADO: {port} ---")
        except Exception as e:
            print(f"ERRO DE CONEXÃO: {e}")
            exit()

        # Definição dos espaços (0=Parar, 1=Esquentar, 2=Esfriar)
        self.action_space = spaces.Discrete(3)
        
        # O modelo PPO foi treinado com 5 inputs normalizados entre -2 e 2
        # Estrutura: [Temp_t, Temp_t-1, Temp_t-2, Erro, Target]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)
        
        # Buffer de memória para as últimas 3 temperaturas (necessário para o Jitter)
        self.temp_history = deque([25.0, 25.0, 25.0], maxlen=3)

    def normalizar(self, valor):
        # Escalonamento para manter os valores próximos da faixa que a IA conhece
        return (valor / 50.0) - 1.0

    def _get_obs(self, temp_real, sp):
        # Atualiza o histórico e monta o vetor de 5 posições
        self.temp_history.append(temp_real)
        erro = sp - temp_real
        
        obs = np.array([
            self.normalizar(self.temp_history[2]), # Atual
            self.normalizar(self.temp_history[1]), # Anterior
            self.normalizar(self.temp_history[0]), # T-2
            self.normalizar(erro),
            self.normalizar(sp)
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Puxa os setpoints reais do meu arquivo setpoint.py
        self._setpoints = gerar_setpoint_pcr(ciclos=2)
        self._current_step = 0
        
        # Garante que o hardware comece desligado e limpa lixo da Serial
        self.ser.reset_input_buffer()
        self.ser.write(b"0\n") 
        time.sleep(1)
        
        # Leitura inicial do DS18B20
        line = self.ser.readline().decode('utf-8').strip()
        temp_real = float(line) if line else 25.0
        
        # Preenche o histórico inicial com a temperatura atual
        self.temp_history = deque([temp_real, temp_real, temp_real], maxlen=3)
        
        sp = self._setpoints[self._current_step]
        obs = self._get_obs(temp_real, sp)
        return obs, {}

    def step(self, action):
        # Envia a decisão da IA para o hardware
        self.ser.write(f"{action}\n".encode())
        
        # Lê o feedback com tratamento de ruído (ignora bytes inválidos)
        try:
            raw_line = self.ser.readline()
            # errors='ignore' evita o erro de UnicodeDecode que você teve
            line = raw_line.decode('utf-8', errors='ignore').strip()
            temp_real = float(line)
        except (UnicodeDecodeError, ValueError):
            # Se a leitura falhar, mantemos a última temperatura do histórico
            temp_real = self.temp_history[-1]
            # print("Aviso: Ruído na leitura Serial detectado.") # Opcional para debug

        sp = self._setpoints[self._current_step]
        obs = self._get_obs(temp_real, sp)
        
        # Cálculo de recompensa
        erro_bruto = sp - temp_real
        reward = -abs(erro_bruto)
        
        self._current_step += 1
        truncated = self._current_step >= len(self._setpoints)
        terminated = False
        
        return obs, reward, terminated, truncated, {}

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    env = RealThermalEnv(port='COM3')
    
    # Carrega o modelo treinado (deve estar na mesma pasta como .zip)
    try:
        model = PPO.load("ppo_pcr_jitter_final", env=env)
        print("--- MODELO PPO CARREGADO E ATIVO ---")
    except Exception as e:
        print(f"ERRO AO CARREGAR MODELO: {e}")
        exit()

    obs, info = env.reset()
    
    print(f"\nIniciando Ciclos de PCR Reais...")
    print("-" * 75)
    print(f"{'PASSO':<8} | {'TEMP':<8} | {'ALVO':<8} | {'AÇÃO':<6} | {'ERRO':<8}")
    print("-" * 75)

    try:
        while True:
            # A IA decide a ação baseada no estado atual do sistema
            action, _states = model.predict(obs, deterministic=True)
            
            # Aplica a ação e recebe o novo estado do mundo real
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Recupera valores brutos para o log do terminal
            temp_atual = env.temp_history[-1]
            setpoint_atual = env._setpoints[env._current_step - 1]
            erro_atual = setpoint_atual - temp_atual
            
            print(f"{env._current_step:<8} | {temp_atual:>5.2f}°C | {setpoint_atual:>5.1f}°C | {action:<6} | {erro_atual:>7.2f}")
            
            if truncated:
                print("\n--- CICLO DE PCR FINALIZADO COM SUCESSO ---")
                break
                
            time.sleep(1) # Delay de 1Hz entre leituras
            
    except KeyboardInterrupt:
        print("\nInterrompido manualmente. Desligando sistema...")
    finally:
        # Comando de segurança para não deixar a Peltier ligada ao sair
        env.ser.write(b"0\n")
        env.ser.close()