import serial
import time
import csv
import datetime
import threading

# Configurações da Serial
PORTA = 'COM3'
BAUDRATE = 115200

print(f"🔌 Conectando ao ESP32 na porta {PORTA} ({BAUDRATE} baud)...")
try:
    ser = serial.Serial(PORTA, BAUDRATE, timeout=1)
    time.sleep(2.5)
    ser.reset_input_buffer()
    print("✅ Conectado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao conectar na porta {PORTA}: {e}")
    exit(1)

# Variáveis Globais do Controle
acao_atual = 0.000
rodando = True
temp_atual = 25.0

# Log de dados
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
arquivo_log = f"log_controle_manual_{timestamp}.csv"

def thread_leitura_e_envio():
    """Thread que mantém a comunicação Serial constante com o ESP32 a 1 Hz."""
    global temp_atual, acao_atual, rodando
    
    with open(arquivo_log, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["passo", "horario", "temp_real", "acao_manual"])
        passo = 0
        
        while rodando:
            passo += 1
            horario = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Envia o comando atual para a Ponte H / ESP32
            try:
                comando = f"{acao_atual:.3f}\n"
                ser.write(comando.encode('utf-8'))
            except Exception as e:
                print(f"\n❌ Erro no envio Serial: {e}")
            
            # Lê o sensor
            try:
                if ser.in_waiting > 0:
                    dados = ser.read_all().decode('utf-8', errors='ignore').strip().split('\n')
                    for linha in reversed(dados):
                        try:
                            val = float(linha.strip())
                            if -50.0 < val < 105.0:
                                temp_atual = val
                                break
                        except ValueError:
                            continue
            except Exception:
                pass
            
            # Salva no log CSV
            writer.writerow([passo, horario, temp_atual, acao_atual])
            f.flush()
            
            # Atualização no console
            print(f"\r[{horario}] Passo {passo:04d} | Temp DS18B20: {temp_atual:6.2f}°C | Potência Aplicada: {acao_atual:6.3f}   ", end="")
            time.sleep(0.95)

# Inicia a Thread de comunicação em segundo plano
t = threading.Thread(target=thread_leitura_e_envio, daemon=True)
t.start()

# --- INTERFACE DE COMANDOS MANUAL VIA CONSOLE ---
time.sleep(1)
print("\n" + "="*60)
print("🎛️  PAINEL DE CONTROLE MANUAL DA BANCADA")
print("="*60)
print("Comandos Rápidos:")
print("  [e]  -> Esquentar Máximo (+1.000)")
print("  [f]  -> Esfriar Máximo   (-1.000)")
print("  [0]  -> Desligar Potência ( 0.000)")
print("  [h]  -> Holding Power     (+0.150)")
print("  [s]  -> Sair e Desligar")
print("Ou digite qualquer valor numérico entre -1.0 e 1.0 (Ex: 0.5, -0.3)")
print("="*60 + "\n")

try:
    while rodando:
        entrada = input().strip().lower()
        
        if entrada == 'e':
            acao_atual = 1.000
            print("\n🔥 >>> Comando aplicado: AQUECIMENTO MÁXIMO (+1.000)")
        elif entrada == 'f':
            acao_atual = -1.000
            print("\n❄️ >>> Comando aplicado: RESFRIAMENTO MÁXIMO (-1.000)")
        elif entrada == '0':
            acao_atual = 0.000
            print("\n🛑 >>> Comando aplicado: POTÊNCIA DESLIGADA (0.000)")
        elif entrada == 'h':
            acao_atual = 0.150
            print("\n⚖️ >>> Comando aplicado: HOLDING POWER (+0.150)")
        elif entrada == 's':
            print("\n🔌 Encerrando sistema...")
            acao_atual = 0.000
            rodando = False
            break
        else:
            try:
                val = float(entrada)
                if -1.0 <= val <= 1.0:
                    acao_atual = val
                    print(f"\n⚙️ >>> Comando customizado aplicado: {acao_atual:+.3f}")
                else:
                    print("\n⚠️ Digite um valor entre -1.0 e 1.0!")
            except ValueError:
                print("\n⚠️ Comando inválido! Use 'e', 'f', '0', 'h', 's' ou um número entre -1.0 e 1.0.")

finally:
    rodando = False
    time.sleep(0.5)
    try:
        ser.write(b"0.000\n")
        ser.close()
    except Exception:
        pass
    print("✅ Sistema desligado com segurança e log salvo.")