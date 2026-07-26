import serial
import time
import csv
import datetime

# Conectando à porta Serial do ESP32
PORTA = 'COM3'
BAUDRATE = 115200

print(f"🔌 Conectando ao ESP32 na porta {PORTA}...")
ser = serial.Serial(PORTA, BAUDRATE, timeout=1)
time.sleep(2.5)
ser.reset_input_buffer()

def ler_temp():
    """Lê a leitura de temperatura mais recente enviada pelo ESP32."""
    try:
        if ser.in_waiting > 0:
            dados = ser.read_all().decode('utf-8', errors='ignore').strip().split('\n')
            for linha in reversed(dados):
                try:
                    val = float(linha.strip())
                    if -50.0 < val < 105.0:
                        return val
                except ValueError:
                    continue
    except Exception:
        pass
    return None

# Nome do arquivo de log do ensaio físico
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
arquivo_csv = f"ensaio_degrau_fisico_{timestamp}.csv"

print("\n🔬 INICIANDO ENSAIO DE IDENTIFICAÇÃO DE SISTEMA")
print("1. Leitura de Base Térmica (10s a 0%)")
print("2. DEGRAU DE AQUECIMENTO (100% de Potência por 30s)")
print("3. RESPOSTA LIVRE / INÉRCIA (0% de Potência por 60s)")
print("--------------------------------------------------")

# Estrutura do Teste: (Ação, Duração em Segundos)
protocolo = [
    (0.000, 10),   # 10s parado para estabilizar leitura inicial
    (1.000, 30),   # 30s de Aquecimento Máximo (+100%)
    (0.000, 60),   # 60s em Potência Zero para medir a Inércia Térmica de Subida
]

passo = 0
try:
    with open(arquivo_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["passo", "horario", "temp_real", "acao"])

        temp_atual = ler_temp() or 25.0

        for acao, duracao in protocolo:
            print(f"\n⚡ MUDANÇA DE ESTADO: Aplicando Potência {acao:.1f} por {duracao}s...")
            inicio_fase = time.time()
            
            while time.time() - inicio_fase < duracao:
                passo += 1
                
                # Envia comando direto para a Ponte H
                ser.write(f"{acao:.3f}\n".encode('utf-8'))
                time.sleep(0.95)
                
                temp_lida = ler_temp()
                if temp_lida is not None:
                    temp_atual = temp_lida
                
                horario = datetime.datetime.now().strftime("%H:%M:%S")
                writer.writerow([passo, horario, temp_atual, acao])
                f.flush()
                
                print(f"Passo: {passo:03d} | Temp: {temp_atual:6.2f}°C | Ação Aplicada: {acao:6.3f}")

except KeyboardInterrupt:
    print("\n⚠️ Teste interrompido pelo usuário.")

finally:
    ser.write(b"0\n")
    ser.close()
    print(f"\n✅ Ensaio finalizado! Dados salvos em: {arquivo_csv}")