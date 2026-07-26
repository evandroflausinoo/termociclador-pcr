import serial
import time

PORTA = 'COM3'  # Confirme se ainda é a COM3
BAUDRATE = 115200

print(f"Tentando conectar na {PORTA}...")

try:
    ser = serial.Serial(PORTA, BAUDRATE, timeout=2)
    time.sleep(2)
    ser.reset_input_buffer()
    print("✅ Conectado! Lendo dados da porta serial (Pressione Ctrl+C para parar):\n")

    while True:
        # Envia um comando neutro
        ser.write(b"0\n")
        raw_line = ser.readline()
        linha = raw_line.decode('utf-8', errors='ignore').strip()
        
        print(f"Dado Bruto Recebido: '{raw_line}' | Texto Processado: '{linha}'")
        time.sleep(1)

except Exception as e:
    print(f"❌ Erro ao abrir a porta serial: {e}")