import pandas as pd
import matplotlib.pyplot as plt

def gerar_grafico(arquivo_csv):
    try:
        df = pd.read_csv(arquivo_csv)
        plt.figure(figsize=(10, 6))
        plt.plot(df['passo'], df['setpoint'], 'r--', label='Alvo (Setpoint)')
        plt.plot(df['passo'], df['temp_real'], 'b-', label='Real (PPO)')
        plt.title(f"Resultado: {arquivo_csv}")
        plt.legend(); plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Erro: {e}")

# Exemplo de como usar:
gerar_grafico("log_pcr_2026-05-06_21-02-13.csv")