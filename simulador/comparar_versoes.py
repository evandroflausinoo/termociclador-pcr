import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# ==============================================================================
# 🎯 REUTILIZÁVEL: BASTA MUDAR AS PASTAS E NOMES PARA OS PRÓXIMOS TESTES!
# ==============================================================================
PASTA_VERSAO_A = "logs/v4"
NOME_VERSAO_A  = "PPO v4 (Baseline)"

PASTA_VERSAO_B = "logs/v5"
NOME_VERSAO_B  = "PPO v5 (Corrigido)"
# ==============================================================================


def carregar_e_agrupar(pasta):
    """Lê todos os CSVs de uma pasta e calcula média e desvio padrão por passo."""
    arquivos = glob.glob(os.path.join(pasta, "*.csv"))
    if not arquivos:
        print(f"❌ Erro: Nenhum arquivo .csv encontrado em '{pasta}'")
        return None, 0

    dfs = [pd.read_csv(f) for f in arquivos]
    df_concat = pd.concat(dfs)

    # Agrupa por passo para obter média e desvio padrão temporal
    df_agrupado = df_concat.groupby('passo').agg(
        setpoint=('setpoint', 'first'),
        temp_media=('temp_real', 'mean'),
        temp_std=('temp_real', 'std')
    ).reset_index()

    # Cálculo do MAE nos patamares
    df_agrupado['erro_abs'] = (df_agrupado['temp_media'] - df_agrupado['setpoint']).abs()
    df_agrupado['block'] = (df_agrupado['setpoint'] != df_agrupado['setpoint'].shift()).cumsum()
    patamares = df_agrupado.groupby('block').filter(lambda x: len(x) > 20)
    patamares_fim = patamares.groupby('block').apply(lambda x: x.tail(int(len(x) * 0.5))).reset_index(drop=True)
    mae_patamar = patamares_fim['erro_abs'].mean() if not patamares_fim.empty else df_agrupado['erro_abs'].mean()

    return df_agrupado, mae_patamar, len(arquivos)


def gerar_comparativo():
    df_a, mae_a, qtd_a = carregar_e_agrupar(PASTA_VERSAO_A)
    df_b, mae_b, qtd_b = carregar_e_agrupar(PASTA_VERSAO_B)

    if df_a is None or df_b is None:
        return

    plt.figure(figsize=(12, 6), dpi=100)

    # 1. Setpoint (Linha de Meta)
    plt.plot(df_a['passo'], df_a['setpoint'], color='#2b2b2b', linestyle='--', 
             linewidth=2.0, label='Setpoint (Desejado)', zorder=2)

    # 2. Versão A (Vermelho)
    label_a = f"{NOME_VERSAO_A} ({qtd_a} testes) — MAE Patamar = {mae_a:.2f}°C"
    plt.plot(df_a['passo'], df_a['temp_media'], color='#d62728', linestyle='-', 
             linewidth=2.0, label=label_a, zorder=3)
    plt.fill_between(df_a['passo'], df_a['temp_media'] - df_a['temp_std'], 
                     df_a['temp_media'] + df_a['temp_std'], color='#d62728', alpha=0.15)

    # 3. Versão B (Azul)
    label_b = f"{NOME_VERSAO_B} ({qtd_b} testes) — MAE Patamar = {mae_b:.2f}°C"
    plt.plot(df_b['passo'], df_b['temp_media'], color='#1f77b4', linestyle='-', 
             linewidth=2.2, label=label_b, zorder=4)
    plt.fill_between(df_b['passo'], df_b['temp_media'] - df_b['temp_std'], 
                     df_b['temp_media'] + df_b['temp_std'], color='#1f77b4', alpha=0.20)

    # Estilização
    plt.title(f"Comparativo de Controle Térmico em Bancada Física: {NOME_VERSAO_A} vs {NOME_VERSAO_B}", 
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Tempo / Passo de Controle (s)', fontsize=11, fontweight='semibold')
    plt.ylabel('Temperatura (°C)', fontsize=11, fontweight='semibold')
    plt.ylim(20, 105)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.tight_layout()

    # Salva na pasta raiz ou logs
    nome_arquivo_saida = f"grafico_COMPARATIVO_{NOME_VERSAO_A.split()[1]}_vs_{NOME_VERSAO_B.split()[1]}.png"
    plt.savefig(nome_arquivo_saida, dpi=300, bbox_inches='tight')
    print(f"\n🌟 GRÁFICO COMPARATIVO GERADO COM SUCESSO: {nome_arquivo_saida}")
    plt.show()

if __name__ == "__main__":
    gerar_comparativo()