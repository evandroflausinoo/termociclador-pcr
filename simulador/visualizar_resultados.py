import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# ==============================================================================
# 🎯 MUDAR ESTAS DUAS LINHAS PARA OS PRÓXIMOS TESTES:
# ==============================================================================
PASTA_DOS_LOGS = "logs/v8"    
NOME_DA_VERSAO = "PPO v8"      # Ex: mude para "PPO v5" no título dos gráficos
# ==============================================================================


def gerar_tudo(pasta, versao):
    arquivos = glob.glob(os.path.join(pasta, "*.csv"))
    
    if not arquivos:
        print(f"❌ Nenhum arquivo .csv encontrado em '{pasta}'. Verifique a pasta!")
        return

    print(f"🚀 Processando {len(arquivos)} logs de '{versao}' em '{pasta}'...")
    dfs = []

    # 1. Gera os gráficos INDIVIDUAIS de cada .csv da pasta
    for arq in arquivos:
        df = pd.read_csv(arq)
        nome_base = os.path.basename(arq).replace('.csv', '')
        
        df['erro'] = df['temp_real'] - df['setpoint']
        df['erro_abs'] = df['erro'].abs()
        df['block'] = (df['setpoint'] != df['setpoint'].shift()).cumsum()
        
        pat = df.groupby('block').filter(lambda x: len(x) > 20)
        pat_fim = pat.groupby('block').apply(lambda x: x.tail(int(len(x) * 0.5))).reset_index(drop=True)
        
        mae = pat_fim['erro_abs'].mean() if not pat_fim.empty else df['erro_abs'].mean()
        std = pat_fim['erro'].std() if not pat_fim.empty else df['erro'].std()

        plt.figure(figsize=(10, 5), dpi=100)
        plt.plot(df['passo'], df['setpoint'], 'k--', linewidth=1.8, label='Setpoint')
        plt.plot(df['passo'], df['temp_real'], 'r-', linewidth=2.0, label=f'Temp Real (MAE={mae:.2f}°C, σ=±{std:.2f}°C)')
        plt.fill_between(df['passo'], df['setpoint'], df['temp_real'], color='red', alpha=0.12)
        
        plt.title(f"Teste Individual — {versao} ({nome_base})", fontweight='bold')
        plt.xlabel('Tempo (s)'); plt.ylabel('Temperatura (°C)'); plt.ylim(20, 105); plt.grid(True, linestyle=':')
        plt.legend(loc='upper right')
        
        plt.savefig(os.path.join(pasta, f"grafico_{nome_base}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        dfs.append(df)

    # 2. Gera o gráfico CONSOLIDADO (Soma/Média de todos os testes com a sombra da variação)
    df_concat = pd.concat(dfs)
    df_agrup = df_concat.groupby('passo').agg(
        setpoint=('setpoint', 'first'),
        temp_media=('temp_real', 'mean'),
        temp_std=('temp_real', 'std')
    ).reset_index()

    df_agrup['erro_abs'] = (df_agrup['temp_media'] - df_agrup['setpoint']).abs()
    df_agrup['block'] = (df_agrup['setpoint'] != df_agrup['setpoint'].shift()).cumsum()
    pat_agrup = df_agrup.groupby('block').filter(lambda x: len(x) > 20)
    pat_fim_agrup = pat_agrup.groupby('block').apply(lambda x: x.tail(int(len(x) * 0.5))).reset_index(drop=True)
    mae_global = pat_fim_agrup['erro_abs'].mean() if not pat_fim_agrup.empty else df_agrup['erro_abs'].mean()

    plt.figure(figsize=(11, 5.5), dpi=100)
    plt.plot(df_agrup['passo'], df_agrup['setpoint'], 'k--', linewidth=2.0, label='Setpoint (Desejado)')
    plt.plot(df_agrup['passo'], df_agrup['temp_media'], '#1f77b4', linewidth=2.2, label=f'Média Real — MAE Patamar = {mae_global:.2f}°C')
    plt.fill_between(df_agrup['passo'], df_agrup['temp_media'] - df_agrup['temp_std'], df_agrup['temp_media'] + df_agrup['temp_std'], color='#1f77b4', alpha=0.25, label='Variância entre Testes (±1σ)')

    plt.title(f"Desempenho Consolidado — {versao} ({len(arquivos)} Execuções na Bancada)", fontweight='bold')
    plt.xlabel('Tempo (s)'); plt.ylabel('Temperatura (°C)'); plt.ylim(20, 105); plt.grid(True, linestyle=':')
    plt.legend(loc='upper right')
    
    saida_consolidada = os.path.join(pasta, f"grafico_CONSOLIDADO_{versao.replace(' ', '_')}.png")
    plt.savefig(saida_consolidada, dpi=300, bbox_inches='tight')
    print(f"✅ Concluído! Gráficos salvos na pasta '{pasta}'.\n🌟 Consolidado: {saida_consolidada}")
    plt.show()

# Executa
gerar_tudo(PASTA_DOS_LOGS, NOME_DA_VERSAO)