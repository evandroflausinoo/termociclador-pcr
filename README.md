# Termociclador PCR: Controle Térmico Inteligente em Rede (PID vs. PPO)

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20BTS7960%20%7C%20TEC1--12706-red.svg)](#3-especificações-de-hardware-e-eletrônica)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Plataforma experimental e computacional de controle térmico em malha fechada aplicada a um protótipo de **termociclador de PCR de baixo custo**. O sistema avalia a robustez e o tempo de convergência de estratégias de controle sob condições adversas de comunicação em redes IoT/Sistemas Ciberfísicos (CPS), incluindo **latência variável (*jitter* de 1,0 a 3,5 s)** e **taxas de perda de pacotes de até 30%**[cite: 8].

O projeto compara o desempenho do **Controle Clássico (PID Amortecido com Anti-Windup)** contra o **Controle Inteligente por Aprendizado por Reforço (*Proximal Policy Optimization* - PPO)** operando em simulação (*Domain Randomization*) e transferido diretamente para a bancada física (*Sim-to-Real Transfer*)[cite: 8, 10, 11].

---

## 1. Evolução Experimental do Projeto

Na fase inicial de prova de conceito, o projeto utilizava um modelo puramente matemático com coeficientes teóricos simétricos ($\beta = 3,0$ e $\alpha = 0,02$) e setpoints com duração fixa de tempo.

A fase atual introduz a **validação física e identificação paramétrica real**:
1. **Construção do Protótipo Físico Instrumentado**: Bloco térmico em alumínio confinado dentro de uma **câmara de isolamento em acrílico**, reduzindo a perda convectiva desordenada e protegendo o sistema contra correntes de ar externas[cite: 6].
2. **Resfriamento Ativo por Convecção Forçada**: Adição de um cooler superior 12 V na câmara, chaveado via MOSFET e acoplado ao pino de PWM reverso (**D26**) do ESP32 para atuar sincronizado com a inversão da pastilha Peltier[cite: 6].
3. **Identificação de Tempos Mortos Assimétricos**: Levantamento experimental dos atrasos de difusão de calor ($L_{\text{heat}} \approx 6,6\text{ s}$ e $L_{\text{cool}} \approx 5,1\text{ s}$) e ganhos reais de transferência térmica[cite: 6, 8].
4. **Máquina de Estados Orientada a Eventos (*Event-Driven PCR*)**: Transição entre fases condicionada a *holds* térmicos contínuos de 30 segundos dentro da faixa de tolerância de cada reação enzimática ($\pm 1,0^\circ\text{C}$)[cite: 8, 9].

---

## 2. Arquitetura do Sistema

                   ┌─────────────────────────────────────────────────────────┐
                   │                   Estação de Controle                   │
                   │                                                         │
                   │   ┌───────────────┐           ┌─────────────────────┐   │
                   │   │  Agente PPO   │           │   Controlador PID   │   │
                   │   │ (SB3 Policy)  │           │   (Anti-Windup)     │   │
                   │   └───────┬───────┘           └──────────┬──────────┘   │
                   │           │     u_raw ∈ [-1, 1]          │              │
                   │           └──────────────┬───────────────┘              │
                   │                          ▼                              │
                   │               ┌───────────────────────┐                 │
                   │               │     Safety Shield     │                 │
                   │               │ (Slew-Rate / T ≤ 98°C)│                 │
                   │               └──────────┬────────────┘                 │
                   │                          │ u_aplicado                   │
                   │                          ▼                              │
                   │   ┌─────────────────────────────────────────────────┐   │
                   │   │  Gymnasium Env (Máquina de Estados de 3 Fases)  │   │
                   │   └──────────────────────┬──────────────────────────┘   │
                   └──────────────────────────┼──────────────────────────────┘
                                              │
                        Transporte: Serial USB (COM) ou UDP (Wi-Fi)
                        Rede Adversária: Jitter (1.0–3.5s) + Perda (30%)
                                              │
                                              ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │             Bancada Física & Câmara Térmica             │
                   │                                                         │
                   │   [ESP32 MCU] ──(PWM)──> [Ponte H BTS7960]              │
                   │        │                         │                      │
                   │      (D26)                 (Polaridade ±)               │
                   │        │                         │                      │
                   │        ▼                         ▼                      │
                   │   [MOSFET Cooler]       [Peltier TEC1-12706]            │
                   │        │                         │                      │
                   │        └───────► [Bloco Térmico] ◄──────────────────────┘
                   │                  (Cúpula de Acrílico)                   │
                   │                          │                              │
                   │                 [Sensor DS18B20]                        │
                   │                          │ Leitura digital              │
                   │                          └───────────► ESP32 Telemetria │
                   └─────────────────────────────────────────────────────────┘

---

## 3. Especificações de Hardware e Eletrônica

* **Controlador Embarcado:** ESP32 NodeMCU (32 bits, 240 MHz) com firmware em C++/Arduino gerenciando aquisição digital de temperatura e saídas PWM bidirecionais.
* **Módulo Termoelétrico:** Pastilha Peltier TEC1-12706 ($I_{\max} = 6\text{ A}, V_{\max} = 12\text{ V}$) fixada sobre bloco usinado de alumínio.
* **Driver de Potência:** Ponte H BTS7960 (capacidade até 43 A) operando com modulação PWM bidirecional ($u > 0$ aquecimento por efeito Peltier direto, $u < 0$ resfriamento por efeito Peltier reverso).
* **Câmara de Testes e Convecção Forçada:** 
  * Cúpula de acrílico para estabilização do microambiente térmico.
  * Cooler superior de 12 V acionado via circuito chaveador com MOSFET acoplado ao pino **D26** (`PINO_ESFRIAR`), ligando e modulando automaticamente a extração de calor durante as fases de resfriamento ($u < 0$).
* **Sensoriamento:** Sensor digital blindado Dallas DS18B20 (comunicação 1-Wire, resolução de 12 bits / $0,0625^\circ\text{C}$) em contato com a cavidade central do bloco.

---

## 4. Gêmeo Digital e Modelo Térmico Identificado

O simulador implementa a dinâmica física do sistema baseada em uma Equação Diferencial Ordinária (EDO) de 1ª ordem (*FOPDT*) com tempo morto assimétrico identificado experimentalmente:

$$\frac{dT(t)}{dt} = \beta(u) \cdot u(t - L) - \alpha \cdot \big(T(t) - T_{\text{amb}}\big) + \omega(t)$$

Onde:
* $u(t) \in [-1,0; +1,0]$ é o sinal de atuação normalizado.
* $\beta(u) = \beta_{\text{heat}}$ quando $u \ge 0$, e $\beta(u) = \beta_{\text{cool}}$ quando $u < 0$.
* $L$ é o tempo morto de transporte térmico ($L_{\text{heat}}$ ou $L_{\text{cool}}$).
* $\omega(t)$ representa distúrbios térmicos e ruído gaussiano do sensor.

### Constantes Calibradas na Bancada Real (`config/thermal_params.yaml`)

| Parâmetro | Símbolo | Valor Identificado | Descrição Física |
| :--- | :---: | :---: | :--- |
| **Temperatura Ambiente** | $T_{\text{amb}}$ | $26,19^\circ\text{C}$ | Temperatura ambiente de referência da bancada |
| **Dissipação Convectiva** | $\alpha$ | $0,00327\text{ s}^{-1}$ | Perda passiva de calor para o ar confinante |
| **Ganho de Aquecimento** | $\beta_{\text{heat}}$ | $0,3686^\circ\text{C}/\text{s}$ | Taxa de elevação térmica a 100% de potência ($u = +1,0$) |
| **Ganho de Resfriamento** | $\beta_{\text{cool}}$ | $0,1200^\circ\text{C}/\text{s}$ | Taxa de resfriamento combinado (Peltier reverso + Cooler) ($u = -1,0$) |
| **Tempo Morto (Aquecimento)** | $L_{\text{heat}}$ | $6,59\text{ s}$ | Atraso de difusão de calor através da junção semicondutora |
| **Tempo Morto (Resfriamento)** | $L_{\text{cool}}$ | $5,13\text{ s}$ | Atraso de difusão no resfriamento ativo |

---

## 5. Máquina de Estados Reativa do Protocolo PCR

Diferente de sistemas de temporização cega, a transição entre fases segue uma **Máquina de Estados Orientada a Eventos**. Uma etapa só é considerada concluída quando o sistema atinge a faixa de tolerância e sustenta a temperatura de forma contínua pelo tempo de *hold* exigido.

             ┌────────────────────────────────────────┐
             │       Estabilização Inicial (20 s)     │
             └───────────────────┬────────────────────┘
                                 │ (t >= 20 s)
                                 ▼
    ┌──────────────────────────────────────────────────────────┐
    │  Desnaturação: Alvo 95°C | Hold 30 s se T >= 94°C        │◄────────┐
    └────────────────────────────┬─────────────────────────────┘         │
                                 │ (Hold ininterrupto cumprido)          │
                                 ▼                                       │ (Se ciclo < total)
    ┌──────────────────────────────────────────────────────────┐         │
    │  Anelamento:   Alvo 55°C | Hold 30 s se T <= 56°C        │         │
    └────────────────────────────┬─────────────────────────────┘         │
                                 │ (Hold ininterrupto cumprido)          │
                                 ▼                                       │
    ┌──────────────────────────────────────────────────────────┐         │
    │  Extensão:     Alvo 72°C | Hold 30 s se T >= 71°C        │─────────┘
    └────────────────────────────┬─────────────────────────────┘
                                 │ (Último ciclo finalizado)
                                 ▼
             ┌────────────────────────────────────────┐
             │          Protocolo Concluído           │
             └────────────────────────────────────────┘

| Fase | Setpoint | Condição de Entrada no *Hold* | Tempo de *Hold* | *Timeout* de Segurança |
| :--- | :---: | :---: | :---: | :---: |
| **Estabilização** | $25,0^\circ\text{C}$ | Transitório inicial de boot | $20\text{ s}$ | — |
| **Desnaturação** | $95,0^\circ\text{C}$ | $T_{\text{real}} \ge 94,0^\circ\text{C}$ | $30\text{ s}$ contínuos | $600\text{ s}$ |
| **Anelamento** | $55,0^\circ\text{C}$ | $T_{\text{real}} \le 56,0^\circ\text{C}$ | $30\text{ s}$ contínuos | $300\text{ s}$ |
| **Extensão** | $72,0^\circ\text{C}$ | $T_{\text{real}} \ge 71,0^\circ\text{C}$ | $30\text{ s}$ contínuos | $300\text{ s}$ |

> **Rigor Bioquímico**: Caso oscilações de controle ou atrasos de rede façam a temperatura sair da faixa de tolerância antes de completar os 30 s, o contador de *hold* é zerado imediatamente, garantindo a integridade química das fitas de DNA.

---

## 6. Estratégias de Controle

### 6.1. Agente Inteligente (PPO - Proximal Policy Optimization)
* **Algoritmo:** PPO com arquitetura Ator-Crítico (*Multi-Layer Perceptron*)[cite: 7].
* **Vetor de Observação ($8$ dimensões normalizadas):**
  $$s_t = \left[ \frac{T_t}{100}, \frac{T_{t-1}}{100}, \frac{T_{t-2}}{100}, \frac{T_{\text{target}} - T_t}{50}, \frac{T_{\text{target}}}{100}, \frac{\Delta T_t / \Delta t}{2}, u_{t-1}, u_{t-2} \right]$$[cite: 9]
* **Espaço de Ação:** Contínuo $u_t \in [-1,0; +1,0]$[cite: 9].
* **Treinamento com Domain Randomization:** Variação estocástica de $\alpha \in [0,0020; 0,0045]$, $T_{\text{amb}} \in [22^\circ\text{C}; 29^\circ\text{C}]$ e injeção de atrasos/perdas de rede na simulação[cite: 8].

### 6.2. Controlador Clássico (PID Amortecido)
* **Estrutura:** PID com derivada atuando na medição ($\Delta T / \Delta t$) para eliminar picos no chaveamento de setpoint[cite: 10].
* **Ganhos Sintonizados:** $K_p = 0,45$, $K_i = 0,020$, $K_d = 1,40$[cite: 10].
* **Anti-Windup Inteligente:** Congelamento condicional da integral durante a saturação do atuador em $\pm 1,0$[cite: 10].
* **Reset de Integral:** A integral acumulada é zerada a cada transição de fase da máquina de estados para evitar arrasto de erro de patamares anteriores[cite: 10].

### 6.3. Camada de Segurança (*Safety Shield*)
* **Corte Térmico de Emergência:** Ativação automática se $T_{\text{real}} \ge 98,0^\circ\text{C}$ (força $u = -0,5$).
* **Limitador de Taxa de Variação (*Slew-Rate Limiter*):** Restringe $\Delta u \le 0,25/\text{s}$ para proteger a integridade dos transistores MOSFET da ponte H BTS7960 contra picos de corrente reversa[cite: 8].

```

## 7. Estrutura do Repositório

pcr_control/
├── config/
│   └── thermal_params.yaml         # Parâmetros físicos e tempos da máquina de estados
├── src/
│   └── pcr_control/
│       ├── comms/                  # Drivers de comunicação (Serial USB / UDP Wi-Fi)
│       │   ├── backend.py
│       │   ├── serial_backend.py
│       │   └── udp_backend.py
│       ├── control/                # Lógicas de controle e segurança
│       │   ├── agent_interface.py  # Interface unificada Agent (act / reset)
│       │   ├── pid.py              # Controlador PID com anti-windup e reset
│       │   └── safety_shield.py    # Safety Shield (slew-rate / corte 98°C)
│       ├── envs/                   # Ambientes Gymnasium e Simulação de Rede
│       │   ├── network_sim.py      # Simulador de jitter e perda de pacotes
│       │   └── termociclador_env.py# Ambiente com Máquina de Estados Reativa
│       ├── logging_utils/          # Telemetria e persistência
│       │   └── experiment_logger.py# Gravador automático de data.csv e meta.json
│       ├── model/                  # Modelo físico do bloco térmico
│       │   ├── thermal_model.py    # EDO não linear FOPDT
│       │   └── setpoint.py         # Perfis térmicos nominais
│       ├── viz/                    # Visualização e Métricas
│       │   └── plots.py            # Geração de painel.png e cálculo de RMSE
│       ├── cli.py                  # Ponto de entrada CLI unificado
│       └── config.py               # Mapeamento do YAML em Dataclasses tipadas
├── logs/                           # Ensaios experimentais estruturados com timestamp
├── modelos/                        # Checkpoints dos modelos treinados do PPO (.zip)
├── CHANGELOG.md                    # Histórico e justificativas metodológicas de engenharia
├── pyproject.toml                  # Dependências e empacotamento do pacote Python
└── README.md                       # Documentação técnica

```

## 8. Instalação e Uso

### 8.1. Instalação
Requer Python 3.10 ou superior:

```bash
# Clone o repositório
git clone [https://github.com/evandroflausinoo/termociclador-pcr.git](https://github.com/evandroflausinoo/termociclador-pcr.git)
cd termociclador-pcr

# Instale o pacote local em modo editável
pip install -e .

8.2. Identificação Térmica da Planta Física (Malha Aberta)
Para coletar dados de caracterização direta da bancada:

# Aquecimento puro a 100% de potência (u = +1.0)
python -m pcr_control.cli caracterizar --fase aquecimento --porta COM3

# Resfriamento ativo (Peltier invertida + Cooler) (u = -1.0)
python -m pcr_control.cli caracterizar --fase resfriamento_ativo --porta COM3 --partir_de 90.0

8.3. Treinamento do Agente PPO em Simulação
Treina a política sob 8 ambientes paralelos com jitter e perdas estocásticas de rede:

python -m pcr_control.cli treinar --passos 1000000 --n_envs 8 --jitter --salvar_dir modelos

8.4. Execução Experimental no Protótipo Físico (Hardware Real)
Executa o protocolo de 2 ciclos completos com registro automático de dados:

# Executar PID no hardware físico via Serial USB
python -m pcr_control.cli executar --pid --serial --porta COM3 --shield clamp_only --ciclos 2

# Executar Agente PPO treinado no hardware físico via Serial USB
python -m pcr_control.cli executar --modelo modelos/ppo_pcr_final.zip --serial --porta COM3 --shield clamp_only --ciclos 2

8.5. Execução em Simulação (Ambiente com Jitter e Perdas de Rede)

# Simulação do PID com atrasos de 1.0s a 3.5s e 30% de perda
python -m pcr_control.cli executar --pid --sim --jitter --shield clamp_only --ciclos 2

# Simulação do PPO com atrasos de 1.0s a 3.5s e 30% de perda
python -m pcr_control.cli executar --modelo modelos/ppo_pcr_final.zip --sim --jitter --shield clamp_only --ciclos 2

8.6. Visualização e Extração de Métricas

python -m pcr_control.cli plotar logs/pid_clamp_only/<pasta_do_ensaio>/data.csv

9. Métricas Científicas de AvaliaçãoCada execução gera um arquivo data.csv e um meta.json estruturados na pasta logs/. O sistema processa automaticamente as seguintes métricas:RMSE Global ($^\circ\text{C}$): Erro quadrático médio em relação ao setpoint dinâmico.Tempo Total de Conclusão do Protocolo ($t_{\text{total}}$): Duração real para validação de todos os holds biológicos.Overshoot na Desnaturação ($^\circ\text{C}$): Desvio de pico acima de $95,0^\circ\text{C}$.Fração Fora da Faixa ($|e| > 0,5^\circ\text{C}$): Percentual do tempo fora da tolerância nominal.Intervenções do Shield: Frequência de atuações do limitador de slew-rate na ponte H.

````
10. GitHub: @evandroflausinoo
