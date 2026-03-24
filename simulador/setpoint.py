import numpy as np


def gerar_setpoint_pcr(
    ciclos: int = 2,
    t_estabilizacao: int = 10,
    t_desnaturacao: int = 20,
    t_anelamento: int = 20,
    t_extensao: int = 20,
) -> np.ndarray:
    """
    Gera o perfil de setpoints de temperatura para simular ciclos de PCR.

    O perfil segue a sequência padrão de um termociclador:
        1. Estabilização em temperatura ambiente (25°C)
        2. Por ciclo:
            - Desnaturação: 95°C — separa as fitas de DNA
            - Anelamento:   55°C — primers se ligam ao DNA
            - Extensão:     72°C — polimerase replica o DNA

    Args:
        ciclos:           número de ciclos PCR a simular
        t_estabilizacao:  duração da fase inicial em segundos
        t_desnaturacao:   duração da desnaturação em segundos
        t_anelamento:     duração do anelamento em segundos
        t_extensao:       duração da extensão em segundos

    Returns:
        Array numpy com a sequência de setpoints (°C) ao longo do tempo.
    """
    perfil = []

    # fase inicial: sistema estabiliza na temperatura ambiente antes de começar os ciclos
    perfil.extend([25.0] * t_estabilizacao)

    for _ in range(ciclos):
        perfil.extend([95.0] * t_desnaturacao)  # desnaturação
        perfil.extend([55.0] * t_anelamento)    # anelamento
        perfil.extend([72.0] * t_extensao)      # extensão

    return np.array(perfil, dtype=float)