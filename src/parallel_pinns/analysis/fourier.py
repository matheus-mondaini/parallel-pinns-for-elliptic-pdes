"""Coeficientes de seno (DST-I) do campo na grade interior e metricas espectrais E_n, E_1 e E_esp."""
import numpy as np
from scipy.fft import dstn


def coeficientes_seno(U):
    # U: matriz N x N na grade interior (indexacao ij). Devolve a[m-1, n-1].
    N = U.shape[0]
    return dstn(U, type=1) / (N + 1) ** 2


def metricas_espectrais(a, n, eps):
    E_n = abs(a[n - 1, n - 1] - eps) / eps
    E_1 = abs(a[0, 0] - 1.0)
    resto = a.copy()
    resto[0, 0] = 0.0
    resto[n - 1, n - 1] = 0.0
    E_esp = np.sqrt(np.sum(resto ** 2))
    return E_n, E_1, E_esp
