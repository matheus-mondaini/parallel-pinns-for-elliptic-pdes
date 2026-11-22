"""Metricas de desempenho paralelo: speedup, eficiencia e ajuste da lei de Amdahl (Eq. 17 do TCC)."""
import numpy as np


def speedup(tempos):
    # tempos: {p: tempo por iteracao}. S(p) = T(1) / T(p)
    return {p: tempos[1] / t for p, t in tempos.items()}


def eficiencia(tempos):
    return {p: s / p for p, s in speedup(tempos).items()}


def ajustar_amdahl(S):
    # Minimos quadrados em 1 - 1/S(p) = f (1 - 1/p), sem p = 1. f: fracao paralelizavel.
    ps = sorted(p for p in S if p != 1)
    x = np.array([1 - 1 / p for p in ps])
    y = np.array([1 - 1 / S[p] for p in ps])
    return float(x @ y / (x @ x))


def amdahl(p, f):
    # Speedup previsto pela lei de Amdahl com fracao paralelizavel f
    return 1 / ((1 - f) + f / p)
