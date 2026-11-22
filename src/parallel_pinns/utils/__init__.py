"""Determinismo, precisao numerica e metricas de desempenho paralelo."""
from .metrics import ajustar_amdahl, amdahl, eficiencia, speedup
from .precision import modo_float64
from .seeds import configurar_determinismo

__all__ = ["ajustar_amdahl", "amdahl", "configurar_determinismo", "eficiencia", "modo_float64", "speedup"]
