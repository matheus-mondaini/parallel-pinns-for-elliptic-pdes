"""Diagnostico: metricas de acuracia, analise espectral (vies espectral) e interface do XPINN."""
from .fourier import coeficientes_seno, metricas_espectrais
from .interface import residuo_por_subdominio, saltos_interface
from .metrics import avaliar, balanco_global, time_to_solution

__all__ = ["avaliar", "balanco_global", "coeficientes_seno", "metricas_espectrais", "residuo_por_subdominio",
           "saltos_interface", "time_to_solution"]
