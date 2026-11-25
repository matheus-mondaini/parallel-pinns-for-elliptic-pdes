"""Experimentos declarativos (configs/*.yaml), execucao e registro de proveniencia (TCC, secao 3.6)."""
from .experiment import arquivo_resultado, carregar_config, carregar_resultados, expandir, manifesto, rodar_experimento

__all__ = ["arquivo_resultado", "carregar_config", "carregar_resultados", "expandir", "manifesto", "rodar_experimento"]
