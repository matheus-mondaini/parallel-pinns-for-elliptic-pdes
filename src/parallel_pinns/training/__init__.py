"""Laco de otimizacao e treino da PINN monolitica e do XPINN (sequencial ou distribuido)."""
from .optimizer import otimizar
from .trainer import SEMENTE, treinar_pinn
from .xpinn_trainer import treinar_xpinn

__all__ = ["SEMENTE", "otimizar", "treinar_pinn", "treinar_xpinn"]
