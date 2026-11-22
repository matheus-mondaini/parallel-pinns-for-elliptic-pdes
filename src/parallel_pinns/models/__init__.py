"""Camada do aproximador neural: MLP monolitica e XPINN (TCC, secao 3.2)."""
from .networks import criar_rede, largura_iso_parametro, n_parametros
from .xpinn import XPINN, agrupar_por_subdominio, pontos_interface

__all__ = ["XPINN", "agrupar_por_subdominio", "criar_rede", "largura_iso_parametro", "n_parametros",
           "pontos_interface"]
