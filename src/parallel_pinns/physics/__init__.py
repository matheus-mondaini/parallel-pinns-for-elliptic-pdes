"""Camada matematica/fisica: benchmarks, operadores, geometria e contorno (TCC, secao 3.2)."""
from .boundary_conditions import ContornoExato, ContornoTransfinito
from .geometry import distancia_contorno, grade_interior, pontos_colocacao, pontos_contorno
from .operators import laplaciano
from .pdes import criar_problema, phi

__all__ = ["ContornoExato", "ContornoTransfinito", "criar_problema", "distancia_contorno", "grade_interior",
           "laplaciano", "phi", "pontos_colocacao", "pontos_contorno"]
