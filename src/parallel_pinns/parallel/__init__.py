"""Motor HPC: lancamento do DDP (gloo, CPU) e fila de treinos independentes (TCC, secao 3.2)."""
from .distributed import cpu_do_rank, nome_resultado, porta_livre, rodar_ddp, rodar_ddp_lote
from .task_queue import PADRAO_TAREFA, arquivo_tarefa, rodar_fila

__all__ = ["PADRAO_TAREFA", "arquivo_tarefa", "cpu_do_rank", "nome_resultado", "porta_livre", "rodar_ddp",
           "rodar_ddp_lote", "rodar_fila"]
