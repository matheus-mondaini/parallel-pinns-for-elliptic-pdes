"""Um treino (monolitica ou XPINN) num processo proprio, com o resultado num .json.

Lancado por ``parallel_pinns.parallel.task_queue.rodar_fila`` ou a mao:

    python -m parallel_pinns.cli.run_task '{"problema": "P1", ...}' saida.json

A configuracao e um JSON com as chaves de ``task_queue.PADRAO_TAREFA``.
"""
import contextlib
import json
import math
import os
import sys
import warnings


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg, saida = json.loads(argv[0]), argv[1]

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # antes de importar o torch
    import torch

    from ..analysis.metrics import avaliar
    from ..physics.pdes import criar_problema
    from ..training import treinar_pinn, treinar_xpinn
    from ..utils.precision import modo_float64

    torch.set_num_threads(1)
    torch.set_default_device(cfg["dispositivo"])
    torch.use_deterministic_algorithms(True, warn_only=True)     # mesma configuracao do notebook
    avisos = set()

    def _aviso(message, category, filename, lineno, file=None, line=None):
        avisos.add(str(message).split("\n")[0][:300])

    warnings.showwarning = _aviso

    with modo_float64() if cfg["float64"] else contextlib.nullcontext():
        prob = criar_problema(cfg["problema"], n=cfg["n"])
        kw = dict(semente=cfg["semente"], N_f=cfg["N_f"], N_b=cfg["N_b"], lam_bc=cfg["lam_bc"],
                  epocas_adam=cfg["epocas_adam"], passos_lbfgs_max=cfg["passos_lbfgs"], imprimir=False)
        if cfg["sem_plato"]:
            kw["tolerancia_plato"] = -math.inf
        if cfg["M"] > 0:
            rede, hist, tempo = treinar_xpinn(prob, M=cfg["M"], lam_i=cfg["lam_i"], **kw)
        else:
            rede, hist, tempo = treinar_pinn(prob, **kw)
        m = avaliar(rede, prob)
        dtype = str(next(rede.parameters()).dtype)
    metricas = {k: float(v) for k, v in m.items() if k in ("L2", "H1", "PDE", "E_n", "E_1", "E_esp")}
    with open(saida, "w") as arquivo:
        json.dump({"cfg": cfg, "metricas": metricas, "it_final": hist["it_final"], "plato": bool(hist["platô"]),
                   "tempo": tempo, "dtype": dtype, "avisos": sorted(avisos)}, arquivo)


if __name__ == "__main__":
    main()
