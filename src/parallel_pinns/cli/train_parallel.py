"""Um treino DDP (monolitica) ou XPINN distribuido, um processo por rank, com o resultado num .json.

Lancado por ``parallel_pinns.parallel.distributed.rodar_ddp`` (que faz o papel do torchrun) ou a mao:

    python -m parallel_pinns.cli.train_parallel --problema P1 --sequencial --saida r.json
"""
import argparse
import json
import math
import warnings

import torch
import torch.distributed as dist

from ..analysis.metrics import metricas_finais
from ..physics.pdes import criar_problema
from ..training import treinar_pinn, treinar_xpinn


def main(argv=None):
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problema", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--semente", type=int, default=0)
    parser.add_argument("--contorno", default="fraco")
    parser.add_argument("--N_f", type=int, default=2000)
    parser.add_argument("--N_b", type=int, default=400)
    parser.add_argument("--epocas_adam", type=int, default=3000)
    parser.add_argument("--passos_lbfgs", type=int, default=100)
    parser.add_argument("--sem_plato", action="store_true")      # orcamento fixo: roda todos os passos do L-BFGS
    parser.add_argument("--sem_avaliacao", action="store_true")  # medicao de tempo: avalia so no inicio e no fim
    parser.add_argument("--sequencial", action="store_true")     # referencia: treino sem distribuir, 1 processo
    parser.add_argument("--M", type=int, default=0)               # 0: monolitica (DDP); M > 0: XPINN M x M distribuido
    parser.add_argument("--lam_i", type=float, default=1.0)       # so no XPINN
    parser.add_argument("--saida", required=True)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    distribuido = not args.sequencial
    if distribuido:
        dist.init_process_group("gloo")      # env://: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT (e USE_LIBUV=0)
    rank = dist.get_rank() if distribuido else 0

    prob = criar_problema(args.problema, n=args.n)
    kw = dict(contorno=args.contorno, N_f=args.N_f, N_b=args.N_b, epocas_adam=args.epocas_adam,
              passos_lbfgs_max=args.passos_lbfgs, semente=args.semente, distribuido=distribuido, imprimir=False)
    if args.sem_plato:
        kw["tolerancia_plato"] = -math.inf
    if args.sem_avaliacao:
        kw["avaliar_a_cada"] = 10**9
    if args.M > 0:
        rede, hist, tempo = treinar_xpinn(prob, M=args.M, lam_i=args.lam_i, **kw)
    else:
        rede, hist, tempo = treinar_pinn(prob, **kw)

    if rank == 0:
        metricas = metricas_finais(rede, prob)
        with open(args.saida, "w") as arquivo:
            json.dump({"processos": dist.get_world_size() if distribuido else 1, "tempo": tempo,
                       "metricas": metricas, "hist": hist}, arquivo)

    if distribuido:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
