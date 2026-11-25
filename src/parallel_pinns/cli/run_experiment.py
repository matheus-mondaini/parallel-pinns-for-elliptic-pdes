"""Roda um experimento declarado em configs/*.yaml: um .json por execucao em <resultados>/<nome>/ e um manifesto.

    python -m parallel_pinns.cli.run_experiment configs/exploratorio/h2_confirmatorio.yaml
    python -m parallel_pinns.cli.run_experiment configs/exploratorio/h2_confirmatorio.yaml --listar

Execucoes que ja tem .json sao reaproveitadas (retomada depois de uma interrupcao), salvo com --refazer.
"""
import argparse

from ..tracking.experiment import arquivo_resultado, carregar_config, expandir, rodar_experimento


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--resultados", default="results", help="pasta raiz dos resultados (padrao: results)")
    parser.add_argument("--listar", action="store_true", help="so lista as execucoes, sem rodar")
    parser.add_argument("--refazer", action="store_true", help="roda de novo mesmo se o .json existir")
    parser.add_argument("--nucleos-cpu", default="2,4,6,8,10",
                        help="processadores logicos das tarefas em CPU, um por tarefa (padrao: 2,4,6,8,10)")
    parser.add_argument("--nucleos-ddp", type=int, default=6, help="maximo de processos DDP ao mesmo tempo (padrao: 6)")
    args = parser.parse_args(argv)

    cfg = carregar_config(args.config)
    execucoes = expandir(cfg)
    print(f"{cfg['nome']}: {len(execucoes)} execucoes ({cfg['tipo']})")
    if args.listar:
        for t in execucoes:
            print(" ", arquivo_resultado(cfg["tipo"], t))
        return
    rodar_experimento(cfg, resultados=args.resultados, reaproveitar=not args.refazer,
                      nucleos_cpu=tuple(int(c) for c in args.nucleos_cpu.split(",")), nucleos_ddp=args.nucleos_ddp)
    print(f"resultados em {args.resultados}/{cfg['nome']}/")


if __name__ == "__main__":
    main()
