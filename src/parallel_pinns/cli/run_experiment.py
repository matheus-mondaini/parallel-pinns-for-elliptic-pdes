"""Roda um experimento declarado em configs/*.yaml: um .json por execucao em <resultados>/<nome>/ e um manifesto.

    python -m parallel_pinns.cli.run_experiment configs/h2_confirmatorio.yaml
    python -m parallel_pinns.cli.run_experiment configs/h2_confirmatorio.yaml --listar

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
    args = parser.parse_args(argv)

    cfg = carregar_config(args.config)
    execucoes = expandir(cfg)
    print(f"{cfg['nome']}: {len(execucoes)} execucoes ({cfg['tipo']})")
    if args.listar:
        for t in execucoes:
            print(" ", arquivo_resultado(cfg["tipo"], t))
        return
    rodar_experimento(cfg, resultados=args.resultados, reaproveitar=not args.refazer)
    print(f"resultados em {args.resultados}/{cfg['nome']}/")


if __name__ == "__main__":
    main()
