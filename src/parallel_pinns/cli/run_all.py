"""Execucao unica de varios experimentos (a execucao final do TCC):

    python -m parallel_pinns.cli.run_all configs/final/*.yaml
    python -m parallel_pinns.cli.run_all configs/final/*.yaml --listar

Primeiro as medicoes de tempo (``exclusivo: true``), uma por vez e sozinhas na maquina. Depois todas as execucoes
de acuracia ao mesmo tempo: a fila de tarefas (uma na GPU e uma por nucleo de --nucleos-cpu) e o DDP em paralelo
(ate --nucleos-ddp processos). Retoma de onde parou: execucoes com .json sao reaproveitadas.
"""
import argparse
import glob

from ..tracking.experiment import arquivo_resultado, carregar_config, expandir, pasta_resultados, rodar_varios


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--resultados", default="results")
    parser.add_argument("--listar", action="store_true", help="so conta as execucoes, sem rodar")
    parser.add_argument("--nucleos-cpu", default="2,4")
    parser.add_argument("--nucleos-ddp", type=int, default=4)
    args = parser.parse_args(argv)

    caminhos = sorted({c for padrao in args.configs for c in glob.glob(padrao)})
    cfgs = [carregar_config(c) for c in caminhos]
    unicas = {}
    for cfg in cfgs:
        execucoes = expandir(cfg)
        novas = {(str(pasta_resultados(cfg, args.resultados)), arquivo_resultado(cfg["tipo"], t)) for t in execucoes}
        print(f"{cfg['nome']:>28}: {len(execucoes):4d} execucoes ({cfg['tipo']}{', tempo' if cfg.get('exclusivo') else ''})")
        unicas.update(dict.fromkeys(novas))
    print(f"{'total sem repeticao':>28}: {len(unicas):4d}")
    if args.listar:
        return
    rodar_varios(cfgs, resultados=args.resultados, nucleos_cpu=tuple(int(c) for c in args.nucleos_cpu.split(",")),
                 nucleos_ddp=args.nucleos_ddp, registrar=lambda msg: print(msg, flush=True))


if __name__ == "__main__":
    main()
