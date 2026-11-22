"""Casos curtos de regressao, executados igualmente pela versao exploratoria (v0) e pelo pacote em src/.

Uso (sempre num processo novo, para o estado do torch ser o mesmo nos dois lados):

    python casos.py v0  <pasta com pinn_comum.py e pinn_xpinn.py>  saida.json
    python casos.py src <qualquer>                                 saida.json

Cada caso treina 30 epocas de Adam + 2 passos de L-BFGS (70 iteracoes) em CPU, 1 thread, modo deterministico,
e guarda as metricas, toda a historia da perda e um hash dos parametros finais: a comparacao e bit a bit.
"""
import hashlib
import json
import math
import sys

CURTO = dict(epocas_adam=30, passos_lbfgs_max=2, tolerancia_plato=-math.inf, imprimir=False)

CASOS = {
    "mono_P1_fraco_s0": ("pinn", "P1", None, dict(semente=0)),
    "mono_P2n11_fraco_lambc0p3_s3": ("pinn", "P2", 11, dict(semente=3, lam_bc=0.3)),
    "mono_P0_exato_s1": ("pinn", "P0", None, dict(semente=1, contorno="exato")),
    "mono_L0_transfinito_s2": ("pinn", "L0", None, dict(semente=2, contorno="transfinito")),
    "xpinn_P2n11_M4_lami0p03_s0": ("xpinn", "P2", 11, dict(semente=0, M=4, lam_i=0.03)),
    "xpinn_L0_M2_exato_s1": ("xpinn", "L0", None, dict(semente=1, M=2, contorno="exato")),
}


def assinatura(rede):
    h = hashlib.sha256()
    for p in rede.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def executar(api):
    resultados = {}
    for nome, (tipo, problema, n, kw) in CASOS.items():
        prob = api["criar_problema"](problema, n=n)
        treinar = api["treinar_pinn"] if tipo == "pinn" else api["treinar_xpinn"]
        rede, hist, _ = treinar(prob, **CURTO, **kw)
        m = api["avaliar"](rede, prob)
        resultados[nome] = {
            "metricas": {k: float(v) for k, v in m.items() if k in ("L2", "H1", "PDE", "E_n", "E_1", "E_esp")},
            "loss": hist["loss"], "it_final": hist["it_final"], "parametros_sha256": assinatura(rede)}
    return resultados


def carregar_api(lado, pasta):
    if lado == "v0":
        sys.path.insert(0, pasta)
        import pinn_comum
        import pinn_xpinn
        return {"criar_problema": pinn_comum.criar_problema, "treinar_pinn": pinn_comum.treinar_pinn,
                "treinar_xpinn": pinn_xpinn.treinar_xpinn, "avaliar": pinn_comum.avaliar}
    from parallel_pinns.analysis import avaliar
    from parallel_pinns.physics import criar_problema
    from parallel_pinns.training import treinar_pinn, treinar_xpinn
    return {"criar_problema": criar_problema, "treinar_pinn": treinar_pinn, "treinar_xpinn": treinar_xpinn,
            "avaliar": avaliar}


if __name__ == "__main__":
    import os
    import warnings

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    lado, pasta, saida = sys.argv[1:4]
    with open(saida, "w") as arquivo:
        json.dump(executar(carregar_api(lado, pasta)), arquivo, indent=1)
