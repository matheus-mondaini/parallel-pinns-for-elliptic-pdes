"""Treino da PINN monolitica: sequencial ou DDP (pontos particionados entre processos)."""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..analysis.metrics import avaliar
from ..models import networks
from ..physics import geometry
from ..physics.boundary_conditions import ContornoExato, ContornoTransfinito
from ..physics.operators import laplaciano
from .optimizer import otimizar

SEMENTE = 0


def treinar_pinn(prob, contorno="fraco", epocas_adam=3000, passos_lbfgs_max=100, iter_por_passo=20,
                 tolerancia_plato=0.01, janela_plato=10, N_f=2000, N_b=400, lr=1e-3, largura=64, camadas_ocultas=4,
                 semente=SEMENTE, lam_pde="auto", lam_bc=1.0, avaliar_a_cada=100, imprimir=True, distribuido=False,
                 N_aval=63):
    # contorno: "fraco" (nucleo, eq. 10 do TCC), "exato" (D produto) ou "transfinito" (sensibilidade, secao 9).
    # tolerancia_plato = -inf: orcamento fixo, roda todos os passos do L-BFGS. N_aval: grade de avaliacao N x N.
    torch.manual_seed(semente)
    rede = networks.criar_rede(largura, camadas_ocultas)
    if contorno == "exato":
        rede_final = ContornoExato(rede, prob["G"])                 # u = D * NN + G
    elif contorno == "transfinito":
        rede_final = ContornoTransfinito(rede, prob["G"])           # u = NN - T[NN] + T[G]
    else:
        rede_final = rede                                           # contorno por penalidade

    # Pontos GLOBAIS, fixos durante o treino (iguais em todos os processos)
    xy_f = geometry.pontos_colocacao(N_f, semente)
    f_f = prob["f"](xy_f[:, 0:1], xy_f[:, 1:2]).detach()
    xy_b = geometry.pontos_contorno(N_b) if contorno == "fraco" else torch.zeros(0, 2)

    # Peso da EDP: "auto" = 1/media(f^2) sobre o conjunto GLOBAL (no DDP nao pode ser por rank).
    # Contorno fraco: poe o residuo na escala do termo de borda (solucoes com amplitude ~1).
    # Contorno exato: a perda tem um termo so, e esse fator apenas deixa os numeros na casa de 1.
    if lam_pde == "auto":
        escala = torch.mean(f_f**2).item()
        lam_pde = 1.0 / escala if escala > 0 else 1.0

    # DDP: cada processo fica com uma fatia de tamanho IGUAL dos pontos (secao 10 do documento)
    modelo, p = rede_final, 1
    if distribuido:
        rank, p = dist.get_rank(), dist.get_world_size()
        assert N_f % p == 0 and len(xy_b) % p == 0, "N_f e N_b precisam ser divisiveis pelo numero de processos"
        xy_f = torch.chunk(xy_f, p)[rank]
        xy_b = torch.chunk(xy_b, p)[rank] if len(xy_b) else xy_b
        modelo = DDP(rede_final)

    xy_f = xy_f.clone().requires_grad_(True)
    f_f = prob["f"](xy_f[:, 0:1], xy_f[:, 1:2]).detach()
    g_b = prob["u"](xy_b[:, 0:1], xy_b[:, 1:2]).detach() if len(xy_b) else None
    n_f = xy_f.shape[0]

    def perdas(registro=False):
        # Uma unica passagem pela rede com interior + contorno (o DDP exige 1 forward por backward).
        # registro=True: so para registrar, sem passar pelo DDP.
        mod = rede_final if registro else modelo
        u = mod(torch.cat([xy_f, xy_b]))
        residuo = -laplaciano(u[:n_f], xy_f) - f_f          # -lap(u) - f deve ser 0
        loss_pde = torch.mean(residuo**2)
        if contorno == "fraco":
            loss_bc = torch.mean((u[n_f:] - g_b)**2)        # u = g na borda (penalidade)
        else:
            loss_bc = torch.zeros(())                        # u = g na borda por construcao
        return lam_pde * loss_pde + lam_bc * loss_bc, loss_pde, loss_bc

    def media_global(valor):
        # Valor medio entre processos (para registrar e para a busca de linha do L-BFGS)
        if not distribuido:
            return valor
        v = valor.detach().clone()
        dist.all_reduce(v)
        return v / p

    hist = {"it": [], "loss": [], "pde": [], "bc": [], "aval_it": [], "aval_tempo": [], "L2": [], "E_n": [], "E_1": [],
            "fim_adam": epocas_adam, "lam_pde": lam_pde, "lam_bc": lam_bc, "contorno": contorno}
    relogio = [0.0]

    def registrar(it, valores, forcar=False):
        hist["it"].append(it)
        hist["loss"].append(valores[0])
        hist["pde"].append(valores[1])
        hist["bc"].append(valores[2])
        if forcar:
            m = avaliar(rede_final, prob, N_aval)
            hist["aval_it"].append(it)
            hist["aval_tempo"].append(relogio[0])
            hist["L2"].append(m["L2"])
            if "n" in prob:
                hist["E_n"].append(m["E_n"])
                hist["E_1"].append(m["E_1"])
            return m

    def mostrar(fase, it, valores, m):
        if imprimir and m is not None:
            extra = f" | E_n {m['E_n']:.3f}" if "n" in prob else ""
            print(f"[{prob['nome']}] {fase} it {it:5d} | loss {valores[0]:.2e} | PDE {valores[1]:.2e} | BC {valores[2]:.2e} | L2 {m['L2']:.2e}{extra}")

    it_final, passos = otimizar(list(rede_final.parameters()), perdas, registrar, mostrar, relogio, media_global,
                                epocas_adam, passos_lbfgs_max, iter_por_passo, lr, tolerancia_plato, janela_plato,
                                avaliar_a_cada)
    hist["it_final"], hist["passos_lbfgs"] = it_final, passos
    hist["platô"] = passos < passos_lbfgs_max
    return rede_final, hist, relogio[0]
