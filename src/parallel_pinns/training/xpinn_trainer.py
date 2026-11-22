"""Treino do XPINN: sequencial ou distribuido por replicacao de parametros (subdominios entre processos)."""
import torch
import torch.distributed as dist

from ..analysis.metrics import avaliar
from ..models.networks import largura_iso_parametro
from ..models.xpinn import XPINN, agrupar_por_subdominio, pontos_interface
from ..physics import geometry
from ..physics.operators import laplaciano
from .optimizer import otimizar
from .trainer import SEMENTE


def treinar_xpinn(prob, M, largura=None, contorno="fraco", epocas_adam=3000, passos_lbfgs_max=100, iter_por_passo=20,
                  tolerancia_plato=0.01, janela_plato=10, N_f=2000, N_b=400, N_I=50, lr=1e-3, camadas_ocultas=4,
                  semente=SEMENTE, lam_pde="auto", lam_bc=1.0, lam_i=1.0, fluxo=True, avaliar_a_cada=100, imprimir=True,
                  distribuido=False):
    # distribuido=True: parametros replicados em todos os processos; cada processo calcula a EDP e o contorno dos
    # SEUS subdominios e uma parte dos segmentos de interface, normalizados pelos totais GLOBAIS. A soma dos
    # gradientes entre processos (all-reduce) e o gradiente da perda global: Adam e L-BFGS ficam identicos.
    torch.manual_seed(semente)
    if largura is None:
        largura = largura_iso_parametro(M, camadas_ocultas=camadas_ocultas)
    rede = XPINN(M, largura, camadas_ocultas, G=prob["G"] if contorno == "exato" else None)
    S = rede.S
    rank, p = (dist.get_rank(), dist.get_world_size()) if distribuido else (0, 1)
    assert S % p == 0, "o numero de subdominios M^2 precisa ser multiplo do numero de processos"
    meus = torch.arange(rank * S // p, (rank + 1) * S // p)        # subdominios deste processo

    # MESMOS pontos globais da monolitica, apenas separados por subdominio
    xy_f = geometry.pontos_colocacao(N_f, semente)
    f_global = prob["f"](xy_f[:, 0:1], xy_f[:, 1:2])
    if lam_pde == "auto":
        escala = torch.mean(f_global**2).item()
        lam_pde = 1.0 / escala if escala > 0 else 1.0

    lote_f, masc_f = agrupar_por_subdominio(xy_f, rede.subdominio(xy_f), S)
    n_f_total = masc_f.sum()                                       # N_f global
    lote_f, masc_f = lote_f[meus], masc_f[meus]
    xy_I, ka, kb, normal = pontos_interface(M, N_I)
    nseg_total = xy_I.shape[0]
    if distribuido:                                                # segmentos de interface: um a cada p
        seg = torch.arange(rank, nseg_total, p)
        xy_I, ka, kb, normal = xy_I[seg], ka[seg], kb[seg], normal[seg]
    # Direcao normal de cada ponto como mascara one-hot: (grad * sel).sum(1) = derivada normal.
    # (Equivale a gather, mas o backward do gather usa scatter_add, que nao e deterministico na GPU.)
    sel_normal = torch.nn.functional.one_hot(normal.repeat_interleave(N_I), 2).to(torch.float32)
    ks = meus

    # Folhas "achatadas" para usar o MESMO laplaciano(u, xy) da monolitica
    xy_f_flat = lote_f.reshape(-1, 2).clone().requires_grad_(True)
    f_flat = prob["f"](xy_f_flat[:, 0:1], xy_f_flat[:, 1:2]).detach()
    xy_I_flat = xy_I.reshape(-1, 2).clone().requires_grad_(True)
    nseg = xy_I.shape[0]
    # Media sobre os pontos de interface GLOBAIS (no sequencial, a media de sempre)
    media_I = (lambda v: v.sum() / (nseg_total * N_I)) if distribuido else torch.mean

    if contorno == "fraco":
        xy_b = geometry.pontos_contorno(N_b)
        lote_b, masc_b = agrupar_por_subdominio(xy_b, rede.subdominio(xy_b), S)
        n_b_total = masc_b.sum()
        lote_b, masc_b = lote_b[meus], masc_b[meus]
        g_b = prob["u"](lote_b[..., 0:1], lote_b[..., 1:2]).detach()

    def perdas(registro=False):
        # Interior: soma sobre os pontos reais / N_f global = mesma L_PDE da monolitica
        u_f = rede.sub_redes(xy_f_flat.view(len(meus), -1, 2), ks).reshape(-1, 1)
        residuo = (-laplaciano(u_f, xy_f_flat) - f_flat) * masc_f.reshape(-1, 1)
        loss_pde = (residuo**2).sum() / n_f_total

        if contorno == "fraco":
            u_b = rede.sub_redes(lote_b, ks)
            loss_bc = (((u_b - g_b) * masc_b)**2).sum() / n_b_total
        else:
            loss_bc = torch.zeros(())

        # Interface: os dois lados avaliados nos mesmos pontos.
        # Convencao de escala: o salto do residuo e normalizado como a L_PDE (x lam_pde);
        # os saltos de u e de du/dn ficam em unidades naturais (as solucoes tem amplitude de ordem 1).
        # Assim o peso da interface em relacao a EDP nao depende de n.
        loss_i = torch.zeros(())
        if nseg > 0:
            xy_I_view = xy_I_flat.view(nseg, N_I, 2)
            u_a = rede.sub_redes(xy_I_view, ka).reshape(-1, 1)
            u_c = rede.sub_redes(xy_I_view, kb).reshape(-1, 1)
            salto_u = media_I((u_a - u_c)**2)
            salto_res = media_I((laplaciano(u_a, xy_I_flat) - laplaciano(u_c, xy_I_flat))**2)  # f se cancela
            loss_i = salto_u + lam_pde * salto_res
            if fluxo:
                grad_a = torch.autograd.grad(u_a, xy_I_flat, torch.ones_like(u_a), create_graph=True)[0]
                grad_c = torch.autograd.grad(u_c, xy_I_flat, torch.ones_like(u_c), create_graph=True)[0]
                salto_fluxo = media_I((((grad_a - grad_c) * sel_normal).sum(1))**2)
                loss_i = loss_i + salto_fluxo

        loss = lam_pde * loss_pde + lam_i * loss_i + lam_bc * loss_bc
        return loss, loss_pde, loss_bc, loss_i

    hist = {"it": [], "loss": [], "pde": [], "bc": [], "int": [], "aval_it": [], "aval_tempo": [], "L2": [], "E_n": [],
            "E_1": [], "fim_adam": epocas_adam, "lam_pde": lam_pde, "lam_bc": lam_bc, "lam_i": lam_i, "largura": largura,
            "fluxo": fluxo, "contorno": contorno, "parametros": sum(p.numel() for p in rede.parameters())}
    relogio = [0.0]

    def registrar(it, valores, forcar=False):
        hist["it"].append(it)
        hist["loss"].append(valores[0])
        hist["pde"].append(valores[1])
        hist["bc"].append(valores[2])
        hist["int"].append(valores[3])
        if forcar:
            m = avaliar(rede, prob)
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
            print(f"[{prob['nome']} XPINN M={M}] {fase} it {it:5d} | loss {valores[0]:.2e} | PDE {valores[1]:.2e} | INT {valores[3]:.2e} | L2 {m['L2']:.2e}{extra}")

    def soma_global(valor):
        # Cada processo tem uma parte da perda global: o valor global e a SOMA entre processos
        if not distribuido:
            return valor
        v = valor.detach().clone()
        dist.all_reduce(v)
        return v

    def somar_gradientes():
        # Um all-reduce so, com todos os gradientes num vetor (como os "buckets" do DDP): menos latencia que um por tensor
        if distribuido:
            grads = [prm.grad for prm in rede.parameters()]
            plano = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(plano)
            for g, parte in zip(grads, torch.split(plano, [g.numel() for g in grads])):
                g.copy_(parte.view_as(g))

    it_final, passos = otimizar(list(rede.parameters()), perdas, registrar, mostrar, relogio, soma_global,
                                epocas_adam, passos_lbfgs_max, iter_por_passo, lr, tolerancia_plato, janela_plato,
                                avaliar_a_cada, somar_gradientes)
    hist["it_final"], hist["passos_lbfgs"] = it_final, passos
    hist["platô"] = passos < passos_lbfgs_max
    return rede, hist, relogio[0]
