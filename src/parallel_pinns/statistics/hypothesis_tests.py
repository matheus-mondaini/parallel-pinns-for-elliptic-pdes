"""Estatistica entre sementes, em escala log10 (os erros variam por ordens de grandeza).

As funcoes devolvem valores em vez de imprimir (no notebook, a impressao ficava junto do calculo).
"""
import numpy as np
from scipy import stats


def resumo_log(valores):
    # Media geometrica, fator de dispersao e IC 95% calculados em log10
    l = np.log10(np.asarray(valores, dtype=float))
    media, dp = l.mean(), l.std(ddof=1)
    meia = stats.t.ppf(0.975, len(l) - 1) * dp / np.sqrt(len(l))
    return 10**media, 10**dp, (10**(media - meia), 10**(media + meia))


def hodges_lehmann(d):
    # Mediana das medias de Walsh: o estimador de posicao associado ao Wilcoxon
    d = np.asarray(d)
    return float(np.median([(d[i] + d[j]) / 2 for i in range(len(d)) for j in range(i, len(d))]))


def comparar_pareado(a, b):
    # a, b: listas de erros, mesma ordem de sementes. Razao geometrica A/B, t pareado e Wilcoxon em log10.
    la, lb = np.log10(a), np.log10(b)
    d = la - lb
    return {"razao": float(10**d.mean()), "pior_em": int(np.sum(d > 0)), "n": len(d),
            "p_t": float(stats.ttest_rel(la, lb).pvalue), "p_wilcoxon": float(stats.wilcoxon(la, lb).pvalue)}


def tost_log(a, b, margem=(0.8, 1.25)):
    # Teste de equivalencia (TOST) pareado em log10: A e B sao equivalentes se a razao geometrica
    # A/B esta dentro da margem fixada ANTES do experimento. Equivale a IC 90% dentro da margem.
    d = np.log10(a) - np.log10(b)
    lo, hi = np.log10(margem[0]), np.log10(margem[1])
    p_inf = stats.ttest_1samp(d, lo, alternative="greater").pvalue
    p_sup = stats.ttest_1samp(d, hi, alternative="less").pvalue
    meia = stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return 10**d.mean(), (10**(d.mean() - meia), 10**(d.mean() + meia)), max(p_inf, p_sup)


def comparar_razao(a, b):
    # Razao geometrica A/B com IC 90%, Hodges-Lehmann e Wilcoxon bilateral, tudo sobre d = log10 A - log10 B
    # (teste da ordem das somas e teste confirmatorio da H2 na secao 13 do notebook).
    d = np.log10(a) - np.log10(b)
    meia = stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"razao": float(10**d.mean()), "ic90": (float(10**(d.mean() - meia)), float(10**(d.mean() + meia))),
            "hodges_lehmann": float(10**hodges_lehmann(d)), "pior_em": int(np.sum(d > 0)), "n": len(d),
            "p_wilcoxon": float(stats.wilcoxon(d).pvalue)}


def desfecho_h2(p, hl_log, alfa):
    # Regra fixada antes do teste confirmatorio: Wilcoxon decide, o Hodges-Lehmann (em log10) da o sentido
    if p >= alfa:
        return "sem diferença detectável"
    return "H2 confirmada" if hl_log < 0 else "H2 refutada"


def falhou(m):
    # Definicao fixada antes do teste confirmatorio: perdeu a solucao ou ignorou o modo alto
    return m["L2"] > 0.1 or m["E_n"] > 0.5
