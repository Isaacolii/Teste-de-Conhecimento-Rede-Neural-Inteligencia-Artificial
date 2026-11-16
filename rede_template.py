"""
Teste: implementação de uma rede neural já treinada (propagação direta)
Matemática Computacional - Prof. Dr. Alexsandro O. A.

Nome(s): Isaac Oliveira
"""

import json
import numpy as np
import pandas as pd


def carregar_rede(arquivo="pesos_rede.json"):
    with open(arquivo, encoding="utf-8") as f:
        cfg = json.load(f)
    for chave in ["W1", "b1", "W2", "b2", "W3", "b3", "normalizacao_divisor"]:
        cfg[chave] = np.array(cfg[chave], dtype=float)
    return cfg


def normalizar(entrega_bruta, divisores):
    return np.asarray(entrega_bruta, dtype=float) / divisores


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def camada(W, b, x, ativacao):
    z = W @ x + b
    return z, ativacao(z)


def propagar(rede, x):
    z1, h1 = camada(rede["W1"], rede["b1"], x, relu)
    z2, h2 = camada(rede["W2"], rede["b2"], h1, tanh)
    z3, p = camada(rede["W3"], rede["b3"], h2, softmax)
    return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "z3": z3, "p": p}


def classificar(rede, x):
    p = propagar(rede, x)["p"]
    i = int(np.argmax(p))
    return rede["classes"][i], p[i]


def _confere(nome, obtido, esperado, tol=1e-3):
    ok = np.allclose(np.asarray(obtido, float), np.asarray(esperado, float), atol=tol)
    print(f"[{'OK' if ok else 'FALHOU'}] {nome}")
    if not ok:
        print("    esperado:", np.round(esperado, 4), "\n    obtido:  ", np.round(obtido, 4))


def validar(rede):
    print("=" * 60 + "\nVALIDAÇÃO: x = (0.60, 0.40, 0.50, 0.20)\n" + "=" * 60)
    x = np.array([0.60, 0.40, 0.50, 0.20])
    r = propagar(rede, x)
    _confere("z1", r["z1"], [1.602, -0.417, 1.433, -1.171, -0.521, 1.274])
    _confere("h1", r["h1"], [1.602, 0.000, 1.433, 0.000, 0.000, 1.274])
    _confere("z2", r["z2"], [2.5760, -0.9262, 4.1550, 0.0951])
    _confere("h2", r["h2"], [0.9885, -0.7288, 0.9995, 0.0948])
    _confere("z3", r["z3"], [-3.5755, 4.4592, -1.0141])
    _confere("p", r["p"], [0.0003, 0.9955, 0.0042])
    classe, prob = classificar(rede, x)
    print(f"Classe prevista: {classe} ({100 * prob:.2f}%)  -> esperado: SEDEX (99.55%)")


if __name__ == "__main__":
    rede = carregar_rede()
    validar(rede)

    np.set_printoptions(precision=4, suppress=True)
    print("\nNúmero de parâmetros:",
          sum(rede[k].size for k in ["W1", "b1", "W2", "b2", "W3", "b3"]))

    x = np.array([0.50, 0.70, 0.40, 0.30])
    r = propagar(rede, x)
    print(f"\nx = {x}")
    for chave in ["z1", "h1", "z2", "h2", "z3", "p"]:
        print(f"  {chave} = {r[chave]}")
    classe, prob = classificar(rede, x)
    print(f"  classe = {classe} ({100 * prob:.2f}%)")

    bruta = [150, 3, 0.95, 1200]
    classe, prob = classificar(rede, normalizar(bruta, rede["normalizacao_divisor"]))
    print(f"\nEncomenda {bruta}: {classe} ({100 * prob:.2f}%)")

    df = pd.read_csv("entregas_teste_rede.csv")
    print("\nEntregas de teste:")
    acertos = 0
    for _, linha in df.iterrows():
        bruta = [linha.distancia_km, linha.peso_kg, linha.urgencia, linha.valor_declarado]
        classe, prob = classificar(rede, normalizar(bruta, rede["normalizacao_divisor"]))
        acertos += classe == linha.modalidade_real
        print(f"  entrega {linha.id_entrega:2d}: prevista = {classe:8s} ({100 * prob:6.2f}%)"
              f" | real = {linha.modalidade_real}")
    print(f"Acurácia: {100 * acertos / len(df):.0f}%")
