# Teste: rede neural já treinada (propagação direta)

**Matemática Computacional** · Prof. Dr. Alexsandro O. A. · Aluno: Isaac Oliveira

Propagação direta de uma rede 4 → 6 (ReLU) → 4 (tanh) → 3 (softmax) feita só com NumPy, seguindo o
[enunciado](https://github.com/alexsandro-fap/test-neuralnetwork). A rede sugere a modalidade de envio
(PAC, SEDEX ou SEDEX 10) a partir de distância, peso, urgência e valor declarado.

```bash
python rede_template.py
```

## Resultados

O exemplo de validação do enunciado passa em todas as etapas (z1, h1, z2, h2, z3 e p): SEDEX com 99.55%.

| Item | Resultado |
| --- | --- |
| Parâmetros (pesos + vieses) | 73 |
| x = (0.50, 0.70, 0.40, 0.30) | PAC, com 95.86% |
| Encomenda de 150 km, 3 kg, urgência 0.95, R$ 1200 | SEDEX 10, com 95.77% |
| Acurácia nas 10 entregas de teste | 80% (erra as entregas 5 e 6: prevê PAC, eram SEDEX) |
