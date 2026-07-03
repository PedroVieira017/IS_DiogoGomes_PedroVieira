# Amostra de demonstração do VisuLogic

Esta pasta contém um `data.jsonl` mínimo (2 exemplos) e as imagens em `img/`,
apenas para demonstrar que o pipeline de avaliação corre dentro do Docker sem
ser preciso descarregar o benchmark completo.

> **Importante:** o código do `eval_model.py` resolve o caminho das imagens a
> partir da pasta do ficheiro, substituindo o nome `data.jsonl`. Por isso o
> ficheiro **tem** de se chamar exatamente `data.jsonl` e as imagens ficam em
> `img/` relativas a esta pasta.

Estes exemplos **não** representam o benchmark real (não são puzzles de
raciocínio visual) — servem só para confirmar o fluxo ponta-a-ponta. Para uma
avaliação real, descarregar o dataset oficial de
[huggingface.co/datasets/VisuLogic/VisuLogic](https://huggingface.co/datasets/VisuLogic/VisuLogic)
e colocar `data.jsonl` + `images/` em `VisuLogic/VisuLogic-Eval/`.
