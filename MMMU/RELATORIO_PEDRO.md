# Execução dos benchmarks e testes cruzados

**Execução e integração:** Pedro Vieira
**Data:** 21 de julho de 2026
**Branch:** `mmmu`

## 1. Objetivo

Este trabalho executa e valida as implementações MMMU, LogicRAG, VisuLogic e MuSLR/LogiCAM, comparando os resultados possíveis com os valores publicados.

Também realiza os dois testes cruzados pedidos:

1. Dataset MuSLR/LogiCAM executado através do fluxo MMMU.
2. Dataset MMMU executado através do fluxo LogiCAM.

## 2. Identificação

Os pedidos enviados à IAedu identificam explicitamente:

```json
{
  "name": "Pedro Vieira"
}
```

A identificação foi corrigida nos clientes e integrações utilizados.

## 3. Ambiente

Foi utilizado o ambiente Conda `mmmu`, com Python 3.10.20.

A máquina possui aproximadamente 16 GB de RAM e não possui GPU NVIDIA. Por esse motivo, não foi possível gerar novamente todas as previsões com os modelos originais de grande dimensão.

A IAedu foi utilizada como backend multimodal nas experiências controladas.

## 4. Datasets oficiais

### MMMU

- Validação: 900 perguntas.
- 30 disciplinas.
- Dataset obtido através do Hugging Face.

### VisuLogic

- 1.000 perguntas oficiais.
- 1.000 imagens oficiais.
- As 1.000 correspondências entre perguntas e imagens foram verificadas.

### MuSLR

- 1.093 registos no `metadata.csv`.
- 1.091 imagens.
- Inclui escolha múltipla e avaliação `True/False/Unknown`.

## 5. Reprodução do MMMU

O avaliador oficial foi executado sobre as previsões LLaVA-1.5-13B fornecidas pelos autores:

```powershell
python .\main_eval_only.py `
  --output_path .\example_outputs\llava1.5_13b\total_val_output.json `
  --answer_path .\answer_dict_val.json
```

Resultado:

| Origem | Exemplos | Accuracy |
|---|---:|---:|
| Artigo/leaderboard, LLaVA-1.5-13B | 900 | 36,4% |
| Avaliação local das previsões fornecidas | 900 | 36,7% |

A diferença é de 0,3 pontos percentuais. O resultado é próximo, mas não exatamente igual. As previsões avaliadas foram fornecidas pelos autores e não foram novamente geradas nesta máquina.

## 6. Reprodução do LogicRAG

Foi executado o código oficial sobre a knowledge base e trajetórias KITTI pré-calculadas:

```powershell
python .\inference_in_kb.py `
  --csv ..\kitti_questions\kitti_que.csv `
  --fol_trans_csv .\translated_queries\question_query_kitti_llama33.csv `
  --kb_dir ..\..\LogicRAG_Data\precomputed_knowledge_base\kb_out_kitti `
  --tracker_dir ..\..\LogicRAG_Data\tracker_trajectories\track_out_kitti `
  --output ..\..\resultados\pedro_logicrag_kitti.csv
```

Resultados:

| Métrica | Artigo | Execução local |
|---|---:|---:|
| Accuracy | 0,91 | 0,94 |
| F1 | 0,95 | 0,97 |

O artigo descreve 100 perguntas, enquanto o ficheiro atualmente disponibilizado pelo repositório contém 101. Os valores locais são superiores, mas não reproduzem exatamente os valores publicados.

## 7. VisuLogic

O dataset oficial completo foi descarregado e validado:

- Perguntas: 1.000
- Imagens oficiais referenciadas: 1.000
- Imagens em falta: 0

Foi executada uma amostra de cinco perguntas, abrangendo cinco categorias diferentes, através da IAedu.

| Exemplos | Corretos | Accuracy |
|---:|---:|---:|
| 5 | 3 | 60,0% |

Resultados individuais:

| ID | Previsão | Correta | Resultado |
|---|---:|---:|---|
| `00000` | C | A | Errado |
| `00001` | D | D | Correto |
| `00002` | C | C | Correto |
| `00003` | C | B | Errado |
| `00004` | D | D | Correto |

Este resultado não é diretamente comparável com a tabela do artigo, porque foi utilizado o backend IAedu e não um dos modelos originais avaliados no artigo.

## 8. MuSLR com fluxo LogiCAM

Foi executada uma amostra de cinco exemplos oficiais do MuSLR com o fluxo integrado LogiCAM:

1. seleção de premissas;
2. identificação do tipo de raciocínio;
3. aplicação das regras lógicas;
4. geração da resposta final.

| Exemplos | Corretos | Accuracy |
|---:|---:|---:|
| 5 | 3 | 60,0% |

Resultados individuais:

| ID | Previsão | Correta | Resultado |
|---|---:|---:|---|
| `flickr30k_7760` | A | B | Errado |
| `flickr30k_6121` | D | D | Correto |
| `coco_2031` | True | True | Correto |
| `coco_6860` | False | False | Correto |
| `rvl_652` | Unknown | True | Errado |

O artigo refere 46,8% para o melhor baseline GPT-4.1 e uma melhoria de 14,13 pontos percentuais através do LogiCAM. A presente experiência utiliza IAedu e apenas cinco exemplos, pelo que não é uma reprodução diretamente comparável desses valores.

## 9. Teste cruzado MuSLR/LogiCAM para MMMU

Cinco exemplos oficiais do MuSLR foram processados com prompting direto no estilo MMMU.

| Dataset | Método | Exemplos | Corretos | Accuracy |
|---|---|---:|---:|---:|
| MuSLR/LogiCAM | MMMU-style | 5 | 3 | 60,0% |

Resultados:

| ID | Previsão | Correta | Resultado |
|---|---:|---:|---|
| `flickr30k_7760` | A | B | Errado |
| `flickr30k_6121` | D | D | Correto |
| `coco_2031` | A | A | Correto |
| `coco_6860` | B | B | Correto |
| `rvl_652` | C | A | Errado |

## 10. Teste cruzado MMMU para LogiCAM

Cinco perguntas da disciplina `Accounting`, do conjunto `validation` do MMMU, foram processadas com o fluxo LogiCAM.

| Dataset | Método | Exemplos | Corretos | Accuracy |
|---|---|---:|---:|---:|
| MMMU validation / Accounting | LogiCAM-style | 5 | 4 | 80,0% |

Resultados:

| ID | Previsão | Correta | Resultado |
|---|---:|---:|---|
| `validation_Accounting_1` | B | B | Correto |
| `validation_Accounting_2` | C | C | Correto |
| `validation_Accounting_3` | B | B | Correto |
| `validation_Accounting_4` | D | D | Correto |
| `validation_Accounting_5` | C | B | Errado |

O avaliador oficial do MMMU confirmou:

- `Accounting`: 5 exemplos, accuracy 0,8
- `Overall`: 5 exemplos, accuracy 0,8

## 11. Resumo

| Experiência | Exemplos | Resultado |
|---|---:|---:|
| MMMU, previsões LLaVA fornecidas | 900 | 36,7% |
| LogicRAG/KITTI oficial | 101 | Accuracy 0,94 / F1 0,97 |
| VisuLogic com IAedu | 5 | 60,0% |
| MuSLR com fluxo LogiCAM | 5 | 60,0% |
| MuSLR/LogiCAM → MMMU | 5 | 60,0% |
| MMMU → LogiCAM | 5 | 80,0% |

## 12. Conclusão

Os resultados publicados não foram todos reproduzidos exatamente:

- MMMU: resultado próximo, com diferença de 0,3 pontos percentuais.
- LogicRAG: valores superiores aos publicados, usando 101 perguntas em vez das 100 descritas no artigo.
- VisuLogic e LogiCAM: integrações executadas sobre amostras oficiais com IAedu, mas sem comparação direta com os modelos originais.

Os dois testes cruzados pedidos foram implementados e executados nos dois sentidos.

## 13. Limitações

As experiências com IAedu utilizam amostras de cinco exemplos e não representam a accuracy integral dos benchmarks.

A reprodução integral exigiria:

- milhares de chamadas multimodais;
- acesso aos modelos originais;
- GPU adequada ou serviços pagos;
- repetição exata das configurações e versões usadas pelos autores.

## 14. Evidências

As previsões, respostas e logs encontram-se em:

- `LogicRAG/resultados/pedro_*`
- `VisuLogic/resultados/pedro_*`
- `MMMU/resultados/pedro_*`
- `MuSLR/resultados/pedro_*`
