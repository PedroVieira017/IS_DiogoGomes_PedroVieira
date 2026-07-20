# Integração MMMU e testes cruzados com LogiCAM

**Execução e integração:** Pedro Vieira  
**Data:** 20 de julho de 2026  
**Branch:** `mmmu`

## 1. Objetivo

Esta implementação integra o benchmark oficial MMMU no projeto e realiza os dois testes cruzados pedidos:

1. Executar o dataset MuSLR/LogiCAM através da implementação MMMU.
2. Executar o dataset MMMU através do método de raciocínio LogiCAM.

## 2. Fontes oficiais

### MMMU

- Artigo: *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*
- Código: https://github.com/MMMU-Benchmark/MMMU
- Dataset: https://huggingface.co/datasets/MMMU/MMMU

### MuSLR / LogiCAM

- Artigo: *Multimodal Symbolic Logical Reasoning*
- Código: https://github.com/Aiden0526/MuSLR
- Dataset: https://huggingface.co/datasets/Aiden0526/MuSLR

## 3. Ambiente

Foi criado o ambiente Conda `mmmu` com Python 3.10.20.

Dependências principais:

- `datasets`
- `Pillow`
- `numpy`
- `pyyaml`
- `requests`
- `python-dotenv`
- `openai`
- `tabulate`
- `tqdm`

Não foi detetada uma GPU NVIDIA. Por esse motivo, não foi possível gerar novamente as previsões com o modelo LLaVA-1.5-13B. Foi utilizada a API IAedu como backend multimodal, seguindo o método das restantes integrações do projeto.

## 4. Ficheiros implementados

- `MMMU/run_iaedu_mmmu.py`: executa perguntas oficiais do MMMU através da IAedu.
- `MMMU/run_muslr_with_mmmu.py`: executa o dataset MuSLR com prompting direto no estilo MMMU.
- `MuSLR/run_mmmu_with_logicam.py`: executa perguntas MMMU com raciocínio inspirado no LogiCAM.
- `MMMU/resultados/`: contém previsões, detalhes e logs de execução.
- `MuSLR/resultados/pedro_*`: contém as evidências do teste MMMU com LogiCAM.

## 5. Dataset MuSLR

O dataset oficial MuSLR foi descarregado do Hugging Face.

Verificação local:

- Registos no `metadata.csv`: 1.093
- Imagens: 1.091

O dataset está excluído do Git através da regra `MuSLR/MuSLR-Dataset/`.

## 6. Validação do avaliador oficial MMMU

O avaliador oficial foi executado com as previsões LLaVA-1.5-13B disponibilizadas pelos autores.

Comando principal:

`python main_eval_only.py --output_path example_outputs/llava1.5_13b/total_val_output.json --answer_path answer_dict_val.json`

Resultado:

| Avaliação | Exemplos | Accuracy |
|---|---:|---:|
| Previsões LLaVA incluídas no repositório oficial | 900 | 36,7% |

Este teste confirma que o avaliador oficial funciona no ambiente utilizado. As previsões foram fornecidas pelos autores e não foram geradas novamente neste computador.

## 7. MMMU executado através da IAedu

Foram executados três exemplos da categoria `Accounting`, do conjunto `validation`.

Resultado:

| Dataset | Categoria | Exemplos | Corretos | Accuracy |
|---|---|---:|---:|---:|
| MMMU validation | Accounting | 3 | 3 | 100% |

O resultado foi confirmado pelo avaliador oficial:

- `Accounting`: 3 exemplos, accuracy 1.0
- `Overall`: 3 exemplos, accuracy 1.0

## 8. Teste cruzado MuSLR/LogiCAM para MMMU

Neste teste, exemplos oficiais do MuSLR foram processados através de prompting direto de escolha múltipla no estilo MMMU.

Resultado:

| Dataset de entrada | Método | Exemplos | Corretos | Accuracy |
|---|---|---:|---:|---:|
| MuSLR/LogiCAM | MMMU-style direct prompting | 3 | 2 | 66,7% |

Resultados individuais:

| ID | Previsão | Correta | Resultado |
|---|---:|---:|---|
| `flickr30k_7760` | A | B | Errado |
| `flickr30k_6121` | D | D | Correto |
| `coco_2031` | A | A (`True`) | Correto |

O adaptador suporta os dois tipos de resposta existentes no MuSLR:

- escolha múltipla, com letras;
- avaliação lógica, com `True`, `False` ou `Unknown`.

## 9. Teste cruzado MMMU para LogiCAM

Neste teste, exemplos oficiais do MMMU foram processados com um fluxo inspirado no LogiCAM:

1. seleção das premissas relevantes;
2. identificação do tipo de raciocínio;
3. raciocínio lógico ou matemático estruturado;
4. conclusão final.

Resultado:

| Dataset de entrada | Método | Exemplos | Corretos | Accuracy |
|---|---|---:|---:|---:|
| MMMU validation / Accounting | LogiCAM-style reasoning | 3 | 3 | 100% |

O resultado foi confirmado pelo avaliador oficial:

- `Accounting`: 3 exemplos, accuracy 1.0
- `Overall`: 3 exemplos, accuracy 1.0

## 10. Resumo

| Experiência | Exemplos | Accuracy |
|---|---:|---:|
| Avaliação das previsões LLaVA fornecidas pelos autores | 900 | 36,7% |
| MMMU com IAedu | 3 | 100% |
| MuSLR/LogiCAM para MMMU | 3 | 66,7% |
| MMMU para LogiCAM | 3 | 100% |

## 11. Limitações

Os testes através da IAedu e os testes cruzados utilizaram amostras controladas de três exemplos. Demonstram que os adaptadores e os dois sentidos de execução funcionam, mas não representam a accuracy integral dos benchmarks.

Uma avaliação completa exigiria pelo menos:

- 900 chamadas para o conjunto validation do MMMU;
- 1.093 chamadas para o MuSLR;
- chamadas adicionais caso os módulos LogiCAM fossem executados separadamente.

O teste MMMU para LogiCAM utiliza o fluxo integrado neste projeto, inspirado na framework LogiCAM. Não constitui uma repetição integral do notebook batch oficial.

## 12. Evidências

As previsões, respostas completas e logs encontram-se em:

- `MMMU/resultados/`
- `MuSLR/resultados/pedro_*`

Os pedidos enviados à IAedu identificam explicitamente:

`"name": "Pedro Vieira"`