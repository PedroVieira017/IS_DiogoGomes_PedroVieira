# Raciocínio multimodal: integração e avaliação

Este projeto reúne várias abordagens de raciocínio sobre imagens, texto e regras lógicas. O objetivo foi perceber como cada implementação funciona, confirmar os resultados que era possível reproduzir e testar a troca de datasets entre o **MMMU** e o **LogiCAM/MuSLR**.

**Responsável pela execução, integração e documentação dos testes: Pedro Vieira.**

O relatório completo, com os comandos, resultados e limitações, está disponível em [MMMU/RELATORIO_PEDRO.md](MMMU/RELATORIO_PEDRO.md).

## Componentes do projeto

- **MMMU** — benchmark multidisciplinar com perguntas de nível universitário que combinam texto e imagens.
- **LogicRAG** — transforma informação visual em factos e aplica raciocínio lógico sobre dados do KITTI.
- **LLaVA-SpaceSGG** — analisa imagens e produz descrições estruturadas de objetos, posições e relações espaciais.
- **VisuLogic** — avalia raciocínio visual através de perguntas de escolha múltipla.
- **MuSLR/LogiCAM** — combina imagens, contexto textual e regras formais para responder a perguntas de lógica simbólica.

Os módulos integrados suportam dois tipos de backend:

- `iaedu`: utiliza a API IAedu;
- `ollama`: utiliza um modelo local através do Ollama.

## Trabalho realizado

Foram concluídas as seguintes tarefas:

- execução do avaliador oficial do MMMU sobre as 900 previsões LLaVA-1.5-13B fornecidas pelos autores;
- execução do mesmo avaliador sobre as 900 previsões Qwen-VL fornecidas pelos autores;
- execução do LogicRAG com os dados KITTI pré-calculados disponíveis no repositório;
- integração da IAedu no VisuLogic e no fluxo LogiCAM;
- teste do VisuLogic e do MuSLR/LogiCAM sobre amostras oficiais;
- teste do dataset MuSLR/LogiCAM no fluxo MMMU;
- teste do dataset MMMU no fluxo LogiCAM;
- registo das previsões, avaliações e logs com identificação de Pedro Vieira.

## Resultados principais

| Experiência | Exemplos | Resultado |
|---|---:|---:|
| MMMU — previsões LLaVA-1.5-13B fornecidas | 900 | 36,7% |
| MMMU — previsões Qwen-VL fornecidas | 900 | 36,1% |
| LogicRAG/KITTI | 101 | Accuracy 0,94 / F1 0,97 |
| VisuLogic com IAedu | 5 | 60,0% |
| MuSLR com fluxo LogiCAM | 5 | 60,0% |
| Dataset MuSLR/LogiCAM no fluxo MMMU | 5 | 60,0% |
| Dataset MMMU no fluxo LogiCAM | 5 | 80,0% |

## Limitações dos testes

Os resultados devem ser interpretados de acordo com o âmbito de cada experiência:

- o dataset completo do VisuLogic tem 1.000 perguntas, mas a inferência foi feita sobre cinco exemplos;
- o MuSLR tem 1.093 registos, mas os testes com LogiCAM utilizaram cinco exemplos;
- cada teste cruzado foi realizado sobre cinco exemplos;
- o MMMU foi avaliado sobre previsões completas fornecidas pelos autores, sem voltar a executar os modelos LLaVA-1.5-13B e Qwen-VL;
- foi utilizado o backend IAedu nos testes próprios, não todos os modelos originais avaliados nos artigos.

A execução integral exigiria milhares de chamadas multimodais e bastante mais tempo. Alguns modelos dependem de APIs comerciais, enquanto os modelos abertos exigem downloads grandes, ambientes específicos e uma GPU adequada. Estas limitações estão explicadas com mais detalhe no relatório.

## Estrutura principal

```text
.
├── LogicRAG/          # raciocínio lógico sobre dados KITTI
├── LLaVA-SpaceSGG/    # análise espacial de imagens
├── MMMU/              # avaliador, integrações, resultados e relatório
├── MuSLR/             # dataset e fluxo LogiCAM
├── VisuLogic/         # benchmark de raciocínio visual
├── integra.py         # menu de execução dos módulos
└── pipeline.py        # pipeline integrado
```

Os datasets completos, imagens, ficheiros `.env` e outros dados de grande dimensão não são guardados no GitHub.

## Configuração da IAedu

Quando for utilizado o backend IAedu, é necessário um ficheiro `.env` com esta estrutura:

```env
OPENAI_API_KEY=colocar_a_chave
OPENAI_API_ENDPOINT=colocar_o_endpoint
IAEDU_CHANNEL_ID=colocar_o_channel_id
IAEDU_THREAD_ID=colocar_um_thread_id
```

As credenciais nunca devem ser enviadas para o GitHub. Dependendo do módulo executado, o `.env` deve estar numa destas pastas:

- `LogicRAG/`
- `LLaVA-SpaceSGG/`
- `VisuLogic/`
- `MuSLR/`

## Execução rápida com Docker

Na raiz do projeto, construir a imagem:

```bash
docker compose build
```

Abrir o menu integrado:

```bash
docker compose run --rm app python integra.py
```

O menu permite escolher o módulo e o backend. No VisuLogic, é possível utilizar o modo de benchmark ou fazer uma pergunta livre sobre uma imagem.

Também é possível executar diretamente uma pergunta livre no pipeline:

```bash
docker compose run --rm app python pipeline.py \
  --image "caminho/para/imagem.jpg" \
  --question "Que relação existe entre os objetos?" \
  --backend iaedu
```

Para usar o Ollama instalado no computador a partir do Docker, o serviço deve aceitar ligações em `http://host.docker.internal:11434`.

## Comandos úteis

### Avaliar as previsões LLaVA no MMMU

```powershell
Push-Location .\MMMU\mmmu

python .\main_eval_only.py `
  --output_path .\example_outputs\llava1.5_13b\total_val_output.json `
  --answer_path .\answer_dict_val.json

Pop-Location
```

### Avaliar as previsões Qwen-VL no MMMU

```powershell
Push-Location .\MMMU\mmmu

python .\main_eval_only.py `
  --output_path .\example_outputs\qwen_vl\total_val_output.json `
  --answer_path .\answer_dict_val.json

Pop-Location
```

### Fazer uma pergunta livre no VisuLogic

```bash
docker compose run --rm app python VisuLogic/VisuLogic-Eval/evaluation/manual_query.py \
  --image "caminho/para/imagem.jpg" \
  --question "O que está a acontecer nesta imagem?" \
  --model_path iaedu
```

### Executar a demonstração do VisuLogic

```bash
docker compose run --rm -w /app/VisuLogic/VisuLogic-Eval app \
  python evaluation/eval_model.py \
  --input_file demo/data.jsonl \
  --output_file outputs/iaedu_docker.jsonl \
  --model_path iaedu \
  --api_timeout 180
```

Esta demonstração contém apenas dois exemplos e serve para confirmar que o fluxo funciona do início ao fim.

### Executar uma demonstração do MuSLR/LogiCAM

```bash
docker compose run --rm app python MuSLR/muslr_agent.py \
  --dataset MuSLR/data/muslr_sample.jsonl \
  --id demo_001
```

### Analisar uma imagem com o LLaVA-SpaceSGG

```bash
docker compose run --rm app python LLaVA-SpaceSGG/dataset_pipeline/stage2/run_iaedu_image.py \
  --image LLaVA-SpaceSGG/images_real/primeira_imagem.png \
  --output-file LLaVA-SpaceSGG/resultados/teste_docker.json
```

## Ollama como alternativa local

Depois de instalar o Ollama, descarregar os modelos necessários:

```bash
ollama pull llama3
ollama pull llava
```

O `llama3` é utilizado em tarefas de texto e o `llava` em tarefas com imagens. Um modelo local evita chamadas a APIs, mas os resultados podem ser diferentes e a execução depende da capacidade do computador.

## Resultados e documentação

As evidências produzidas por Pedro Vieira encontram-se nestas pastas:

- `LogicRAG/resultados/pedro_*`
- `MMMU/resultados/pedro_*`
- `MuSLR/resultados/pedro_*`
- `VisuLogic/resultados/pedro_*`

Para consultar todos os comandos, resultados individuais e justificações metodológicas, ver [MMMU/RELATORIO_PEDRO.md](MMMU/RELATORIO_PEDRO.md).
