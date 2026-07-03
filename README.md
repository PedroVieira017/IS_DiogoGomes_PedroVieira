# IS_DiogoGomes_PedroVieira

Projeto de demonstracao com tres modulos de raciocinio visual:

- **LogicRAG**: usa uma base de conhecimento pre-computada do KITTI, traduz factos em First-Order Logic para linguagem natural e envia o contexto para um agente IAedu.
- **LLaVA-SpaceSGG**: envia uma imagem real para a API IAedu e recebe uma descricao estruturada com objetos, caixas, relacoes espaciais, camadas de profundidade e perguntas/respostas comparativas.
- **VisuLogic**: integra o codigo oficial de avaliacao do benchmark VisuLogic e envia imagem + pergunta para a API IAedu ou para um modelo local via Ollama.

Os fluxos LogicRAG, LLaVA-SpaceSGG e VisuLogic podem correr de duas formas:

- `iaedu`: envia o pedido para a API IAedu na nuvem.
- `ollama`: corre um modelo local atraves do Ollama (`llama3` para texto, `llava` para visao), sem internet nem chave de API.

O objetivo do repositorio e deixar um fluxo reproduzivel para demonstracao, screenshots e apresentacao.

## Estrutura

```text
.
+-- LogicRAG/
|   +-- parse_kb_to_csv.py
|   +-- driving_agent.py
|   +-- environment.yml
|   +-- README.md
+-- LLaVA-SpaceSGG/
|   +-- dataset_pipeline/stage2/run_iaedu_image.py
|   +-- dataset_pipeline/stage2/iaedu_client.py
|   +-- dataset_pipeline/stage2/visualize_layers.py
|   +-- images_real/
|   +-- README.md
+-- VisuLogic/
|   +-- README.md
|   +-- resultados/
|   +-- VisuLogic-Eval/
|       +-- models/iaedu_api.py
|       +-- models/ollama_vision.py
+-- .gitignore
```

## Requisitos

O caminho recomendado é o **Docker** — corre os quatro módulos num único contentor, sem instalar Python, Conda nem pacotes de IA localmente (ver [Execução via Docker](#execução-via-docker)).

- Docker + Docker Compose (caminho principal)
- Credenciais da API IAedu (ficheiros `.env` por módulo) **ou** Ollama instalado no host para inferência local
- Dados pre-computados do LogicRAG em `LogicRAG/LogicRAG_Data/`

Alternativamente, é possível correr sem Docker com Miniconda/Anaconda e um ambiente `lrag` (instruções em PowerShell nas secções de demo mais abaixo).

Os ficheiros `.env`, dados grandes, zips e outputs locais nao devem ser enviados para o GitHub.

## Configuracao das Chaves

Criar um ficheiro `.env` dentro de `LogicRAG/`:

```env
OPENAI_API_KEY=colocar_a_chave_iaedu
OPENAI_API_ENDPOINT=colocar_o_endpoint_stream_iaedu
IAEDU_CHANNEL_ID=colocar_o_channel_id
IAEDU_THREAD_ID=colocar_um_thread_id
```

Criar um ficheiro `.env` dentro de `LLaVA-SpaceSGG/`:

```env
OPENAI_API_KEY=colocar_a_chave_iaedu
OPENAI_API_ENDPOINT=colocar_o_endpoint_stream_iaedu
IAEDU_CHANNEL_ID=colocar_o_channel_id
IAEDU_THREAD_ID=colocar_um_thread_id
```

Criar tambem um ficheiro `.env` dentro de `VisuLogic/`:

```env
OPENAI_API_KEY=colocar_a_chave_iaedu
OPENAI_API_ENDPOINT=colocar_o_endpoint_stream_iaedu
IAEDU_CHANNEL_ID=colocar_o_channel_id
IAEDU_THREAD_ID=colocar_um_thread_id
```

Para confirmar que os ficheiros existem sem mostrar as chaves:

```powershell
Get-Content .\LogicRAG\.env | ForEach-Object { ($_ -split '=')[0] + '=***' }
Get-Content .\LLaVA-SpaceSGG\.env | ForEach-Object { ($_ -split '=')[0] + '=***' }
Get-Content .\VisuLogic\.env | ForEach-Object { ($_ -split '=')[0] + '=***' }
```

## Execução via Docker 

Todos os quatro módulos do projeto (**LogicRAG**, **LLaVA-SpaceSGG**, **VisuLogic** e **MuSLR**) podem ser executados dentro de um único contentor Docker, sem a necessidade de configurar ambientes Conda locais ou instalar pacotes de IA pesados.

### 1. Construir a Imagem Docker
A partir da raiz do projeto, execute:
```bash
docker compose build
```

### 2. Configuração do Backend (IAedu ou Ollama)
O contentor utiliza as credenciais nos ficheiros `.env` locais de cada módulo. 
- **Para IAedu (Nuvem):** Garanta que os ficheiros `.env` estão configurados nas pastas respetivas de cada módulo com as chaves corretas.
- **Para Ollama (Local no Host):** Para aceder ao Ollama instalado no seu computador a partir de dentro do Docker, os scripts utilizam `http://host.docker.internal:11434`.
  > [!IMPORTANT]
  > Para que o Ollama no seu computador aceite ligações vindas do Docker, deve configurá-lo para escutar em todas as interfaces. 
  > - No macOS/Linux, arranque o Ollama no terminal com: `OLLAMA_HOST=0.0.0.0 ollama serve`
  > - No Windows, defina a variável de ambiente de sistema `OLLAMA_HOST` para `0.0.0.0` e reinicie o Ollama.

### 3. Comandos para Executar cada Módulo

#### A. LogicRAG
> [!NOTE]
> Estes scripts esperam ser executados a partir da pasta `LogicRAG/` (usam caminhos relativos como `LogicRAG_Data/...`), por isso o comando fixa o working-dir com `-w /app/LogicRAG`. Requer os dados pré-computados em `LogicRAG/LogicRAG_Data/precomputed_knowledge_base/kb_out_kitti/`.

1. Traduzir factos em First-Order Logic para linguagem natural:
   ```bash
   docker compose run --rm -w /app/LogicRAG app python parse_kb_to_csv.py
   ```
2. Executar inferência (IAedu ou Ollama):
   ```bash
   docker compose run --rm -w /app/LogicRAG app python driving_agent.py
   ```

#### B. LLaVA-SpaceSGG
1. Executar análise da imagem:
   ```bash
   docker compose run --rm app python LLaVA-SpaceSGG/dataset_pipeline/stage2/run_iaedu_image.py --image LLaVA-SpaceSGG/images_real/primeira_imagem.png --output-file LLaVA-SpaceSGG/resultados/teste_docker.json
   ```
2. Gerar visualização das camadas de profundidade:
   ```bash
   docker compose run --rm app python LLaVA-SpaceSGG/dataset_pipeline/stage2/visualize_layers.py --result-file LLaVA-SpaceSGG/resultados/teste_docker.json --output-file LLaVA-SpaceSGG/resultados/visualizacoes/teste_layers_docker.png
   ```

#### C. VisuLogic
Executar a avaliação com a **amostra de demonstração** já incluída no repositório (2 exemplos + imagens em `VisuLogic/VisuLogic-Eval/demo/`), sem precisar de descarregar o benchmark completo:
```bash
docker compose run --rm -w /app/VisuLogic/VisuLogic-Eval app python evaluation/eval_model.py --input_file demo/data.jsonl --output_file outputs/iaedu_docker.jsonl --model_path iaedu --api_timeout 180
```
> [!NOTE]
> A amostra `demo/` serve apenas para confirmar que o pipeline corre ponta-a-ponta. Para uma avaliação real, descarregar o dataset oficial de [huggingface.co/datasets/VisuLogic/VisuLogic](https://huggingface.co/datasets/VisuLogic/VisuLogic), colocar `data.jsonl` + `images/` em `VisuLogic/VisuLogic-Eval/` e usar `--input_file VisuLogic/VisuLogic-Eval/data.jsonl`.

#### D. MuSLR
Executar a inferência de raciocínio simbólico (exemplo demonstrativo):
```bash
docker compose run --rm app python MuSLR/muslr_agent.py --dataset MuSLR/data/muslr_sample.jsonl --id demo_001
```

## Preparar Ambiente

A partir da raiz do projeto:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira"
conda activate lrag
```

Se for necessario instalar o suporte a `.env`:

```powershell
python -m pip install python-dotenv
```

## Dados do LogicRAG

O projeto espera a pasta:

```text
LogicRAG/LogicRAG_Data/
```

Esta pasta contem os dados grandes e esta ignorada pelo Git. Se os dados ja tiverem sido extraidos noutro diretorio, podem ser copiados para dentro de `LogicRAG/`.

Exemplo:

```powershell
Copy-Item -Recurse -Path "C:\caminho\para\LogicRAG_Data" -Destination ".\LogicRAG\LogicRAG_Data"
```

## Demo 1: LogicRAG + IAedu

Entrar na pasta do LogicRAG:

```powershell
cd .\LogicRAG
```

Converter os ficheiros de conhecimento pre-computado para factos em linguagem natural:

```powershell
python .\parse_kb_to_csv.py
```

Resultado esperado:

- O script encontra os ficheiros KB em `LogicRAG_Data/precomputed_knowledge_base/kb_out_kitti`.
- Gera o ficheiro local `resultados_kitti.csv`.
- Mostra no terminal os factos traduzidos.

Enviar o ultimo facto para o agente IAedu:

```powershell
python .\driving_agent.py
```

Resultado esperado:

- O script le o ultimo facto em `resultados_kitti.csv`.
- Envia o contexto para a API IAedu.
- Mostra uma resposta do agente com a acao recomendada para o veiculo.
- Guarda o historico local em `logic_rag_response.csv`.

## Demo 2: LLaVA-SpaceSGG + IAedu

Voltar a raiz e entrar na pasta do LLaVA-SpaceSGG:

```powershell
cd ..
cd .\LLaVA-SpaceSGG
```

Executar a analise de uma imagem real:

```powershell
python .\dataset_pipeline\stage2\run_iaedu_image.py --image .\images_real\primeira_imagem.png --output-file .\resultados\teste_iaedu_image.json
```

Ver o resultado:

```powershell
Get-Content .\resultados\teste_iaedu_image.json -TotalCount 40
```

Resultado esperado:

- O script envia a imagem para o agente IAedu.
- O output JSON contem uma descricao da imagem, objetos, bounding boxes, relacoes espaciais, camadas de profundidade e perguntas/respostas comparativas.

Desenhar as camadas de profundidade sobre a imagem (precisa de `opencv-python`):

```powershell
python .\dataset_pipeline\stage2\visualize_layers.py --result-file .\resultados\teste_iaedu_image.json --output-file .\resultados\visualizacoes\teste_layers.png
```

Resultado esperado:

- Le as `Layer N: <ref>...</ref><box>[[...]]</box>` do JSON e desenha cada caixa, com uma cor por camada de profundidade.
- Guarda a imagem anotada no caminho indicado.
- Nota: voltar a correr `visualize_layers.py` sempre que o JSON for regerado, senao a imagem fica dessincronizada do resultado.

## Demo 3 (Alternativa): Inferencia Local com Ollama

Em vez da nuvem IAedu, os fluxos podem correr com modelos locais via Ollama. Util para demonstrar sem internet nem chave de API.

Preparar o Ollama (uma vez):

```bash
ollama serve            # arrancar o servidor local (fica a correr)
ollama pull llama3      # modelo de texto para o LogicRAG
ollama pull llava       # modelo de visao para o LLaVA-SpaceSGG
```

LogicRAG com Ollama (a partir de `LogicRAG/`):

```bash
LLM_BACKEND=ollama OLLAMA_MODEL=llama3 python driving_agent.py
```

LLaVA-SpaceSGG com Ollama (a partir de `LLaVA-SpaceSGG/`):

```bash
LLM_BACKEND=ollama OLLAMA_VISION_MODEL=llava python dataset_pipeline/stage2/run_iaedu_image.py --image images_real/primeira_imagem.png --output-file resultados/ollama_primeira.json
```

Notas:

- `LLM_BACKEND` por omissao e `iaedu`; basta defini-la como `ollama` para alternar (no PowerShell usar `$env:LLM_BACKEND="ollama"`).
- O modelo local `llava` da uma descricao mais solta e nao garante o formato estruturado de boxes 0-999 como a IAedu, por isso o resultado nao e identico ao da nuvem.
- Em maquinas com pouca RAM, usar modelos mais leves (ex. `OLLAMA_MODEL=llama3.2:3b`).

## Modulo 3: VisuLogic

O VisuLogic fica em:

```text
VisuLogic/
```

Este modulo usa o codigo oficial de avaliacao do benchmark VisuLogic. O dataset completo e os outputs locais nao devem ser enviados para o GitHub.

Foi acrescentado um adaptador `iaedu`, usando as mesmas variaveis da API IAedu dos outros modulos. Tambem foi mantido o adaptador `ollama:llava`, permitindo avaliar o benchmark com o modelo multimodal local do Ollama.

Preparar ambiente:

```powershell
cd .\VisuLogic\VisuLogic-Eval
conda create -n visulogic python=3.10 -y
conda activate visulogic
python -m pip install -r requirements.txt
```

Testes rapidos:

```powershell
python -m py_compile .\evaluation\eval_model.py .\models\__init__.py .\models\iaedu_api.py
python .\evaluation\eval_model.py --help
```

Exemplo de avaliacao local com Ollama:

```powershell
$env:OLLAMA_TIMEOUT="900"
python .\evaluation\eval_model.py --input_file .\data.jsonl --output_file .\outputs\ollama_llava_visulogic.jsonl --model_path ollama:llava --base_url "http://localhost:11434/api/generate"
Remove-Item Env:\OLLAMA_TIMEOUT
```

Exemplo de avaliacao com IAedu:

```powershell
python .\evaluation\eval_model.py --input_file .\data.jsonl --output_file .\outputs\iaedu_visulogic.jsonl --model_path iaedu --api_timeout 180
```

Ver instrucoes completas em:

```text
VisuLogic/README.md
```

## Verificacao Antes de Commit

Antes de fazer commit, confirmar o estado do repositorio:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira"
git status --short
```

Nao fazer `git add .` neste projeto, porque existem dados grandes, ficheiros `.env` e outputs locais.

Para adicionar apenas documentacao:

```powershell
git add README.md
git commit -m "docs: add project demo guide"
git push
```

## Notas Para a Apresentacao

Screenshots uteis:

- Terminal com `conda activate lrag`.
- Confirmacao dos `.env` mascarados com `=***`.
- Execucao de `python .\parse_kb_to_csv.py`.
- Execucao de `python .\driving_agent.py` com a resposta do agente.
- Execucao de `run_iaedu_image.py` e visualizacao do JSON.
- Estrutura e comandos do modulo `VisuLogic`.
- GitHub com o commit de documentacao.

Nunca colocar chaves reais da API em screenshots, slides ou commits.
