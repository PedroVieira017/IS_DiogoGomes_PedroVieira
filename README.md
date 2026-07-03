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

- Windows com PowerShell
- Miniconda ou Anaconda
- Ambiente conda `lrag`
- Credenciais da API IAedu
- Dados pre-computados do LogicRAG em `LogicRAG/LogicRAG_Data/`

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
