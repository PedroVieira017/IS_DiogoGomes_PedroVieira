# Execucao com Docker no portatil do IPVC

Esta configuracao permite demonstrar o projeto sem instalar ambientes Conda
nem dependencias Python no portatil. Apenas e necessario ter o Docker Desktop
aberto e usar um terminal na raiz do repositorio.

## 1. Confirmar o Docker Desktop

```powershell
docker version
docker compose version
```

O comando `docker version` deve mostrar as secoes `Client` e `Server`.

## 2. Construir a imagem

```powershell
docker build -t ipvc-multimodal:latest .
```

Durante o build e executado automaticamente `docker_check.py`. O build so
termina se as dependencias e os recursos de demonstracao estiverem presentes.

## 3. Teste offline obrigatorio

```powershell
docker run --rm ipvc-multimodal:latest python docker_check.py
```

O resultado final esperado e:

```text
CONTENTOR PRONTO PARA A DEMONSTRACAO
```

## 4. Demonstracao offline do MMMU

Esta demonstracao nao precisa de internet, GPU ou credenciais:

```powershell
docker run --rm ipvc-multimodal:latest `
  python MMMU/run_official_eval_docker.py
```

Executa o avaliador oficial sobre 900 previsoes LLaVA-1.5-13B e 900 previsoes
Qwen-VL fornecidas pelos autores.

## 5. Preparar a IAedu

As credenciais nunca ficam dentro da imagem. Criar um `.env` local:

```powershell
Copy-Item .env.example .env
code .env
```

Preencher `OPENAI_API_KEY`, `OPENAI_API_ENDPOINT` e `IAEDU_CHANNEL_ID`.

## 6. Abrir o menu integrado e guardar resultados no computador

```powershell
docker run --rm -it `
  --env-file .\.env `
  -v "${PWD}:/app" `
  ipvc-multimodal:latest
```

O volume `-v "${PWD}:/app"` permite guardar os resultados diretamente na
pasta do projeto no computador. Em PowerShell devem ser usados dois hifens em
`--rm`; o carater `–` copiado de processadores de texto nao funciona.

## 7. Executar cada projeto diretamente

Os comandos seguintes fazem o mesmo que as opcoes do menu, mas permitem
arrancar diretamente o projeto pretendido.

### Logic-RAG

```powershell
docker run --rm `
  --env-file .\.env `
  -e LLM_BACKEND=iaedu `
  -v "${PWD}:/app" `
  ipvc-multimodal:latest `
  python LogicRAG/driving_agent.py
```

### LLaVA-SpaceSGG

```powershell
docker run --rm `
  --env-file .\.env `
  -e LLM_BACKEND=iaedu `
  -v "${PWD}:/app" `
  ipvc-multimodal:latest `
  python LLaVA-SpaceSGG/dataset_pipeline/stage2/run_iaedu_image.py `
  --image LLaVA-SpaceSGG/images_real/primeira_imagem.png `
  --output-file LLaVA-SpaceSGG/resultados/docker_llava.json
```

### MuSLR / LogiCAM

```powershell
docker run --rm `
  --env-file .\.env `
  -v "${PWD}:/app" `
  ipvc-multimodal:latest `
  python MuSLR/muslr_agent.py `
  --backend iaedu `
  --dataset MuSLR/data/muslr_sample.jsonl `
  --id demo_001 `
  --image LLaVA-SpaceSGG/images_real/transito.jpg `
  --output-file MuSLR/resultados/docker_muslr.json
```

### VisuLogic (dois exemplos de demonstracao)

```powershell
docker run --rm `
  --env-file .\.env `
  -v "${PWD}:/app" `
  ipvc-multimodal:latest `
  python VisuLogic/VisuLogic-Eval/evaluation/eval_model.py `
  --input_file VisuLogic/VisuLogic-Eval/demo/data.jsonl `
  --output_file VisuLogic/VisuLogic-Eval/outputs/docker_visulogic.jsonl `
  --model_path iaedu
```

O MMMU e executado diretamente pelo comando offline apresentado na seccao 4.

## 8. Pipeline completo sobre uma imagem

```powershell
docker run --rm `
  --env-file .\.env `
  -v "${PWD}:/app" `
  ipvc-multimodal:latest `
  python pipeline.py `
  --image LLaVA-SpaceSGG/images_real/transito.jpg `
  --question "Que elementos representam maior risco nesta cena?" `
  --backend iaedu `
  --output-dir resultados_pipeline/docker_ipvc
```

## Alternativa com Docker Compose

```powershell
docker compose build
docker compose run --rm app
```

O `.env` e opcional para o teste offline, mas e necessario para utilizar a
IAedu. Os datasets completos, pesos de modelos e credenciais nao sao copiados
para a imagem.
