# VisuLogic

Integracao do artigo **VisuLogic: A Benchmark for Evaluating Visual Reasoning in Multi-modal Large Language Models** no projeto `IS_DiogoGomes_PedroVieira`.

- Artigo: https://arxiv.org/abs/2504.15279
- Pagina oficial: https://visulogic-benchmark.github.io/VisuLogic/
- Codigo de avaliacao: https://github.com/VisuLogic-Benchmark/VisuLogic-Eval
- Dataset do benchmark: https://huggingface.co/datasets/VisuLogic/VisuLogic
- Codigo de treino: https://github.com/VisuLogic-Benchmark/VisuLogic-Train

## Objetivo da Integracao

O VisuLogic avalia raciocinio visual em modelos multimodais atraves de perguntas de escolha multipla sobre imagens. O foco deste modulo e integrar o codigo de avaliacao oficial, ligar a avaliacao a API IAedu como nos outros modulos, documentar como preparar o benchmark e deixar comandos reproduziveis para uma demonstracao controlada.

Foi usado o **Eval Code** em vez do **Train Code** porque o treino exige datasets grandes e infraestrutura pesada com GPUs de elevada memoria. Para este trabalho, a parte importante e compreender o benchmark, preparar a avaliacao e demonstrar como um modelo multimodal pode ser testado.

## Estrutura

```text
VisuLogic/
+-- README.md
+-- resultados/
+-- VisuLogic-Eval/
    +-- evaluation/
    +-- models/
        +-- iaedu_api.py
        +-- ollama_vision.py
    +-- scripts/
    +-- assets/
    +-- requirements.txt
    +-- README.md
```

## Preparar Ambiente

A partir da raiz do projeto:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira"
conda create -n visulogic python=3.10 -y
conda activate visulogic
cd .\VisuLogic\VisuLogic-Eval
python -m pip install -r requirements.txt
```

Nota: o ficheiro `requirements.txt` instala PyTorch com CUDA 12.1 e tambem inclui dependencias pesadas de modelos multimodais. Em maquinas sem GPU adequada, usar este modulo sobretudo para analise/documentacao ou avaliar atraves da API IAedu.

## Configurar API IAedu

Criar um ficheiro `.env` dentro de `VisuLogic/`, no mesmo estilo dos outros modulos:

```env
OPENAI_API_KEY=colocar_a_chave_iaedu
OPENAI_API_ENDPOINT=colocar_o_endpoint_stream_iaedu
IAEDU_CHANNEL_ID=colocar_o_channel_id
IAEDU_THREAD_ID=colocar_um_thread_id
```

O ficheiro `.env` esta ignorado pelo GitHub e nao deve ser enviado para o repositorio.

## Preparar Dados do Benchmark

O dataset oficial nao deve ser enviado para o GitHub. Descarregar manualmente a partir de:

```text
https://huggingface.co/datasets/VisuLogic/VisuLogic
```

Depois colocar localmente dentro de:

```text
VisuLogic/VisuLogic-Eval/
+-- data.jsonl
+-- images/
    +-- 00000.png
    +-- 00001.png
```

Estes ficheiros estao ignorados pelo `.gitignore`.

## Testes Rapidos

Confirmar que o codigo compila:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira\VisuLogic\VisuLogic-Eval"
python -m py_compile .\evaluation\eval_model.py .\models\__init__.py .\models\iaedu_api.py
```

Confirmar que o avaliador mostra os argumentos esperados:

```powershell
python .\evaluation\eval_model.py --help
```

## Executar Avaliacao

### Avaliacao com API IAedu

Esta e a opcao alinhada com os modulos LogicRAG e LLaVA-SpaceSGG. O avaliador envia a imagem e a pergunta para o agente IAedu, recebe a resposta e extrai a opcao final (`A`, `B`, `C` ou `D`).

Executar o VisuLogic com IAedu:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira\VisuLogic\VisuLogic-Eval"
mkdir outputs
python .\evaluation\eval_model.py `
  --input_file .\data.jsonl `
  --output_file .\outputs\iaedu_visulogic.jsonl `
  --model_path iaedu `
  --api_timeout 180
```

Se nao for usado `.env`, tambem e possivel passar os valores no comando:

```powershell
python .\evaluation\eval_model.py `
  --input_file .\data.jsonl `
  --output_file .\outputs\iaedu_visulogic.jsonl `
  --model_path iaedu `
  --api_key "COLOCAR_API_KEY" `
  --base_url "COLOCAR_ENDPOINT_IAEDU" `
  --channel_id "COLOCAR_CHANNEL_ID" `
  --thread_id "visulogic" `
  --api_timeout 180
```

### Avaliacao com Ollama Local

Esta e a alternativa local ja usada nos modulos LogicRAG e LLaVA-SpaceSGG. O Ollama deve estar a correr e o modelo `llava` deve estar instalado.

Preparar o Ollama:

```powershell
ollama serve
ollama pull llava
```

Executar o VisuLogic com Ollama:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira\VisuLogic\VisuLogic-Eval"
mkdir outputs
$env:OLLAMA_TIMEOUT="900"
python .\evaluation\eval_model.py `
  --input_file .\data.jsonl `
  --output_file .\outputs\ollama_llava_visulogic.jsonl `
  --model_path ollama:llava `
  --base_url "http://localhost:11434/api/generate"
Remove-Item Env:\OLLAMA_TIMEOUT
```

Nota: sem `judge_api_key`, quando a resposta do Ollama nao tiver uma opcao clara (`A`, `B`, `C` ou `D`), o avaliador marca a extracao como `N`. Para resultados mais robustos, pode ser usado um judge OpenAI-compatible.

### Avaliacao com API OpenAI-compatible

Exemplo com modelo OpenAI-compatible:

```powershell
cd "C:\Users\Master\Desktop\IS_moreDiogoGomes\2025-LMM-LogicRag-Visual\IS_DiogoGomes_PedroVieira\VisuLogic\VisuLogic-Eval"
mkdir outputs
python .\evaluation\eval_model.py `
  --input_file .\data.jsonl `
  --output_file .\outputs\gpt4o_visulogic.jsonl `
  --model_path gpt-4o `
  --api_key "COLOCAR_API_KEY" `
  --judge_api_key "COLOCAR_API_KEY"
```

Se for necessario usar um endpoint compativel com OpenAI:

```powershell
python .\evaluation\eval_model.py `
  --input_file .\data.jsonl `
  --output_file .\outputs\modelo_visulogic.jsonl `
  --model_path gpt-4o `
  --api_key "COLOCAR_API_KEY" `
  --base_url "COLOCAR_BASE_URL" `
  --judge_api_key "COLOCAR_API_KEY" `
  --judge_base_url "COLOCAR_BASE_URL"
```

## Resultado Esperado

O script percorre as perguntas em `data.jsonl`, envia cada imagem e pergunta ao modelo, extrai a opcao final (`A`, `B`, `C` ou `D`) e calcula a accuracy geral e por categoria:

- Quantitative Reasoning
- Spatial Reasoning
- Positional Reasoning
- Attribute Reasoning
- Stylistic Reasoning
- Other

Os outputs locais devem ficar em:

```text
VisuLogic/VisuLogic-Eval/outputs/
```

ou, para resultados resumidos do trabalho:

```text
VisuLogic/resultados/
```

## Notas Para a Apresentacao

- Mostrar a pagina oficial do VisuLogic e o artigo no arXiv.
- Explicar que o benchmark evita atalhos linguisticos e foca raciocinio visual.
- Justificar o uso do Eval Code: avaliacao reproduzivel sem treinar modelos de raiz.
- Mostrar a estrutura `data.jsonl + images/`.
- Mostrar o comando `eval_model.py --help` e, se houver credenciais/dados, uma execucao curta.
