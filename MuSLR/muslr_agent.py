import argparse
import base64
import json
import os
from datetime import datetime
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent


# Instância de demonstração (exemplo do artigo, Figura 1) — usada quando não é
# passado um dataset, para permitir uma demo reproduzível sem o dataset do HuggingFace.
DEMO_INSTANCE = {
    "id": "demo_001",
    "domain": "Autonomous Driving",
    "symbol": "FOL",
    "depth": 4,
    "image_file": "data/demo_001.png",
    "full_context": (
        "1. If driving straight is allowed, then there is no road construction. "
        "2. On a road without a parking sign, the driver should either go straight or must change lane. "
        "3. On a two-way road with one lane per direction, if the driver must change lanes, the driver "
        "must wait for vehicles in the opposing lane to pass before changing lanes."
    ),
    "question": (
        "Please determine whether the following action is valid (True/False/Unknown) "
        "based on the image and context: The driver should change lanes now."
    ),
    "choices": ["True", "False", "Unknown"],
    "answer": "False",
}


# Prompt inspirado na framework LogiCAM (selecionar premissas -> identificar o tipo de
# raciocínio -> aplicar regras lógicas formais e concluir).
PROMPT_TEMPLATE = """You are a multimodal symbolic logical reasoning system (MuSLR / LogiCAM).
Reason over the IMAGE and the CONTEXT using formal logical rules (propositional and first-order logic).
Work step by step: (1) select the premises relevant to the image, (2) identify the reasoning type,
(3) apply the logical rules (e.g. Modus Ponens / Modus Tollens) and draw a sound conclusion.

Context:
{context}

Question:
{question}

Answer options: {choices}

Return your step-by-step symbolic reasoning and end with a final line in the exact format:
FINAL ANSWER: <one of: {choices}>"""


def carregar_instancia(dataset, instance_id):
    """Carrega uma instância do dataset MuSLR (.json ou .jsonl). Sem dataset, usa a demo."""
    if not dataset:
        print("Sem --dataset: a usar a instância de demonstração do artigo.\n")
        return DEMO_INSTANCE

    path = Path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {path}")

    texto = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".jsonl":
        instancias = [json.loads(linha) for linha in texto.splitlines() if linha.strip()]
    else:
        dados = json.loads(texto)
        instancias = dados if isinstance(dados, list) else list(dados.values())

    if instance_id:
        for inst in instancias:
            if str(inst.get("id")) == str(instance_id):
                return inst
        raise KeyError(f"id não encontrado no dataset: {instance_id}")
    return instancias[0]


def construir_prompt(inst):
    choices = ", ".join(inst.get("choices", ["True", "False", "Unknown"]))
    return PROMPT_TEMPLATE.format(
        context=inst.get("full_context", ""),
        question=inst.get("question", ""),
        choices=choices,
    )


def chamar_iaedu(prompt, image_path):
    """Envia o prompt e a imagem usando o cliente IAedu partilhado."""
    import sys

    client_dir = (
        BASE_DIR.parent
        / "LLaVA-SpaceSGG"
        / "dataset_pipeline"
        / "stage2"
    )

    if str(client_dir) not in sys.path:
        sys.path.insert(0, str(client_dir))

    from iaedu_client import call_iaedu

    api_key = os.getenv("OPENAI_API_KEY")
    api_endpoint = os.getenv("OPENAI_API_ENDPOINT")
    channel_id = os.getenv("IAEDU_CHANNEL_ID")
    thread_id = (
        os.getenv("IAEDU_THREAD_ID")
        or f"muslr_{datetime.now():%Y%m%d_%H%M%S}"
    )

    if not api_key or not api_endpoint or not channel_id:
        raise RuntimeError(
            "Faltam credenciais no .env "
            "(OPENAI_API_KEY, OPENAI_API_ENDPOINT, IAEDU_CHANNEL_ID)."
        )

    print(f"Using IAedu thread: {thread_id}\n")

    return call_iaedu(
        message=prompt,
        channel_id=channel_id,
        thread_id=thread_id,
        user_info={
            "name": "Pedro Vieira",
            "project": "MuSLR / LogiCAM",
        },
        endpoint=api_endpoint,
        api_key=api_key,
        timeout=180,
        image_path=image_path,
    )

def chamar_ollama_vision(prompt, image_path):
    """Envia o prompt (e a imagem em base64, se existir) a um modelo local via Ollama."""
    model = os.getenv("OLLAMA_VISION_MODEL") or os.getenv("OLLAMA_MODEL") or "llava"
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "600"))

    print(f"Using Ollama model: {model}\n")
    payload = {"model": model, "prompt": prompt, "stream": True}
    if image_path and Path(image_path).exists():
        payload["images"] = [base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")]

    response = requests.post(endpoint, json=payload, stream=True, timeout=timeout)
    response.raise_for_status()

    partes = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line.decode("utf-8"))
        token = chunk.get("response", "")
        if token:
            print(token, end="", flush=True)
            partes.append(token)
        if chunk.get("done"):
            break
    return "".join(partes).strip()


def main():
    parser = argparse.ArgumentParser(description="MuSLR / LogiCAM — raciocínio simbólico multimodal via IAedu ou Ollama.")
    parser.add_argument("--backend", choices=["iaedu", "ollama"], default=None,
                        help="Motor: 'iaedu' (nuvem) ou 'ollama' (local). Por defeito usa LLM_BACKEND ou iaedu.")
    parser.add_argument("--dataset", default=None, help="Ficheiro .json/.jsonl com instâncias MuSLR. Sem isto, usa a demo.")
    parser.add_argument("--id", default=None, help="id da instância a usar (opcional).")
    parser.add_argument("--image", default=None, help="Caminho da imagem (sobrepõe o image_file da instância).")
    parser.add_argument("--output-file", default=str(BASE_DIR / "resultados" / "muslr_output.json"))
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(BASE_DIR.parent / "VisuLogic" / ".env")
    except ImportError:
        pass

    backend = args.backend or os.getenv("LLM_BACKEND", "iaedu").strip().lower()
    print("MuSLR / LogiCAM — Raciocínio Simbólico Multimodal\n")
    print(f"Motor selecionado: {backend.upper()}\n")

    inst = carregar_instancia(args.dataset, args.id)
    image_path = args.image or inst.get("image_file")
    if image_path and not Path(image_path).exists():
        print(f"(Aviso: imagem '{image_path}' não encontrada; a enviar apenas o contexto textual.)\n")
        image_path = None

    prompt = construir_prompt(inst)
    print(f"Instância: {inst.get('id')} | domínio: {inst.get('domain')} | lógica: {inst.get('symbol')}\n")
    print("Resposta:")
    print("-" * 50)

    try:
        if backend == "ollama":
            resposta = chamar_ollama_vision(prompt, image_path)
        else:
            resposta = chamar_iaedu(prompt, image_path)
        print("\n" + "-" * 50)

        saida = {
            "id": inst.get("id"),
            "backend": backend,
            "question": inst.get("question"),
            "gold_answer": inst.get("answer"),
            "model_answer": resposta,
        }
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(saida, indent=4, ensure_ascii=False), encoding="utf-8")
        print(f"Resposta guardada em {out_path}")
        return 0

    except requests.exceptions.ConnectionError:
        print(f"\nErro de ligação. Confirma que o serviço está a correr "
              f"({'ollama serve' if backend == 'ollama' else 'IAedu / internet'}).")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
