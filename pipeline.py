#!/usr/bin/env python3
"""Pipeline de integracao dos 4 projetos: uma imagem entra, um relatorio unico sai."""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "LLaVA-SpaceSGG" / "dataset_pipeline"))

from stage2.iaedu_client import call_iaedu, call_ollama_vision, load_dotenv


PROMPT_LLAVA = """Analyze the image and generate an open-vocabulary scene graph.
Return:
1. A short scene description.
2. Objects with normalized bounding boxes: <ref>object</ref><box>[[x1, y1, x2, y2]]</box>.
3. Spatial relations in this format: <pred>subject relation object</pred>.
4. Depth layers from closest to farthest.
Use coordinates from 0 to 999. Be concise."""

PROMPT_LOGICRAG = """You are the decision module of an intelligent perception system (Logic-RAG style).
Our scene graph engine extracted the following spatial facts from the image:
{factos}

Academic offline task: classify the scene with one high-level label:
- SAFE: nothing requires attention.
- CAUTION: something deserves monitoring.
- ALERT: an element requires immediate attention.
Answer with the label and two short sentences of justification."""

PROMPT_MUSLR = """You are a multimodal symbolic logical reasoning system (MuSLR / LogiCAM).
Use the IMAGE and these premises extracted from its scene graph:
{premissas}

Question: is the following statement valid (True/False/Unknown)?
Statement: "{afirmacao}"

Work step by step: (1) select relevant premises, (2) identify the reasoning type,
(3) apply formal logical rules and conclude.
End with exactly: FINAL ANSWER: <True/False/Unknown>"""

PROMPT_VISULOGIC = """Visual reasoning task (VisuLogic style). Look at the image and answer:

Question: Which element of the scene is closest to the camera?
A) The background
B) The main subject in the foreground
C) Nothing is distinguishable
D) All elements are at the same distance

Reason briefly and answer with only one final option: A, B, C, or D."""


def perguntar(mensagem, image_path, backend, thread_prefix):
    if backend == "ollama":
        return call_ollama_vision(mensagem, image_path)
    channel_id = os.environ.get("IAEDU_CHANNEL_ID")
    thread_id = f"{thread_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return call_iaedu(mensagem, channel_id=channel_id, thread_id=thread_id,
                      user_info={}, image_path=image_path)


def extrair_relacoes(grafo_texto):
    rels = re.findall(r"<pred>(.*?)</pred>", grafo_texto, re.DOTALL)
    rels = [re.sub(r"\s+", " ", r).strip() for r in rels if r.strip()]
    if rels:
        return rels
    padrao = re.compile(r"\b(above|below|left|right|front|behind|closer|farther|on|under)\b", re.I)
    return [l.strip("-* ").strip() for l in grafo_texto.splitlines() if padrao.search(l)][:8]


def desenhar_boxes(image_path, grafo, output_file):
    """Desenha TODAS as bounding boxes <ref>...<box> do grafo sobre a imagem (via Pillow)."""
    from PIL import Image, ImageDraw
    padrao = re.compile(
        r"<ref>(.*?)</ref>\s*<box>\s*\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]\s*</box>",
        re.DOTALL)
    boxes = padrao.findall(grafo)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    cores = [(46, 204, 113), (52, 152, 219), (231, 76, 60),
             (155, 89, 182), (241, 196, 15), (26, 188, 156)]
    for i, (label, x1, y1, x2, y2) in enumerate(boxes):
        px1 = int(int(x1) / 999 * W); py1 = int(int(y1) / 999 * H)
        px2 = int(int(x2) / 999 * W); py2 = int(int(y2) / 999 * H)
        cor = cores[i % len(cores)]
        draw.rectangle([px1, py1, px2, py2], outline=cor, width=3)
        draw.text((px1 + 3, max(py1 - 12, 2)), label.strip(), fill=cor)
    img.save(output_file)
    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description="Pipeline integrado dos 4 projetos.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--backend", choices=["iaedu", "ollama"], default=None)
    parser.add_argument("--output-dir", default="resultados_pipeline")
    args = parser.parse_args()

    load_dotenv()
    backend = args.backend or os.environ.get("LLM_BACKEND", "iaedu").strip().lower()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PIPELINE INTEGRADO: LLaVA -> Logic-RAG -> MuSLR -> VisuLogic")
    print(f"  Imagem: {image_path}  |  Backend: {backend.upper()}")
    print("=" * 60)

    relatorio = {"imagem": str(image_path), "backend": backend,
                 "timestamp": datetime.now().isoformat()}

    print("\n[1/4] LLaVA-SpaceSGG: a gerar o grafo de cena...")
    grafo = perguntar(PROMPT_LLAVA, image_path, backend, "pipe_llava")
    relatorio["etapa1_llava_spacesgg"] = {"scene_graph": grafo}
    relacoes = extrair_relacoes(grafo)
    print(f"\n      -> {len(relacoes)} relacoes espaciais extraidas.")

    print("\n[2/4] Logic-RAG: a avaliar a cena a partir das relacoes...")
    factos = "\n".join(f"- {r}" for r in relacoes) or grafo[:600]
    decisao = perguntar(PROMPT_LOGICRAG.format(factos=factos), image_path, backend, "pipe_logicrag")
    relatorio["etapa2_logicrag"] = {"factos_usados": relacoes, "avaliacao": decisao}

    print("\n[3/4] MuSLR/LogiCAM: a validar logicamente uma afirmacao...")
    afirmacao = "The main subject of the image is in front of the background."
    validacao = perguntar(PROMPT_MUSLR.format(premissas=factos, afirmacao=afirmacao),
                          image_path, backend, "pipe_muslr")
    m = re.search(r"FINAL ANSWER:\s*(True|False|Unknown)", validacao, re.I)
    relatorio["etapa3_muslr_logicam"] = {"afirmacao": afirmacao, "raciocinio": validacao,
                                         "veredito": m.group(1) if m else "N/A"}

    print("\n[4/4] VisuLogic: pergunta de raciocinio visual...")
    qa = perguntar(PROMPT_VISULOGIC, image_path, backend, "pipe_visulogic")
    op = re.search(r"\b([ABCD])\b(?!.*\b[ABCD]\b)", qa, re.DOTALL)
    relatorio["etapa4_visulogic"] = {"resposta": qa, "opcao_final": op.group(1) if op else "N/A"}

    json_path = out_dir / "relatorio_final.json"
    json_path.write_text(json.dumps(relatorio, indent=4, ensure_ascii=False), encoding="utf-8")

    imagem_anotada = out_dir / "imagem_anotada.png"
    n_boxes = 0
    try:
        n_boxes = desenhar_boxes(image_path, grafo, imagem_anotada)
        print(f"\n      -> {n_boxes} boxes desenhadas na imagem.")
    except Exception as exc:
        imagem_anotada = None
        print(f"(Aviso: nao foi possivel anotar a imagem: {exc})")

    print("\n" + "=" * 60)
    print("  RELATORIO FINAL")
    print("=" * 60)
    print(f"  Grafo de cena ......... {len(grafo)} carateres, {len(relacoes)} relacoes")
    print(f"  Avaliacao Logic-RAG ... {decisao.splitlines()[0] if decisao else 'N/A'}")
    print(f"  Veredito MuSLR ........ {relatorio['etapa3_muslr_logicam']['veredito']}")
    print(f"  Opcao VisuLogic ....... {relatorio['etapa4_visulogic']['opcao_final']}")
    print(f"\n  JSON:   {json_path}")
    if imagem_anotada:
        print(f"  Imagem: {imagem_anotada}  ({n_boxes} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
