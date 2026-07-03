#!/usr/bin/env python3
"""Menu unico para correr os 4 projetos dentro do container (IAedu / Ollama)."""
import os
import subprocess


def cmd_logicrag():
    return ["python", "LogicRAG/driving_agent.py"]


def cmd_llava():
    return ["python", "LLaVA-SpaceSGG/dataset_pipeline/stage2/run_iaedu_image.py",
            "--image", "LLaVA-SpaceSGG/images_real/primeira_imagem.png",
            "--output-file", "LLaVA-SpaceSGG/resultados/docker_llava.json"]


def cmd_muslr():
    return ["python", "MuSLR/muslr_agent.py",
            "--dataset", "MuSLR/data/muslr_sample.jsonl", "--id", "demo_001"]


def cmd_visulogic(backend):
    ep = os.getenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434/api/generate")
    base = ["python", "VisuLogic/VisuLogic-Eval/evaluation/eval_model.py",
            "--input_file", "VisuLogic/VisuLogic-Eval/data.jsonl",
            "--output_file", "VisuLogic/VisuLogic-Eval/outputs/docker_visulogic.jsonl"]
    if backend == "ollama":
        base += ["--model_path", "ollama:llava", "--base_url", ep]
    else:
        base += ["--model_path", "gpt-4o",
                 "--api_key", os.getenv("OPENAI_API_KEY", ""),
                 "--base_url", os.getenv("OPENAI_API_ENDPOINT", "")]
    return base


PROJETOS = {"1": "Logic-RAG", "2": "LLaVA-SpaceSGG", "3": "MuSLR / LogiCAM", "4": "VisuLogic"}


def main():
    print("=" * 52)
    print("  IS_DiogoGomes_PedroVieira - Integracao dos 4 projetos")
    print("=" * 52)
    for k, nome in PROJETOS.items():
        print("   " + k + ". " + nome)
    print("   0. Sair")
    escolha = input("\nProjeto a correr: ").strip()
    if escolha not in PROJETOS:
        return 0
    backend = input("Backend [iaedu/ollama] (Enter = iaedu): ").strip().lower() or "iaedu"
    if backend not in ("iaedu", "ollama"):
        backend = "iaedu"
    os.environ["LLM_BACKEND"] = backend
    if escolha == "1":
        cmd = cmd_logicrag()
    elif escolha == "2":
        cmd = cmd_llava()
    elif escolha == "3":
        cmd = cmd_muslr()
    else:
        cmd = cmd_visulogic(backend)
    print("\n>>> " + PROJETOS[escolha] + "  |  backend = " + backend + "\n")
    return subprocess.call(cmd, env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
