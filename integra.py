#!/usr/bin/env python3
"""Menu unico para correr os projetos dentro do container."""
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
            "--dataset", "MuSLR/data/muslr_sample.jsonl", "--id", "demo_001",
            "--image", "LLaVA-SpaceSGG/images_real/transito.jpg",
            "--output-file", "MuSLR/resultados/docker_muslr.json"]


def cmd_mmmu():
    return ["python", "MMMU/run_official_eval_docker.py"]


def configurar_visulogic_path():
    """Make the official VisuLogic models package importable from the project root."""
    eval_dir = "VisuLogic/VisuLogic-Eval"
    current_path = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current_path.split(os.pathsep) if entry]
    if eval_dir not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([eval_dir, *entries])


def cmd_visulogic_avaliacao(backend):
    ep = os.getenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434/api/generate")
    # eval_model.py faz sys.path.append(".") e importa o pacote `models`, por isso
    # precisa que essa pasta esteja no PYTHONPATH quando corre a partir de /app.
    configurar_visulogic_path()
    base = ["python", "VisuLogic/VisuLogic-Eval/evaluation/eval_model.py",
            "--input_file", "VisuLogic/VisuLogic-Eval/demo/data.jsonl",
            "--output_file", "VisuLogic/VisuLogic-Eval/outputs/docker_visulogic.jsonl"]
    if backend == "ollama":
        base += ["--model_path", "ollama:llava", "--base_url", ep]
    else:
        base += ["--model_path", "iaedu"]
    return base


def cmd_visulogic_manual(backend, image_path, question):
    ep = os.getenv("OLLAMA_ENDPOINT", "http://host.docker.internal:11434/api/generate")
    configurar_visulogic_path()
    base = ["python", "VisuLogic/VisuLogic-Eval/evaluation/manual_query.py",
            "--image", image_path, "--question", question]
    if backend == "ollama":
        base += ["--model_path", "ollama:llava", "--base_url", ep]
    else:
        base += ["--model_path", "iaedu"]
    return base


PROJETOS = {
    "1": "Logic-RAG",
    "2": "LLaVA-SpaceSGG",
    "3": "MuSLR / LogiCAM",
    "4": "VisuLogic",
    "5": "MMMU (avaliacao oficial offline)",
}


def main():
    print("=" * 52)
    print("  Pedro Vieira - Benchmarks de raciocinio multimodal")
    print("=" * 52)
    for k, nome in PROJETOS.items():
        print("   " + k + ". " + nome)
    print("   0. Sair")
    escolha = input("\nProjeto a correr: ").strip()
    if escolha not in PROJETOS:
        return 0
    if escolha == "5":
        print("\n>>> MMMU | avaliacao offline das previsoes oficiais\n")
        return subprocess.call(cmd_mmmu(), env=os.environ)
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
        modo = input("Modo VisuLogic [manual/benchmark] (Enter = manual): ").strip().lower()
        if modo in ("benchmark", "avaliacao", "avaliação"):
            cmd = cmd_visulogic_avaliacao(backend)
        else:
            image_path = input("Caminho da imagem: ").strip().strip('"').strip("'")
            question = input("Pergunta sobre a imagem: ").strip()
            if not image_path or not question:
                print("A imagem e a pergunta sao obrigatorias no modo manual.")
                return 1
            cmd = cmd_visulogic_manual(backend, image_path, question)
    print("\n>>> " + PROJETOS[escolha] + "  |  backend = " + backend + "\n")
    return subprocess.call(cmd, env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
