"""Verificacao offline da imagem Docker e dos recursos de demonstracao."""

from __future__ import annotations

import importlib
import json
import os
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODULES = {
    "requests": "requests",
    "dotenv": "python-dotenv",
    "PIL": "Pillow",
    "tqdm": "tqdm",
    "openai": "openai",
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "yaml": "PyYAML",
}

REQUIRED_FILES = [
    "integra.py",
    "pipeline.py",
    "LogicRAG/driving_agent.py",
    "LogicRAG/resultados_kitti.csv",
    "LLaVA-SpaceSGG/dataset_pipeline/stage2/run_iaedu_image.py",
    "LLaVA-SpaceSGG/images_real/primeira_imagem.png",
    "LLaVA-SpaceSGG/images_real/transito.jpg",
    "MuSLR/muslr_agent.py",
    "MuSLR/data/muslr_sample.jsonl",
    "VisuLogic/VisuLogic-Eval/evaluation/eval_model.py",
    "VisuLogic/VisuLogic-Eval/evaluation/manual_query.py",
    "VisuLogic/VisuLogic-Eval/demo/data.jsonl",
    "VisuLogic/VisuLogic-Eval/demo/img/demo_puzzle_1.png",
    "MMMU/mmmu/main_eval_only.py",
    "MMMU/mmmu/answer_dict_val.json",
    "MMMU/mmmu/example_outputs/llava1.5_13b/total_val_output.json",
    "MMMU/mmmu/example_outputs/qwen_vl/total_val_output.json",
    "MMMU/run_official_eval_docker.py",
]

PYTHON_ENTRYPOINTS = [
    "integra.py",
    "pipeline.py",
    "LogicRAG/driving_agent.py",
    "LLaVA-SpaceSGG/dataset_pipeline/stage2/run_iaedu_image.py",
    "MuSLR/muslr_agent.py",
    "VisuLogic/VisuLogic-Eval/evaluation/eval_model.py",
    "VisuLogic/VisuLogic-Eval/evaluation/manual_query.py",
    "MMMU/run_official_eval_docker.py",
]


def main() -> int:
    failures: list[str] = []

    print("=" * 62)
    print("  VERIFICACAO DO CONTENTOR - PEDRO VIEIRA")
    print("=" * 62)
    print("Python:", sys.version.split()[0])

    for module_name, package_name in MODULES.items():
        try:
            importlib.import_module(module_name)
            print(f"[OK] Dependencia: {package_name}")
        except Exception as exception:
            failures.append(f"Dependencia {package_name}: {exception}")
            print(f"[ERRO] Dependencia: {package_name}")

    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if path.is_file():
            print(f"[OK] Ficheiro: {relative_path}")
        else:
            failures.append(f"Ficheiro em falta: {relative_path}")
            print(f"[ERRO] Ficheiro: {relative_path}")

    try:
        demo_path = ROOT / "VisuLogic/VisuLogic-Eval/demo/data.jsonl"
        rows = [
            json.loads(line)
            for line in demo_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != 2:
            raise ValueError(f"esperados 2 exemplos, encontrados {len(rows)}")
        print("[OK] Dataset de demonstracao VisuLogic: 2 exemplos")
    except Exception as exception:
        failures.append(f"Demo VisuLogic invalida: {exception}")
        print("[ERRO] Dataset de demonstracao VisuLogic")

    for relative_path in PYTHON_ENTRYPOINTS:
        try:
            py_compile.compile(str(ROOT / relative_path), doraise=True)
            print(f"[OK] Python: {relative_path}")
        except Exception as exception:
            failures.append(f"Compilacao {relative_path}: {exception}")
            print(f"[ERRO] Python: {relative_path}")

    iaedu_variables = (
        "OPENAI_API_KEY",
        "OPENAI_API_ENDPOINT",
        "IAEDU_CHANNEL_ID",
    )
    configured = all(os.environ.get(name) for name in iaedu_variables)
    print(
        "[INFO] IAedu: "
        + (
            "credenciais configuradas"
            if configured
            else "sem credenciais (a avaliacao offline continua disponivel)"
        )
    )

    print("=" * 62)
    if failures:
        print("CONTENTOR INVALIDO")
        for failure in failures:
            print("-", failure)
        return 1

    print("CONTENTOR PRONTO PARA A DEMONSTRACAO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
