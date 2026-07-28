"""Executa no Docker a avaliacao oficial MMMU sem chamadas externas."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MMMU_DIR = Path(__file__).resolve().parent / "mmmu"

MODELS = {
    "llava": (
        "LLaVA-1.5-13B",
        "example_outputs/llava1.5_13b/total_val_output.json",
    ),
    "qwen": (
        "Qwen-VL",
        "example_outputs/qwen_vl/total_val_output.json",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Avalia localmente as previsoes oficiais disponibilizadas "
            "com o MMMU. Nao usa API nem GPU."
        )
    )
    parser.add_argument(
        "--model",
        choices=["all", *MODELS],
        default="all",
        help="Modelo a avaliar; por omissao avalia os dois.",
    )
    return parser.parse_args()


def run_model(key: str) -> int:
    label, prediction_path = MODELS[key]
    print(flush=True)
    print("=" * 62, flush=True)
    print(
        f"  MMMU - {label} - 900 previsoes de validacao",
        flush=True,
    )
    print("=" * 62, flush=True)

    command = [
        sys.executable,
        "main_eval_only.py",
        "--output_path",
        prediction_path,
        "--answer_path",
        "answer_dict_val.json",
    ]
    completed = subprocess.run(command, cwd=MMMU_DIR, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    selected = list(MODELS) if args.model == "all" else [args.model]

    print("Responsavel pela reproducao: Pedro Vieira", flush=True)
    print(
        "Modo offline: nao utiliza IAedu, outra API ou GPU.",
        flush=True,
    )

    for key in selected:
        return_code = run_model(key)
        if return_code != 0:
            return return_code

    print(flush=True)
    print("Avaliacao oficial MMMU concluida.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
