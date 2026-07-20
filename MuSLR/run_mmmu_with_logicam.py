import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


MUSLR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MUSLR_DIR.parent
MMMU_INTEGRATION_DIR = PROJECT_ROOT / "MMMU"

sys.path.insert(
    0,
    str(MMMU_INTEGRATION_DIR),
)

from run_iaedu_mmmu import (
    call_iaedu,
    extract_prediction,
    load_dotenv,
    load_mmmu_dataset,
    normalize_options,
    save_input_image,
)


def build_logicam_prompt(
    example,
    subject,
    options,
):
    """
    Constrói um prompt inspirado na framework LogiCAM.

    O MMMU não contém regras lógicas formais explícitas.
    O adaptador pede ao modelo para extrair premissas da
    imagem e aplicar raciocínio estruturado.
    """

    question = example["question"]
    question_type = example["question_type"]

    if question_type == "multiple-choice":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        formatted_options = "\n".join(
            f"{letters[index]}) {option}"
            for index, option in enumerate(options)
        )

        answer_instruction = (
            "End with exactly: "
            "FINAL ANSWER: <option letter>"
        )

        options_section = (
            f"\nOptions:\n{formatted_options}"
        )
    else:
        answer_instruction = (
            "End with exactly: "
            "FINAL ANSWER: <short answer>"
        )

        options_section = ""

    return f"""You are a multimodal reasoning system using the LogiCAM methodology.

This is a cross-dataset experiment using an MMMU question.
MMMU does not provide explicit formal logical rules, so derive
the relevant premises from the supplied image or images,
the question, and the required expert knowledge.

Discipline:
{subject}

Question:
{question}
{options_section}

Follow these LogiCAM-inspired stages:

1. Premise Selection
Select only the visual and textual premises relevant to the question.

2. Reasoning Type Identification
Identify whether the problem requires quantitative, spatial,
causal, symbolic, scientific, commonsense, or another reasoning type.

3. Structured Reasoning
Apply the selected premises carefully. Use explicit logical or
mathematical steps when appropriate. Do not assume visual facts
that are not supported by the supplied image.

4. Conclusion
Provide the final answer in the required format.

{answer_instruction}"""


def extract_logicam_answer(
    response,
    question_type,
    option_count,
):
    """Extrai a conclusão final produzida pelo LogiCAM."""

    response = str(response).strip()

    if question_type == "multiple-choice":
        return extract_prediction(
            response=response,
            question_type=question_type,
            option_count=option_count,
        )

    matches = re.findall(
        r"FINAL\s+ANSWER\s*:\s*(.+)",
        response,
        flags=re.IGNORECASE,
    )

    if matches:
        return matches[-1].strip()

    return response


def save_results(
    predictions,
    details,
    output_path,
):
    """Guarda previsões oficiais e raciocínios detalhados."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            predictions,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    details_path = output_path.with_name(
        f"{output_path.stem}_details.json"
    )

    details_path.write_text(
        json.dumps(
            details,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Executa o dataset MMMU com um fluxo "
            "de raciocínio inspirado no LogiCAM."
        )
    )

    parser.add_argument(
        "--subject",
        default="Accounting",
        help="Categoria do MMMU.",
    )

    parser.add_argument(
        "--split",
        default="validation",
        choices=[
            "dev",
            "validation",
            "test",
        ],
        help="Divisão do MMMU.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Índice inicial.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Número de exemplos.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout da IAedu.",
    )

    parser.add_argument(
        "--output",
        default=str(
            MUSLR_DIR
            / "resultados"
            / "pedro_mmmu_com_logicam.json"
        ),
        help="Ficheiro JSON de previsões.",
    )

    args = parser.parse_args()

    if args.start < 0:
        raise ValueError(
            "--start nao pode ser negativo."
        )

    if args.limit <= 0:
        raise ValueError(
            "--limit deve ser superior a zero."
        )

    load_dotenv(
        PROJECT_ROOT
        / "VisuLogic"
        / ".env"
    )

    channel_id = os.environ.get(
        "IAEDU_CHANNEL_ID"
    )

    if not channel_id:
        raise RuntimeError(
            "IAEDU_CHANNEL_ID nao esta configurado."
        )

    dataset = load_mmmu_dataset(
        args.subject,
        args.split,
    )

    if args.start >= len(dataset):
        raise ValueError(
            f"--start={args.start} excede "
            f"os {len(dataset)} exemplos."
        )

    end = min(
        args.start + args.limit,
        len(dataset),
    )

    output_path = Path(
        args.output
    )

    image_directory = (
        MUSLR_DIR
        / "resultados"
        / "mmmu_images"
    )

    predictions = {}
    details = []

    run_id = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    total_to_process = end - args.start

    for position, index in enumerate(
        range(args.start, end),
        start=1,
    ):
        example = dataset[index]
        example_id = example["id"]
        question_type = example["question_type"]

        print(
            f"[{position}/{total_to_process}] "
            f"{example_id}"
        )

        options = normalize_options(
            example.get("options")
        )

        image_path = save_input_image(
            example,
            image_directory / f"{example_id}.png",
        )

        prompt = build_logicam_prompt(
            example=example,
            subject=args.subject,
            options=options,
        )

        response = ""
        prediction = "N/A"
        error = None

        try:
            response = call_iaedu(
                message=prompt,
                channel_id=channel_id,
                thread_id=(
                    "pedro-cross-logicam-mmmu-"
                    f"{run_id}-{example_id}"
                ),
                user_info={
                    "name": "Pedro Vieira",
                    "project": (
                        "LogiCAM tested with "
                        "MMMU dataset"
                    ),
                },
                image_path=image_path,
                timeout=args.timeout,
            )

            prediction = extract_logicam_answer(
                response=response,
                question_type=question_type,
                option_count=len(options),
            )

        except Exception as exception:
            response = ""
            prediction = "N/A"
            error = str(exception)

        ground_truth = str(
            example["answer"]
        ).strip()

        prediction_text = str(
            prediction
        ).strip()

        is_correct = (
            prediction_text.casefold()
            == ground_truth.casefold()
        )

        predictions[example_id] = prediction

        details.append(
            {
                "id": example_id,
                "experiment": (
                    "MMMU dataset with "
                    "LogiCAM-style reasoning"
                ),
                "subject": args.subject,
                "split": args.split,
                "question_type": question_type,
                "question": example["question"],
                "options": options,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "logicam_reasoning": response,
                "image_path": (
                    str(image_path)
                    if image_path
                    else None
                ),
                "error": error,
            }
        )

        print("Previsao:", prediction)
        print("Resposta correta:", ground_truth)

        if error:
            print("Erro:", error)

        save_results(
            predictions=predictions,
            details=details,
            output_path=output_path,
        )

    correct = sum(
        item["correct"]
        for item in details
    )

    total = len(details)

    accuracy = (
        correct / total
        if total
        else 0
    )

    details_path = output_path.with_name(
        f"{output_path.stem}_details.json"
    )

    print()
    print(
        "Experiencia: "
        "MMMU -> LogiCAM"
    )
    print(f"Resultado: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Previsoes: {output_path}")
    print(f"Detalhes: {details_path}")


if __name__ == "__main__":
    main()