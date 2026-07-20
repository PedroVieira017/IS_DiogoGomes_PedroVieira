import argparse
import ast
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw


MMMU_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MMMU_DIR.parent

CLIENT_DIR = (
    PROJECT_ROOT
    / "LLaVA-SpaceSGG"
    / "dataset_pipeline"
    / "stage2"
)

sys.path.insert(0, str(CLIENT_DIR))

from iaedu_client import call_iaedu, load_dotenv


def normalize_options(raw_options):
    """Converte as opções do MMMU para uma lista."""

    if raw_options is None:
        return []

    if isinstance(raw_options, str):
        try:
            parsed = ast.literal_eval(raw_options)
        except (SyntaxError, ValueError):
            return [raw_options]

        if isinstance(parsed, (list, tuple)):
            return list(parsed)

        return [str(parsed)]

    if isinstance(raw_options, (list, tuple)):
        return list(raw_options)

    try:
        return list(raw_options)
    except TypeError:
        return [str(raw_options)]


def save_input_image(example, output_path):
    """Guarda uma imagem ou combina as várias imagens da pergunta."""

    images = []

    for number in range(1, 8):
        image = example.get(f"image_{number}")

        if image is not None:
            images.append(
                (number, image.convert("RGB"))
            )

    if not images:
        return None

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(images) == 1:
        images[0][1].save(output_path)
        return output_path

    prepared = []
    maximum_width = 1200
    maximum_height = 800

    for number, image in images:
        scale = min(
            1.0,
            maximum_width / image.width,
            maximum_height / image.height,
        )

        resized_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )

        resized_image = image.resize(
            resized_size
        )

        prepared.append(
            (number, resized_image)
        )

    label_height = 40

    canvas_width = max(
        image.width
        for _, image in prepared
    )

    canvas_height = sum(
        image.height + label_height
        for _, image in prepared
    )

    canvas = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)
    y_position = 0

    for number, image in prepared:
        draw.text(
            (10, y_position + 10),
            f"IMAGE {number}",
            fill="black",
        )

        y_position += label_height

        canvas.paste(
            image,
            (0, y_position),
        )

        y_position += image.height

    canvas.save(output_path)
    return output_path


def build_prompt(example):
    """Constrói um prompt segundo o formato do MMMU."""

    question = example["question"]
    question_type = example["question_type"]

    options = normalize_options(
        example.get("options")
    )

    if question_type == "multiple-choice":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        formatted_options = "\n".join(
            f"{letters[index]}) {option}"
            for index, option in enumerate(options)
        )

        return f"""MMMU expert-level multimodal question.

{question}

Options:
{formatted_options}

Examine all supplied images carefully.
Answer using only the letter of the correct option."""

    return f"""MMMU expert-level multimodal question.

{question}

Examine all supplied images carefully.
Answer using only a single word, number, or short phrase."""


def extract_prediction(
    response,
    question_type,
    option_count,
):
    """Extrai a resposta final devolvida pela IAedu."""

    response = str(response).strip()

    if question_type != "multiple-choice":
        return response

    if option_count <= 0:
        return "N/A"

    valid_letters = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:option_count]
    )

    clean_response = response.upper().strip()

    if (
        clean_response
        and clean_response in valid_letters
    ):
        return clean_response

    patterns = [
        (
            r"(?:FINAL\s+ANSWER|ANSWER|OPTION)"
            r"\s*[:\-]?\s*\(?([A-Z])\)?"
        ),
        r"^\s*\(?([A-Z])\)?[\.\)]?\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            clean_response,
            flags=re.MULTILINE,
        )

        valid_matches = [
            match
            for match in matches
            if match in valid_letters
        ]

        if valid_matches:
            return valid_matches[-1]

    return "N/A"


def save_results(
    predictions,
    details,
    output_path,
):
    """Guarda previsões e informação detalhada."""

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


def load_mmmu_dataset(subject, split):
    """
    Carrega o dataset oficial no Hugging Face.

    A mudança temporária de diretório impede que a pasta
    local MMMU/mmmu seja confundida com MMMU/MMMU.
    """

    original_directory = Path.cwd()

    try:
        os.chdir(MMMU_DIR / "mmmu")

        dataset = load_dataset(
            "MMMU/MMMU",
            subject,
            split=split,
        )
    finally:
        os.chdir(original_directory)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Executa o dataset oficial MMMU "
            "através da API IAedu."
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
        help="Divisão do dataset.",
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
        default=1,
        help="Número de exemplos.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout da API em segundos.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Ficheiro JSON de previsões.",
    )

    args = parser.parse_args()

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
            "IAEDU_CHANNEL_ID não está configurado."
        )

    dataset = load_mmmu_dataset(
        args.subject,
        args.split,
    )

    if args.start < 0:
        raise ValueError(
            "--start não pode ser negativo."
        )

    if args.limit <= 0:
        raise ValueError(
            "--limit deve ser superior a zero."
        )

    if args.start >= len(dataset):
        raise ValueError(
            f"--start={args.start} excede "
            f"o dataset com {len(dataset)} exemplos."
        )

    end = min(
        args.start + args.limit,
        len(dataset),
    )

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            MMMU_DIR
            / "resultados"
            / (
                f"iaedu_{args.subject}_"
                f"{args.split}.json"
            )
        )

    image_directory = (
        MMMU_DIR
        / "resultados"
        / "images"
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

        prompt = build_prompt(example)

        try:
            response = call_iaedu(
                message=prompt,
                channel_id=channel_id,
                thread_id=(
                    f"pedro-mmmu-{run_id}-"
                    f"{example_id}"
                ),
                user_info={
                    "name": "Pedro Vieira",
                    "project": "MMMU",
                },
                image_path=image_path,
                timeout=args.timeout,
            )

            prediction = extract_prediction(
                response=response,
                question_type=(
                    example["question_type"]
                ),
                option_count=len(options),
            )

            error = None

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
            prediction_text.lower()
            == ground_truth.lower()
        )

        predictions[example_id] = prediction

        details.append(
            {
                "id": example_id,
                "subject": args.subject,
                "split": args.split,
                "question_type": (
                    example["question_type"]
                ),
                "question": example["question"],
                "options": options,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "raw_response": response,
                "image_path": (
                    str(image_path)
                    if image_path
                    else None
                ),
                "error": error,
            }
        )

        print("Previsão:", prediction)
        print("Correta:", ground_truth)

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
    print(f"Resultado: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Previsões: {output_path}")
    print(f"Detalhes: {details_path}")


if __name__ == "__main__":
    main()