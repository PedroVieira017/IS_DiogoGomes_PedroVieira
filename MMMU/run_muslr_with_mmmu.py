import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

from run_iaedu_mmmu import (
    MMMU_DIR,
    PROJECT_ROOT,
    call_iaedu,
    extract_prediction,
    load_dotenv,
    normalize_options,
)


DEFAULT_METADATA = (
    PROJECT_ROOT
    / "MuSLR"
    / "MuSLR-Dataset"
    / "metadata.csv"
)


def load_records(metadata_path):
    """Carrega o metadata.csv oficial do MuSLR."""

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def clean_choice(choice):
    """Remove etiquetas como A., B), C: ou D-."""

    return re.sub(
        r"^\s*[A-Z]\s*[\.\):\-]\s*",
        "",
        str(choice),
    ).strip()

def get_record_options(record):
    """
    Obtém as opções do exemplo.

    Nas tarefas de avaliação de verdade, o MuSLR deixa
    o campo choices vazio. Nesses casos são usadas as
    opções True, False e Unknown.
    """

    options = normalize_options(
        record.get("choices")
    )

    options = [
        str(option).strip()
        for option in options
        if str(option).strip()
    ]

    if not options:
        return [
            "True",
            "False",
            "Unknown",
        ]

    return options

def answer_to_letter(answer, options):
    """
    Converte a resposta correta para uma letra.

    Alguns exemplos usam B ou D, enquanto outros usam
    diretamente True, False ou Unknown.
    """

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    valid_letters = letters[:len(options)]
    raw_answer = str(answer).strip()

    if raw_answer.upper() in valid_letters:
        return raw_answer.upper()

    cleaned_answer = clean_choice(
        raw_answer
    ).casefold()

    for index, option in enumerate(options):
        cleaned_option = clean_choice(
            option
        ).casefold()

        if cleaned_option == cleaned_answer:
            return letters[index]

    return raw_answer.upper()


def response_to_letter(response, options):
    """Converte a resposta do modelo para a letra da opção."""

    prediction = extract_prediction(
        response=response,
        question_type="multiple-choice",
        option_count=len(options),
    )

    if prediction != "N/A":
        return prediction

    clean_response = clean_choice(
        str(response)
    ).strip().casefold()

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for index, option in enumerate(options):
        clean_option = clean_choice(
            option
        ).casefold()

        if clean_response == clean_option:
            return letters[index]

        answer_patterns = [
            f"answer: {clean_option}",
            f"answer is {clean_option}",
            f"final answer: {clean_option}",
            f"final answer is {clean_option}",
        ]

        if any(
            pattern in clean_response
            for pattern in answer_patterns
        ):
            return letters[index]

    return "N/A"


def build_mmmu_prompt(record, options):
    """
    Cria um prompt direto ao estilo do MMMU.

    Não utiliza os módulos especializados do LogiCAM.
    """

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    formatted_options = "\n".join(
        f"{letters[index]}) {clean_choice(option)}"
        for index, option in enumerate(options)
    )

    return f"""Cross-dataset evaluation: MuSLR data using an MMMU-style direct multimodal prompt.

Logical context:
{record["full_context"]}

Question:
{record["question"]}

Options:
{formatted_options}

Use both the image and the logical context.
Select the option that can be correctly derived.
Answer using only the letter of the correct option."""


def save_results(
    predictions,
    details,
    output_path,
):
    """Guarda previsões e detalhes da experiência."""

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
            "Testa o dataset MuSLR/LogiCAM "
            "com o método direto usado no MMMU."
        )
    )

    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA),
        help="Caminho para o metadata.csv do MuSLR.",
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
            MMMU_DIR
            / "resultados"
            / "pedro_muslr_com_mmmu.json"
        ),
        help="Ficheiro JSON de previsões.",
    )

    args = parser.parse_args()

    metadata_path = Path(
        args.metadata
    ).resolve()

    output_path = Path(
        args.output
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata nao encontrado: {metadata_path}"
        )

    records = load_records(
        metadata_path
    )

    dataset_root = metadata_path.parent

    if args.start < 0:
        raise ValueError(
            "--start nao pode ser negativo."
        )

    if args.limit <= 0:
        raise ValueError(
            "--limit deve ser superior a zero."
        )

    if args.start >= len(records):
        raise ValueError(
            f"--start={args.start} excede "
            f"os {len(records)} exemplos."
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

    end = min(
        args.start + args.limit,
        len(records),
    )

    selected_records = records[
        args.start:end
    ]

    predictions = {}
    details = []

    run_id = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    for position, record in enumerate(
        selected_records,
        start=1,
    ):
        example_id = record["id"]

        print(
            f"[{position}/{len(selected_records)}] "
            f"{example_id}"
        )

        options = get_record_options(
            record
        )

        prediction = "N/A"
        response = ""
        error = None
        image_path = None

        if not options:
            error = (
                "O exemplo nao contem opcoes."
            )
        else:
            file_name = str(
                record.get("file_name", "")
            ).strip()

            if not file_name:
                error = (
                    "O exemplo nao indica uma imagem."
                )
            else:
                image_path = (
                    dataset_root
                    / Path(file_name)
                ).resolve()

                if not image_path.is_file():
                    error = (
                        "Imagem nao encontrada: "
                        f"{image_path}"
                    )
                else:
                    prompt = build_mmmu_prompt(
                        record,
                        options,
                    )

                    try:
                        response = call_iaedu(
                            message=prompt,
                            channel_id=channel_id,
                            thread_id=(
                                "pedro-cross-mmmu-muslr-"
                                f"{run_id}-{example_id}"
                            ),
                            user_info={
                                "name": "Pedro Vieira",
                                "project": (
                                    "MMMU tested with "
                                    "MuSLR/LogiCAM dataset"
                                ),
                            },
                            image_path=image_path,
                            timeout=args.timeout,
                        )

                        prediction = (
                            response_to_letter(
                                response,
                                options,
                            )
                        )

                    except Exception as exception:
                        response = ""
                        prediction = "N/A"
                        error = str(exception)

        ground_truth = answer_to_letter(
            record["answer"],
            options,
        )

        prediction_text = str(
            prediction
        ).strip().upper()

        is_correct = (
            prediction_text == ground_truth
        )

        predictions[example_id] = prediction

        details.append(
            {
                "id": example_id,
                "experiment": (
                    "MuSLR/LogiCAM dataset "
                    "with MMMU-style direct prompting"
                ),
                "domain": record.get("domain"),
                "symbol": record.get("symbol"),
                "depth": record.get("depth"),
                "question": record.get("question"),
                "context": record.get(
                    "full_context"
                ),
                "options": options,
                "prediction": prediction,
                "ground_truth_raw": (
                    record.get("answer")
                ),
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
        "MuSLR/LogiCAM -> MMMU"
    )
    print(f"Resultado: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Previsoes: {output_path}")
    print(f"Detalhes: {details_path}")


if __name__ == "__main__":
    main()