#!/usr/bin/env python3
"""Ask a free-form visual question about a user-selected image."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError


EVAL_DIR = Path(__file__).resolve().parent.parent
VISULOGIC_DIR = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from models import load_model


def _required_value(value, prompt, field_name):
    """Use a CLI value or request it interactively when it was omitted."""
    if value and value.strip():
        return value.strip()
    try:
        value = input(prompt).strip()
    except EOFError as exc:
        raise ValueError(f"{field_name} is required in non-interactive mode.") from exc
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")
    return value


def _validate_image(raw_path):
    # Quotes are useful in shell commands but are not part of a path pasted into input().
    image_path = Path(raw_path.strip().strip('"').strip("'")).expanduser()
    if not image_path.is_file():
        raise ValueError(f"Image not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"The selected file is not a valid image: {image_path}") from exc
    return image_path.resolve()


def _default_output_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return VISULOGIC_DIR / "resultados" / f"pergunta_manual_{timestamp}.json"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Use VisuLogic with any local image and a free-form question. "
            "When --image or --question is omitted, the value is requested in the terminal."
        )
    )
    parser.add_argument("--image", help="Path to a local image (PNG, JPEG, WEBP, etc.).")
    parser.add_argument("--question", help="Free-form question about the selected image.")
    parser.add_argument(
        "--model_path", "--model-path", default="iaedu",
        help="Vision backend/model (default: iaedu; example: ollama:llava).",
    )
    parser.add_argument(
        "--output_file", "--output-file",
        help="Optional JSON output path. By default it is saved in VisuLogic/resultados/.",
    )
    parser.add_argument("--api_key", "--api-key", default="", help="IAedu/API key.")
    parser.add_argument("--base_url", "--base-url", default="", help="IAedu or Ollama endpoint.")
    parser.add_argument("--channel_id", "--channel-id", default="", help="IAedu channel ID.")
    parser.add_argument("--thread_id", "--thread-id", default="visulogic-manual", help="IAedu thread prefix.")
    parser.add_argument("--api_timeout", "--api-timeout", type=int, default=180, help="API timeout in seconds.")
    parser.add_argument(
        "--user_prompt", "--user-prompt", default="",
        help="Optional extra response instruction for the model.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        raw_image = _required_value(args.image, "Image path: ", "image")
        question = _required_value(args.question, "Question about the image: ", "question")
        image_path = _validate_image(raw_image)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"\nImage: {image_path}")
    print(f"Question: {question}")
    print(f"Model: {args.model_path}\n")

    try:
        model = load_model(args)
        response = model.predict(
            {
                "image_path": str(image_path),
                "text": question,
                "response_mode": "free",
            }
        )
    except Exception as exc:
        print(f"VisuLogic request failed: {exc}", file=sys.stderr)
        return 1

    response = str(response).strip()
    if not response:
        print("VisuLogic returned an empty response.", file=sys.stderr)
        return 1
    if response.lower().startswith(("error in ollama prediction:", "error in prediction:")):
        print(f"VisuLogic request failed: {response}", file=sys.stderr)
        return 1

    output_path = Path(args.output_file).expanduser() if args.output_file else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image": str(image_path),
        "question": question,
        "model": getattr(model, "name", args.model_path),
        "response": response,
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    output_path.write_text(
        json.dumps(result, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("VisuLogic response:\n")
    print(response)
    print(f"\nResult saved to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
