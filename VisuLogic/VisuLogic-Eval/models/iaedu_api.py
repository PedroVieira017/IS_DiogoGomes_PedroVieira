import hashlib
import http.client
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict

from models.base_model import BaseModel


DEFAULT_ENDPOINT = "https://api.iaedu.pt/agent-chat/api/v1/agent/cmor5objoex9gfp01vm7p95jh/stream"


def load_dotenv(env_path=None):
    env_path = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _extract_text(payload):
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return ""

    for key in ("content", "text", "message", "response", "delta", "answer", "body"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = _extract_text(value)
            if nested:
                return nested

    choices = payload.get("choices")
    if isinstance(choices, list):
        return "".join(filter(None, (_extract_text(choice) for choice in choices)))

    return ""


def _clean_response(text):
    text = re.sub(r"^Processing\s*", "", text.strip())
    text = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "",
        text,
    ).strip()

    half = len(text) // 2
    if len(text) % 2 == 0 and text[:half] == text[half:]:
        text = text[:half].strip()

    return text


def _parse_stream(raw_response):
    parts = []
    for raw_line in raw_response.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            parts.append(line)
            continue
        text = _extract_text(payload)
        if text:
            parts.append(text)

    return _clean_response("".join(parts).strip() or raw_response.strip())


def _encode_multipart_form(fields, files=None):
    boundary = f"----iaedu-{uuid.uuid4().hex}"
    chunks = []

    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    for key, file_path in (files or {}).items():
        path = Path(file_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{key}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


class IAeduAPIModel(BaseModel):
    def __init__(
        self,
        model_name: str = "iaedu",
        api_key: str = None,
        endpoint: str = None,
        channel_id: str = None,
        thread_id: str = None,
        user_prompt: str = None,
        timeout: int = 120,
    ):
        load_dotenv()
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("OPENAI_API_ENDPOINT") or DEFAULT_ENDPOINT
        self.channel_id = channel_id or os.environ.get("IAEDU_CHANNEL_ID")
        self.thread_id = thread_id or os.environ.get("IAEDU_THREAD_ID") or "visulogic"
        self.user_prompt = user_prompt or ""
        self.timeout = int(timeout or os.environ.get("IAEDU_TIMEOUT", "120"))

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set for IAedu.")
        if not self.channel_id:
            raise RuntimeError("IAEDU_CHANNEL_ID is not set for IAedu.")

    @property
    def name(self) -> str:
        return "iaedu"

    def predict(self, input_data: Dict) -> str:
        image_path = Path(input_data["image_path"])
        question = input_data["text"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        prompt = (
            f"{question}\n\n"
            f"{self.user_prompt}\n\n"
            "Answer with only one final option: A, B, C, or D."
        )

        return self._call_iaedu(prompt, image_path)

    def _call_iaedu(self, message: str, image_path: Path) -> str:
        request_hash = hashlib.sha1(f"{image_path}|{message}".encode("utf-8")).hexdigest()[:8]
        item_thread_id = f"{self.thread_id}-{image_path.stem}-{request_hash}"
        form = {
            "message": message,
            "thread_id": item_thread_id,
            "channel_id": self.channel_id,
            "user_info": json.dumps({"name": "VisuLogic"}, ensure_ascii=False),
        }
        body, boundary = _encode_multipart_form(form, files={"files": image_path})
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "x-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                try:
                    raw_bytes = response.read()
                except http.client.IncompleteRead as error:
                    raw_bytes = error.partial
                raw = raw_bytes.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"IAedu API error {error.code}: {detail}") from error

        return _parse_stream(raw)
