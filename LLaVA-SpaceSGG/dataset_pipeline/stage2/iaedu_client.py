import base64
import json
import mimetypes
import os
import re
import socket
import http.client
import urllib.error
import urllib.request
import uuid
from pathlib import Path


DEFAULT_ENDPOINT = "https://api.iaedu.pt/agent-chat/api/v1/agent/cmor5objoex9gfp01vm7p95jh/stream"
DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"


def load_dotenv(env_path=None):
    env_path = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


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
        parts = []
        for choice in choices:
            text = _extract_text(choice)
            if text:
                parts.append(text)
        return "".join(parts)

    return ""


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


def call_iaedu(message, channel_id, thread_id, user_info=None, endpoint=None, api_key=None, timeout=120, image_path=None):
    load_dotenv()
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    endpoint = endpoint or os.environ.get("OPENAI_API_ENDPOINT") or DEFAULT_ENDPOINT
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not channel_id:
        raise RuntimeError("channel_id is required for the IAedu agent API.")

    form = {
        "message": message,
        "thread_id": thread_id,
        "channel_id": channel_id,
        "user_info": json.dumps(
    user_info or {"name": "Pedro Vieira"},
    ensure_ascii=False,
),
    }
    files = {"files": image_path} if image_path else None
    body, boundary = _encode_multipart_form(form, files=files)
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "x-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            try:
                raw_bytes = response.read()
            except http.client.IncompleteRead as error:
                raw_bytes = error.partial
            raw = raw_bytes.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"IAedu API error {error.code}: {detail}") from error

    return _parse_stream(raw)


def call_ollama_vision(message, image_path, model=None, endpoint=None, timeout=None):
    load_dotenv()
    model = model or os.environ.get("OLLAMA_VISION_MODEL") or os.environ.get("OLLAMA_MODEL") or "llava"
    endpoint = endpoint or os.environ.get("OLLAMA_ENDPOINT") or DEFAULT_OLLAMA_ENDPOINT
    timeout = int(timeout or os.environ.get("OLLAMA_TIMEOUT", "600"))

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image_base64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "prompt": message,
        "images": [image_base64],
        "stream": True,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parts = []
            for raw_line in response:
                if not raw_line:
                    continue

                payload = json.loads(raw_line.decode("utf-8", errors="replace"))
                token = payload.get("response", "")
                if token:
                    print(token, end="", flush=True)
                    parts.append(token)

                if payload.get("done"):
                    break
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error {error.code}: {detail}") from error
    except socket.timeout as error:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s. Try a smaller image/prompt or set OLLAMA_TIMEOUT to a larger value."
        ) from error

    return "".join(parts).strip()
