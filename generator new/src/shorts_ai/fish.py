"""Озвучка Fish Audio: текст, настройки, запрос и сохранение MP3."""

from collections.abc import Mapping
from http import HTTPStatus
import os
from pathlib import Path
import stat

import httpx

from .files import read_utf8, write_bytes_atomic
from .transport import REQUEST_TIMEOUT, request_limited


TTS_URL = "https://api.fish.audio/v1/tts"
MAX_AUDIO_BYTES = 32 * 1024 * 1024
MODELS = frozenset(("s1", "s2-pro", "s2.1-pro", "s2.1-pro-free", "drama-3-preview"))


def read_text(path: Path) -> str:
    text = read_utf8(path)
    if not text.strip():
        raise ValueError("текст для озвучки не должен быть пустым")
    return text


def read_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    key = source.get("FISH_API_KEY", "")
    if not key.strip():
        raise ValueError("задайте FISH_API_KEY в окружении или .env")
    if not key.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValueError("FISH_API_KEY содержит недопустимые символы")
    model = source.get("FISH_MODEL") or "s2.1-pro-free"
    if model not in MODELS:
        raise ValueError("неподдерживаемый FISH_MODEL; проверьте название модели в документации Fish Audio")
    return {
        "api_key": key,
        "reference_id": source.get("FISH_REFERENCE_ID") or "8336207c3431432cb60e8851e78ba437",
        "model": model,
    }


def audio_exists(path: Path) -> bool:
    try:
        info = path.stat()
    except FileNotFoundError:
        if path.is_symlink():
            raise ValueError(f"{path.name}: повреждённая ссылка; проверьте файл вручную") from None
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError(f"{path.name} должен быть непустым обычным файлом; проверьте его вручную")
    return True


async def request_audio(
    client: httpx.AsyncClient,
    text: str,
    config: Mapping[str, str],
    *,
    timeout: float = REQUEST_TIMEOUT,
) -> bytes:
    try:
        response = await request_limited(
            client, "POST", TTS_URL, max_bytes=MAX_AUDIO_BYTES, timeout=timeout,
            read_error_body=False,
            headers={
                "Authorization": "Bearer " + config["api_key"],
                "Content-Type": "application/json",
                "model": config["model"],
            },
            json={"text": text, "reference_id": config["reference_id"], "format": "mp3"},
        )
    except UnicodeError:
        raise ValueError("Fish Audio: не удалось закодировать данные запроса") from None
    except ValueError as error:
        raise ValueError(f"Fish Audio: {error}") from None
    if response.status_code != 200:
        try:
            reason = f" ({HTTPStatus(response.status_code).phrase})"
        except ValueError:
            reason = ""
        raise ValueError(f"Fish Audio вернул HTTP {response.status_code}{reason}")
    content_type = response.headers.get("Content-Type", "").lower()
    if "json" in content_type or "text/" in content_type:
        raise ValueError("Fish Audio вернул текст вместо MP3")
    if not response.body:
        raise ValueError("Fish Audio вернул пустое аудио")
    return response.body


def save_audio(path: Path, audio: bytes) -> None:
    write_bytes_atomic(path, audio)
