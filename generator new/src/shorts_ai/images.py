"""OpenAI Images: запросы, ограниченные повторы, проверка и сохранение PNG."""

import asyncio
import base64
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from http import HTTPStatus
import io
import json
import os
from pathlib import Path
import random
import re
import stat
import time
from datetime import timezone

import httpx
from PIL import Image

from .files import write_bytes_atomic
from .transport import HTTPResponse, REQUEST_TIMEOUT, request_limited


IMAGES_URL = "https://api.openai.com/v1/images/generations"
MAX_RESPONSE_BYTES = 48 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
IMAGE_TIMEOUT = 600.0


class ImageAPIError(ValueError):
    def __init__(self, status: int, code: str = "", kind: str = "", retry_after: str = ""):
        self.status = status
        self.code = code
        self.kind = kind
        self.retry_after = retry_after
        try:
            reason = f" ({HTTPStatus(status).phrase})"
        except ValueError:
            reason = ""
        super().__init__(f"OpenAI Images вернул HTTP {status}{reason}")


def read_config(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    key = source.get("OPENAI_API_KEY", "")
    if not key.strip():
        raise ValueError("задайте OPENAI_API_KEY в окружении или .env")
    if not key.isascii() or any(ord(char) < 32 or ord(char) == 127 for char in key):
        raise ValueError("OPENAI_API_KEY содержит недопустимые символы")
    config = {
        "api_key": key,
        "model": source.get("OPENAI_IMAGE_MODEL") or "gpt-image-2.5-sunburst",
        "size": source.get("OPENAI_IMAGE_SIZE") or "1024x1536",
        "quality": source.get("OPENAI_IMAGE_QUALITY") or "low",
    }
    if config["size"] not in ("1024x1024", "1024x1536", "1536x1024"):
        raise ValueError("OPENAI_IMAGE_SIZE должен быть 1024x1024, 1024x1536 или 1536x1024")
    if config["quality"] not in ("low", "medium", "high"):
        raise ValueError("OPENAI_IMAGE_QUALITY должен быть low, medium или high")
    return config


def _verify_png_end(data: bytes) -> None:
    # Pillow verify() не проверяет CRC завершающего IEND: проверяем сам блок.
    position = 8
    while position + 12 <= len(data):
        length = int.from_bytes(data[position:position + 4], "big")
        end = position + 12 + length
        if end > len(data):
            break
        if data[position + 4:position + 8] == b"IEND":
            if data[position:end] == b"\x00\x00\x00\x00IEND\xae\x42\x60\x82":
                return
            break
        position = end
    raise ValueError("повреждённый PNG")


def validate_png(data: bytes) -> None:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("PNG пуст или превышает 32 МиБ")
    try:
        image = Image.open(io.BytesIO(data), formats=["PNG"])
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("некорректный заголовок PNG") from None
    with image:
        if image.width > 4096 or image.height > 4096:
            raise ValueError("PNG превышает допустимые размеры 4096×4096")
        try:
            image.verify()
            _verify_png_end(data)
            with Image.open(io.BytesIO(data), formats=["PNG"]) as pixels:
                pixels.load()
        except (OSError, ValueError, SyntaxError):
            raise ValueError("повреждённый PNG") from None


def image_exists(path: Path) -> bool:
    try:
        info = path.stat()
    except FileNotFoundError:
        if path.is_symlink():
            raise ValueError(f"{path.name}: повреждённая ссылка; проверьте файл вручную") from None
        return False
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_IMAGE_BYTES:
        raise ValueError(f"{path.name} должен быть обычным PNG-файлом размером до 32 МиБ")
    with path.open("rb") as source:
        data = source.read(MAX_IMAGE_BYTES + 1)
    try:
        validate_png(data)
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}; проверьте или удалите файл вручную") from None
    return True


def save_image(path: Path, data: bytes) -> None:
    validate_png(data)
    write_bytes_atomic(path, data)


def _invalid_json_constant(value: str) -> None:
    raise ValueError("Некорректное JSON-число")


def _api_error(response: HTTPResponse) -> ImageAPIError:
    code, kind = "", ""
    try:
        document, _ = json.JSONDecoder(parse_constant=_invalid_json_constant).raw_decode(
            response.body.decode("utf-8").lstrip()
        )
        error = document.get("error") if isinstance(document, dict) else None
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else ""
            kind = error.get("type") if isinstance(error.get("type"), str) else ""
    except (ValueError, RecursionError):
        pass
    return ImageAPIError(response.status_code, code, kind, response.headers.get("Retry-After", ""))


def retry_delay(header: str, attempt: int, *, now: float | None = None) -> float:
    delay = (1 << attempt) + random.randrange(250) / 1000
    if re.fullmatch(r"[0-9]+", header):
        try:
            seconds = int(header)
        except ValueError:
            return delay
        if seconds <= (1 << 64) - 1:
            return max(delay, min(seconds, ((1 << 63) - 1) / 1_000_000_000))
    try:
        date = parsedate_to_datetime(header)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(delay, date.timestamp() - (time.time() if now is None else now))
    except (ValueError, TypeError, OverflowError):
        return delay


async def request_image_once(
    client: httpx.AsyncClient, text: str, config: Mapping[str, str], *, timeout: float = REQUEST_TIMEOUT,
) -> bytes:
    try:
        response = await request_limited(
            client, "POST", IMAGES_URL, max_bytes=MAX_RESPONSE_BYTES, error_max_bytes=64 * 1024, timeout=timeout,
            headers={"Authorization": "Bearer " + config["api_key"], "Content-Type": "application/json"},
            json={
                "model": config["model"], "prompt": text, "size": config["size"], "quality": config["quality"],
                "n": 1, "output_format": "png",
            },
        )
    except UnicodeError:
        raise ValueError("OpenAI Images: не удалось закодировать данные запроса") from None
    except ValueError as error:
        raise ValueError(f"OpenAI Images: {error}") from None
    if response.status_code != 200:
        raise _api_error(response)
    try:
        payload = json.loads(response.body.decode("utf-8"), parse_constant=_invalid_json_constant)
    except (ValueError, RecursionError):
        raise ValueError("OpenAI Images вернул некорректный JSON") from None
    data = payload.get("data") if isinstance(payload, dict) else None
    if (not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict)
            or not isinstance(data[0].get("b64_json"), str) or not data[0]["b64_json"]):
        raise ValueError("OpenAI Images не вернул одно изображение в b64_json")
    try:
        encoded = data[0]["b64_json"].replace("\r", "").replace("\n", "")
        return base64.b64decode(encoded, validate=True)
    except ValueError:
        raise ValueError("OpenAI Images вернул некорректный Base64") from None


async def request_image(
    client: httpx.AsyncClient, text: str, config: Mapping[str, str], *,
    request_timeout: float = REQUEST_TIMEOUT, total_timeout: float = IMAGE_TIMEOUT,
) -> bytes:
    if total_timeout <= 0 or not (total_timeout < float("inf")):
        raise ValueError("Нужен положительный конечный таймаут изображения")
    try:
        async with asyncio.timeout(total_timeout):
            for attempt in range(3):
                try:
                    data = await request_image_once(client, text, config, timeout=request_timeout)
                    await asyncio.sleep(0)
                    return data
                except ImageAPIError as error:
                    if (error.status != 429 or attempt == 2
                            or error.code == "insufficient_quota" or error.kind == "insufficient_quota"
                            or error.code.startswith("billing_") or error.kind.startswith("billing_")):
                        raise
                    await asyncio.sleep(retry_delay(error.retry_after, attempt))
    except TimeoutError:
        raise ValueError("OpenAI Images: истекло время генерации изображения") from None
