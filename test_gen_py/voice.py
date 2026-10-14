"""Озвучка generator/story.txt через Fish Audio, без генерации картинок."""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


GENERATOR_DIR = Path(__file__).resolve().parents[1] / "generator"
TTS_URL = "https://api.fish.audio/v1/tts"
MAX_AUDIO_BYTES = 32 * 1024 * 1024


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Неверный формат .env в строке {number}")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Незакрытые кавычки в .env в строке {number}")
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError("Текст для озвучки не должен быть пустым")
    return text


def read_config() -> dict[str, str]:
    key = os.environ.get("FISH_API_KEY", "")
    if not key.strip():
        raise ValueError("Задайте FISH_API_KEY в generator/.env или окружении")
    return {
        "api_key": key,
        "reference_id": os.environ.get("FISH_REFERENCE_ID")
        or "8336207c3431432cb60e8851e78ba437",
        "model": os.environ.get("FISH_MODEL") or "s2.1-pro-free",
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_audio(text: str, config: dict[str, str]) -> bytes:
    body = json.dumps({
        "text": text,
        "reference_id": config["reference_id"],
        "format": "mp3",
    }, ensure_ascii=False).encode("utf-8")
    request = Request(TTS_URL, data=body, headers={
        "Authorization": "Bearer " + config["api_key"],
        "Content-Type": "application/json",
        "model": config["model"],
    }, method="POST")
    # Не пересылаем ключ при редиректе; повторов платного запроса нет.
    opener = build_opener(NoRedirect())
    try:
        with opener.open(request, timeout=180) as response:
            if response.status != 200:
                raise ValueError(f"Fish Audio вернул HTTP {response.status}")
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" in content_type or "text/" in content_type:
                raise ValueError("Fish Audio вернул текст вместо MP3")
            audio = response.read(MAX_AUDIO_BYTES + 1)
    except HTTPError as error:
        error.close()
        raise ValueError(f"Fish Audio вернул HTTP {error.code}") from None
    except URLError as error:
        raise ValueError(f"Не удалось подключиться к Fish Audio: {error.reason}") from None
    if not audio or len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("Fish Audio вернул пустое или слишком большое аудио")
    return audio


def save_audio(path: Path, audio: bytes) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".voice-", suffix=".mp3", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(audio)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    load_env(GENERATOR_DIR / ".env")
    text = read_text(GENERATOR_DIR / "story.txt")
    output = GENERATOR_DIR / "voice.mp3"
    if output.exists():
        if not output.is_file() or output.stat().st_size == 0:
            raise ValueError("voice.mp3 должен быть непустым обычным файлом; проверьте его вручную")
        print("voice.mp3 уже существует — генерация пропущена. Для нового текста переместите или удалите файл.")
        return
    config = read_config()
    print("Генерируем озвучку через Fish Audio...", flush=True)
    audio = request_audio(text, config)
    save_audio(output, audio)
    print(f"Готово: {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nГенерация прервана.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)
