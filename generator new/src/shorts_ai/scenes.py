"""Разметка сценария: один абзац соответствует одному пронумерованному промпту."""

import json
from pathlib import Path
import re

from .files import read_utf8, write_bytes_atomic
from .prompts import read_prompts


def create_scenes(script: Path, prompts: Path, output: Path) -> dict:
    text = read_utf8(script).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"{script.name}: сценарий не должен быть пустым")
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    numbered_prompts = read_prompts(prompts)
    if len(paragraphs) != len(numbered_prompts):
        raise ValueError(
            f"Авторазметка: в {script.name} {len(paragraphs)} абзацев, "
            f"а в {prompts.name} {len(numbered_prompts)} промптов. "
            "Разделите сценарий пустыми строками: один абзац на один промпт, "
            "или подготовьте scenes.json вручную."
        )
    if any(not any(character.isalnum() for character in paragraph) for paragraph in paragraphs):
        raise ValueError("Авторазметка: каждый абзац должен содержать слова")
    document = {
        "version": 1,
        "language": "ru",
        "scenes": [{"number": number, "text": paragraph} for number, paragraph in enumerate(paragraphs, 1)],
    }
    data = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_bytes_atomic(output, data)
    return document
