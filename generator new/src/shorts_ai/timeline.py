"""Формат timeline v1, хеш озвучки и проверки, перенесённые из Go."""

import hashlib
import json
import math
from pathlib import Path
import stat
import sys

from .ffmpeg import FPS

MAX_TIMELINE_BYTES = 16 << 20


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_half_away(value: float) -> int:
    """math.Round из Go: половины округляются от нуля, без bankers rounding."""
    absolute = abs(value)
    whole = math.floor(absolute)
    rounded = whole + (absolute - whole >= .5)
    return -rounded if value < 0 else rounded


def _fields(value: object, fields: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Неверные или отсутствующие поля JSON таймингов")


def _number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_timeline(timing: dict, duration: float, count: int) -> list[int]:
    _fields(timing, {"version", "audio_sha256", "duration", "scenes", "words"})
    if not _number(duration) or duration <= 0 or duration * FPS >= float(sys.maxsize):
        raise ValueError("Неверная длительность озвучки")
    if (type(timing["version"]) is not int or timing["version"] != 1 or
            not _number(timing["duration"]) or abs(timing["duration"] - duration) > .01):
        raise ValueError("Неверная версия или длительность таймингов; пересоздайте timeline.json")
    if not isinstance(timing["audio_sha256"], str):
        raise ValueError("Неверный audio_sha256")
    scenes, words = timing["scenes"], timing["words"]
    if type(count) is not int or count < 1 or not isinstance(scenes, list) or len(scenes) != count:
        raise ValueError("Количество сцен в timeline.json должно совпадать с количеством картинок")
    if not isinstance(words, list):
        raise ValueError("В timeline.json нужен список words")
    frames = []
    previous_end, previous_frame = 0.0, 0
    for index, scene in enumerate(scenes):
        _fields(scene, {"number", "text", "start", "end"})
        if (type(scene["number"]) is not int or scene["number"] != index + 1 or
                not isinstance(scene["text"], str) or not scene["text"].strip() or
                not _number(scene["start"]) or not _number(scene["end"]) or
                abs(scene["start"] - previous_end) > .001 or scene["end"] <= scene["start"] or
                scene["end"] > duration + .001):
            raise ValueError(f"Сцена {index + 1}: неверный номер, текст или границы; сцены должны идти подряд без разрывов")
        end_frame = round_half_away(scene["end"] * FPS)
        if index == count - 1:
            if abs(scene["end"] - duration) > .001:
                raise ValueError("Последняя сцена должна заканчиваться в конце озвучки")
            end_frame = math.ceil(duration * FPS)
        frames.append(end_frame - previous_frame)
        if frames[-1] < 1:
            raise ValueError(f"Сцена {index + 1} короче одного кадра")
        previous_end, previous_frame = scene["end"], end_frame
    seen = set()
    previous_start, previous_scene = -1.0, 1
    for word in words:
        _fields(word, {"text", "start", "end", "scene"})
        if (type(word["scene"]) is not int or not previous_scene <= word["scene"] <= count or
                not isinstance(word["text"], str) or not word["text"].strip() or
                not _number(word["start"]) or not _number(word["end"]) or word["start"] < 0 or
                word["end"] <= word["start"] or word["end"] > duration + .02 or word["start"] < previous_start):
            raise ValueError("В timeline.json неверные слова или порядок их таймингов")
        scene = scenes[word["scene"] - 1]
        if word["start"] < scene["start"] - .03 or word["end"] > scene["end"] + .03:
            raise ValueError(f"Слово «{word['text']}» выходит за границы сцены {word['scene']}")
        seen.add(word["scene"])
        previous_start, previous_scene = word["start"], word["scene"]
    for number in range(1, count + 1):
        if number not in seen:
            raise ValueError(f"Сцена {number} не содержит таймингов слов")
    return frames


def read_timeline(path: Path, audio: Path, duration: float, count: int) -> tuple[dict, list[int]]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TIMELINE_BYTES:
        raise ValueError("Тайминги должны быть обычным JSON-файлом размером до 16 МиБ")
    with path.open("rb") as source:
        content = source.read(MAX_TIMELINE_BYTES + 1)
    if len(content) > MAX_TIMELINE_BYTES:
        raise ValueError("Тайминги должны быть размером до 16 МиБ")
    try:
        timing = json.loads(content.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise ValueError("Неверный JSON таймингов или лишние данные после JSON") from None
    frames = validate_timeline(timing, duration, count)
    if timing["audio_sha256"] != file_hash(audio):
        raise ValueError("Тайминги относятся к другой озвучке; пересоздайте timeline.json")
    return timing, frames
