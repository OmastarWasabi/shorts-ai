"""Чтение пронумерованных многострочных промптов с правилами Go-версии."""

from dataclasses import dataclass
from pathlib import Path
import re
import sys

from .files import read_utf8


MARKER = re.compile(r"\[([0-9]{2,})\]")


@dataclass(frozen=True)
class Prompt:
    number: int
    text: str


def _finish_prompt(number: int, lines: list[str]) -> Prompt:
    text = "\n".join(lines).strip()
    if not text:
        raise ValueError(f"промпт [{number:02d}] пуст")
    return Prompt(number, text)


def read_prompts(path: Path) -> list[Prompt]:
    content = read_utf8(path)
    prompts = []
    lines = []
    number = 0
    for line_index, line in enumerate(content.split("\n"), 1):
        line = line.removesuffix("\r")
        trimmed = line.strip()
        marker = MARKER.fullmatch(trimmed)
        if marker is not None:
            if number:
                prompts.append(_finish_prompt(number, lines))
                lines = []
            try:
                next_number = int(marker[1])
            except ValueError:
                next_number = -1
            if next_number > sys.maxsize or next_number != number + 1:
                raise ValueError(
                    f"строка {line_index}: ожидался маркер [{number + 1:02d}]; "
                    "проверьте порядок и дубликаты"
                )
            number = next_number
            continue
        if len(trimmed) > 1 and trimmed[0] == "[" and trimmed[1] in "0123456789":
            raise ValueError(f"неверный маркер в строке {line_index}; используйте [01] на отдельной строке")
        if not number:
            if trimmed:
                raise ValueError("prompts.txt должен начинаться с маркера [01]")
            continue
        lines.append(line)
    if not number:
        raise ValueError("prompts.txt не содержит промптов")
    prompts.append(_finish_prompt(number, lines))
    for prompt in prompts:
        if len(prompt.text) > 32000:
            raise ValueError(f"промпт [{prompt.number:02d}] превышает 32000 символов")
    return prompts
