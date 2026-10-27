"""Чтение .env и переключателей с правилами исходной Go-версии."""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import os
from pathlib import Path
import re


ENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
TRUE_VALUES = frozenset(("1", "t", "T", "true", "TRUE", "True"))
FALSE_VALUES = frozenset(("0", "f", "F", "false", "FALSE", "False"))


@dataclass(frozen=True)
class GenerationConfig:
    generate_voice: bool = True
    generate_images: bool = True


def load_env(path: Path, environ: MutableMapping[str, str] | None = None) -> None:
    """Существующее окружение, включая пустые значения, имеет приоритет."""
    target = os.environ if environ is None else environ
    try:
        content = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return
    except UnicodeDecodeError:
        raise ValueError(".env должен быть в кодировке UTF-8") from None

    for number, line in enumerate(content.split("\n"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or ENV_KEY.fullmatch(key) is None:
            raise ValueError(f"неверный формат .env в строке {number}; ожидается KEY=value")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"незакрытые кавычки в .env в строке {number}")
            value = value[1:-1]
        if key not in target:
            try:
                target[key] = value
            except ValueError:
                raise ValueError(f"неверное значение переменной в .env в строке {number}") from None


def read_bool_env(name: str, environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    if name not in source:
        return True
    value = source[name].strip()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"{name} должен быть true или false")


def read_config(environ: Mapping[str, str] | None = None) -> GenerationConfig:
    return GenerationConfig(
        generate_voice=read_bool_env("GENERATE_VOICE", environ),
        generate_images=read_bool_env("GENERATE_IMAGES", environ),
    )
