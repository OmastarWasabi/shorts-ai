"""Пути проекта, чтение UTF-8 и атомарная запись результата."""

import os
from pathlib import Path
import tempfile


def project_directory(path: Path) -> Path:
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Каталог проекта не существует или не является папкой: {directory}")
    return directory


def project_path(directory: Path, path: str | Path) -> Path:
    """Относительный путь отсчитывается от каталога данных, абсолютный сохраняется."""
    path = Path(path)
    return path if path.is_absolute() else directory / path


def read_utf8(path: Path) -> str:
    """Удаляет один начальный BOM; сохраняет CRLF и остальное содержимое."""
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError(f"{path.name} должен быть в кодировке UTF-8") from None


def write_bytes_atomic(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    """Публикует полностью записанный файл; по умолчанию существующий не заменяет."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".shorts-ai-", delete=False) as output:
            temporary_path = Path(output.name)
            output.write(data)
        if overwrite:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
