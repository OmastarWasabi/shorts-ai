"""Монтаж PNG и озвучки: равномерные сцены или тайминги речи с ASS-субтитрами."""

import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

from .captions import write_captions
from .ffmpeg import FPS, audio_duration, find_tools, mux, render_scene, require_ass
from .timeline import file_hash, read_timeline


def read_images(directory: Path) -> list[Path]:
    by_number: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".png" or re.fullmatch(r"[0-9]+", path.stem) is None:
            continue
        try:
            number = int(path.stem)
        except ValueError:
            number = 0
        if number < 1 or number > sys.maxsize:
            raise ValueError(f"Неверный номер изображения {path.name}")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"{path.name} должен быть обычным PNG-файлом")
        if number in by_number:
            raise ValueError(f"Одинаковый номер у {by_number[number].name} и {path.name}")
        by_number[number] = path.absolute()
    if not by_number:
        raise ValueError("Нет изображений: ожидаются 01.png, 02.png и далее")
    images = []
    for expected, number in enumerate(sorted(by_number), 1):
        if number != expected:
            raise ValueError(f"Пропущено изображение {expected:02d}.png")
        images.append(by_number[number])
    return images


def split_frames(duration: float, scenes: int) -> list[int]:
    total = duration * FPS
    if scenes < 1 or not math.isfinite(total) or total <= 0 or total >= float(sys.maxsize):
        raise ValueError("Неверная длительность озвучки или количество сцен")
    count = math.ceil(total)
    if count < scenes:
        raise ValueError("Озвучка слишком короткая: на изображение должно приходиться хотя бы 1/30 секунды")
    frames, remainder = divmod(count, scenes)
    return [frames + (index < remainder) for index in range(scenes)]


def render(images_directory: Path, audio: Path, output: Path, *,
           timeline: Path | None = None, subtitles: bool = True) -> None:
    if output.suffix.lower() != ".mp4":
        raise ValueError("Выходной файл должен иметь расширение .mp4")
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"{output} уже существует; выберите другой --output или переместите старый файл")
    encoder, probe = find_tools()
    images = read_images(images_directory)
    audio = audio.absolute()
    duration = audio_duration(audio, binary=probe)
    timing = None
    if timeline is not None:
        timing, frames = read_timeline(timeline, audio, duration, len(images))
    else:
        frames = split_frames(duration, len(images))
    burn_subtitles = timing is not None and subtitles
    if burn_subtitles:
        require_ass(encoder)
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".montage-") as temporary:
        work = Path(temporary)
        if burn_subtitles:
            write_captions(work / "captions.ass", timing["words"])
        listing = []
        for index, (image, count) in enumerate(zip(images, frames)):
            name = f"clip_{index + 1:04d}.mp4"
            print(f"Монтаж: кадр {index + 1}/{len(images)}, {count / FPS:.2f} сек.", file=sys.stderr)
            try:
                render_scene(encoder, image, count, index, work / name)
            except (OSError, ValueError) as error:
                raise ValueError(f"Кадр {index + 1} ({image.name}): {error}") from error
            # В concat попадают только созданные нами относительные имена.
            listing.append(f"file '{name}'\n")
        list_path = work / "scenes.txt"
        list_path.write_text("".join(listing), encoding="utf-8")
        final = work / "final.mp4"
        mux(encoder, list_path, audio, duration, final, subtitles=burn_subtitles)
        if timing is not None and file_hash(audio) != timing["audio_sha256"]:
            raise ValueError("Озвучка изменилась во время монтажа; пересоздайте тайминги")
        # На одной файловой системе: публикация целиком, без перезаписи даже при гонке.
        os.link(final, output)
