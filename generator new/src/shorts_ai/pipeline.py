"""Последовательный запуск генерации, выравнивания речи и монтажа."""

import asyncio
from datetime import datetime
from pathlib import Path
from time import perf_counter

from .alignment import run_alignment
from .config import GenerationConfig
from .files import create_run
from .generation import run_generation
from .montage import render

OUTPUT_ROOT = Path.home() / "Documents" / "shorts-ai" / "shorts" / "ai"


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


def run_pipeline(directory: Path, config: GenerationConfig, *, music: Path | None = None) -> Path:
    started = perf_counter()
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run = create_run(directory, OUTPUT_ROOT / name)
    print(f"Папка ролика: {run}", flush=True)

    # run_generation возвращается после завершения голоса и всех изображений.
    print("Генерация начата.", flush=True)
    stage_started = perf_counter()
    asyncio.run(run_generation(run, config))
    print(f"Генерация завершена: {_duration(perf_counter() - stage_started)}", flush=True)

    audio = run / "voice.mp3"
    timeline = run / "timeline.json"
    print("Выравнивание начато.", flush=True)
    stage_started = perf_counter()
    run_alignment(
        run / "scenes.json", run / "story.txt", audio, timeline,
        prompts_path=run / "prompts.txt",
    )
    print(f"Выравнивание завершено: {_duration(perf_counter() - stage_started)}", flush=True)

    output = run / "final.mp4"
    print("Монтаж начат.", flush=True)
    stage_started = perf_counter()
    render(run / "images", audio, output, timeline=timeline, music=music)
    print(f"Монтаж завершён: {_duration(perf_counter() - stage_started)}", flush=True)
    print(f"Всего: {_duration(perf_counter() - started)}", flush=True)
    return output
