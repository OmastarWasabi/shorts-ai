"""Команды генерации, выравнивания речи и монтажа."""

import argparse
import asyncio
from pathlib import Path
import sys

from . import __version__
from .config import load_env, read_config
from .files import project_directory, project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shorts-ai", description="Генерация материалов и монтаж коротких видео."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--project-dir", type=Path, default=Path("generator"),
        help="каталог данных и .env (по умолчанию ./generator); указывается перед командой",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("generate", help="генерации, включённые в .env")
    voice = commands.add_parser("voice", help="только озвучка, независимо от флагов .env")
    voice.add_argument("--script", type=Path, default=Path("story.txt"))
    voice.add_argument("--output", type=Path, default=Path("voice.mp3"))
    images = commands.add_parser("images", help="только изображения, независимо от флагов .env")
    images.add_argument("--prompts", type=Path, default=Path("prompts.txt"))
    images.add_argument("--output-dir", type=Path, default=Path("images"))
    align = commands.add_parser("align", help="тайминги сцен и слов")
    for name, default in (("scenes", "scenes.json"), ("script", "story.txt"), ("audio", "voice.mp3"), ("output", "timeline.json")):
        align.add_argument(f"--{name}", type=Path, default=Path(default))
    align.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    align.add_argument("--force", action="store_true")
    align.add_argument("--download-model", action="store_true")
    align.add_argument("--prompts", type=Path, default=Path("prompts.txt"),
                       help="промпты для авторазметки, если файл --scenes отсутствует; один абзац на промпт")
    montage = commands.add_parser("montage", help="монтаж готовых материалов")
    for name, default in (("images", "images"), ("audio", "voice.mp3"), ("output", "final.mp4")):
        montage.add_argument(f"--{name}", type=Path, default=Path(default))
    montage.add_argument("--timeline", type=Path)
    montage.add_argument("--no-subtitles", action="store_false", dest="subtitles", default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        directory = project_directory(args.project_dir)
        if args.command in ("generate", "voice", "images"):
            load_env(directory / ".env")
            if args.command == "generate":
                cfg = read_config()
                if not cfg.generate_voice and not cfg.generate_images:
                    print("Все генерации отключены; запросы не выполнялись.")
                    return 0
                from .generation import run_generation

                asyncio.run(run_generation(directory, cfg))
                return 0
            if args.command == "voice":
                return run_voice(directory, args.script, args.output)
            if args.command == "images":
                from .generation import generate_images

                asyncio.run(generate_images(project_path(directory, args.prompts), project_path(directory, args.output_dir)))
                return 0
        if args.command == "montage":
            from .montage import render

            output = project_path(directory, args.output)
            render(project_path(directory, args.images), project_path(directory, args.audio), output,
                   timeline=project_path(directory, args.timeline) if args.timeline is not None else None,
                   subtitles=args.subtitles)
            print(f"Видео сохранено: {output}")
            return 0
        if args.command == "align":
            from .alignment import run_alignment

            run_alignment(
                project_path(directory, args.scenes), project_path(directory, args.script),
                project_path(directory, args.audio), project_path(directory, args.output),
                device=args.device, force=args.force, download_model=args.download_model,
                prompts_path=project_path(directory, args.prompts),
            )
            return 0
    except KeyboardInterrupt:
        print("\nВыполнение прервано.", file=sys.stderr)
        return 130
    except (OSError, ValueError, NotImplementedError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


def run_voice(directory: Path, script: Path, output: Path) -> int:
    # Справка и полностью отключённая генерация обходятся без HTTP/ML-импортов.
    from .generation import generate_voice

    asyncio.run(generate_voice(project_path(directory, script), project_path(directory, output)))
    return 0
