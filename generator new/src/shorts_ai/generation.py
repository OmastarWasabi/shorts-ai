"""Управление генерацией: проверки, клиент провайдера и сохранение результата."""

import asyncio
from pathlib import Path

from . import fish, images
from .config import GenerationConfig
from .prompts import read_prompts
from .transport import REQUEST_TIMEOUT, create_client


async def generate_voice(
    script_path: Path, output_path: Path, *, timeout: float = REQUEST_TIMEOUT,
) -> bool:
    """Возвращает True для нового MP3 и False для пропущенного готового файла."""
    text = fish.read_text(script_path)
    if fish.audio_exists(output_path):
        print(
            f"{output_path.name} уже существует — генерация пропущена. "
            "Для нового текста переместите или удалите файл."
        )
        return False
    config = fish.read_config()
    print("Fish Audio: генерируем озвучку...", flush=True)
    async with create_client() as client:
        audio = await fish.request_audio(client, text, config, timeout=timeout)
    # Даём ожидающей отмене сработать до публикации готового файла.
    await asyncio.sleep(0)
    fish.save_audio(output_path, audio)
    print(f"Готово: {output_path}")
    return True


async def generate_images(
    prompts_path: Path, output_directory: Path, *,
    request_timeout: float = REQUEST_TIMEOUT, total_timeout: float = images.IMAGE_TIMEOUT,
) -> int:
    prompts = read_prompts(prompts_path)
    pending = [prompt for prompt in prompts if not images.image_exists(output_directory / f"{prompt.number:02d}.png")]
    if not pending:
        print("Все изображения уже существуют — генерация пропущена.")
        return 0
    config = images.read_config()
    output_directory.mkdir(parents=True, exist_ok=True)
    print(f"OpenAI Images: платная генерация {len(pending)} изображений, максимум 3 запроса одновременно.", flush=True)
    failures = []
    jobs = iter(pending)
    async with create_client() as client:
        async def worker():
            for prompt in jobs:
                try:
                    data = await images.request_image(
                        client, prompt.text, config, request_timeout=request_timeout, total_timeout=total_timeout,
                    )
                    await asyncio.sleep(0)
                    images.save_image(output_directory / f"{prompt.number:02d}.png", data)
                except (OSError, ValueError) as error:
                    failures.append((prompt.number, str(error)))
                await asyncio.sleep(0)

        async with asyncio.TaskGroup() as group:
            for _ in range(min(3, len(pending))):
                group.create_task(worker())
    if failures:
        raise ValueError("\n".join(f"изображение {number:02d}: {message}" for number, message in sorted(failures)))
    print(f"Готово: создано {len(pending)} изображений в {output_directory}")
    return len(pending)


async def run_generation(directory: Path, config: GenerationConfig) -> None:
    if not config.generate_voice and not config.generate_images:
        print("Все генерации отключены; запросы не выполнялись.")
        return
    preflight_errors = []
    for enabled, reader, path in (
        (config.generate_voice, fish.read_text, directory / "story.txt"),
        (config.generate_images, read_prompts, directory / "prompts.txt"),
    ):
        if enabled:
            try:
                reader(path)
            except (OSError, ValueError) as error:
                preflight_errors.append(str(error))
    if preflight_errors:
        raise ValueError("\n".join(preflight_errors))
    failures = []

    async def run(name, operation, *args):
        try:
            await operation(*args)
        except (OSError, ValueError) as error:
            failures.append(f"{name}: {error}")

    async with asyncio.TaskGroup() as group:
        if config.generate_voice:
            group.create_task(run("Fish Audio", generate_voice, directory / "story.txt", directory / "voice.mp3"))
        if config.generate_images:
            group.create_task(run("OpenAI Images", generate_images, directory / "prompts.txt", directory / "images"))
    if failures:
        raise ValueError("\n".join(failures))
