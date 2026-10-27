"""Align known scene text to a single narration file with WhisperX."""

from bisect import bisect_right
import json
import math
from numbers import Real
import os
from pathlib import Path
import re
import unicodedata

from .ffmpeg import audio_duration as probe_duration, command, find_tools
from .files import read_utf8, write_bytes_atomic
from .timeline import file_hash, validate_timeline


def tokens(text):
    text = unicodedata.normalize("NFD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def alignment_text(text):
    text = unicodedata.normalize("NFD", text)
    # Remove stress marks, keeping Russian letters such as й and ё intact.
    return " ".join(unicodedata.normalize("NFC", text.replace("\u0301", "")).split())


def read_scenes(path, script_path):
    document = json.loads(read_utf8(path))
    if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("scenes.json: ожидается version=1")
    scenes = document.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("scenes.json: нужен непустой список scenes")
    for number, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict) or type(scene.get("number")) is not int or scene["number"] != number:
            raise ValueError("Номера сцен должны идти подряд с 1")
        if not isinstance(scene.get("text"), str) or not tokens(scene["text"]):
            raise ValueError(f"Сцена {number}: нужен непустой text")
    script = read_utf8(script_path)
    if tokens(" ".join(scene["text"] for scene in scenes)) != tokens(script):
        raise ValueError("Текст всех сцен должен совпадать с story.txt по словам и порядку")
    language = document.get("language", "ru")
    if not isinstance(language, str) or not re.fullmatch(r"[a-z]{2,3}", language):
        raise ValueError("Неверный код языка в scenes.json")
    return scenes, language


def audio_duration(path):
    duration = probe_duration(path.absolute(), timeout=30)
    if duration > 180:
        raise ValueError("Этот режим выравнивания рассчитан на озвучку не длиннее 180 секунд")
    return duration


def load_audio(path):
    # Тот же PCM mono/16 kHz и нормализация, что в WhisperX load_audio;
    # общий запуск процесса гарантирует завершение FFmpeg при Ctrl+C.
    import numpy as np

    encoder, _ = find_tools()
    data = command(encoder, "-nostdin", "-threads", "0", "-i", str(path.resolve()),
                   "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000", "-")
    return np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0


def build_timeline(scenes, aligned_words, duration, audio_hash):
    if not isinstance(aligned_words, list):
        raise ValueError("WhisperX вернул неверный список слов")
    expected, offsets = [], [0]
    for scene in scenes:
        expected.extend(tokens(scene["text"]))
        offsets.append(len(expected))
    words, cursor, previous_start = [], 0, -1.0
    for word in aligned_words:
        if not isinstance(word, dict) or not isinstance(word.get("word", ""), str):
            raise ValueError("WhisperX вернул неверный текст слова")
        text = word.get("word", "").strip()
        normalized = tokens(text)
        if not normalized:
            continue
        if expected[cursor:cursor + len(normalized)] != normalized:
            raise ValueError(f"Не удалось сопоставить слово «{text}» с текстом сцен")
        scene_index = bisect_right(offsets, cursor) - 1
        if cursor + len(normalized) > offsets[scene_index + 1]:
            raise ValueError("Граница сцены оказалась внутри слова")
        start, end = word.get("start"), word.get("end")
        if (not isinstance(start, Real) or not isinstance(end, Real) or isinstance(start, bool) or isinstance(end, bool)
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or end > duration + 0.02 or start < previous_start):
            raise ValueError(f"WhisperX не определил корректное время слова «{text}»")
        words.append({"text": text, "start": float(start), "end": float(end), "scene": scene_index + 1})
        previous_start = start
        cursor += len(normalized)
    if cursor != len(expected):
        raise ValueError("WhisperX не смог выровнять весь сценарий; проверьте соответствие текста и озвучки")
    first_words = {}
    for word in words:
        first_words.setdefault(word["scene"], word["start"])
    if len(first_words) != len(scenes):
        raise ValueError("У одной из сцен отсутствуют тайминги")
    cuts = [0.0] + [first_words[number] for number in range(2, len(scenes) + 1)] + [duration]
    timed_scenes = []
    for i, scene in enumerate(scenes):
        if cuts[i + 1] - cuts[i] < 1 / 30:
            raise ValueError(f"Сцена {i + 1} короче одного видеокадра; проверьте разметку")
        timed_scenes.append({"number": scene["number"], "text": scene["text"], "start": cuts[i], "end": cuts[i + 1]})
    for word in words:
        scene = timed_scenes[word["scene"] - 1]
        if word["start"] < scene["start"] - 0.03 or word["end"] > scene["end"] + 0.03:
            raise ValueError(f"Слово «{word['text']}» выходит за границы сцены {word['scene']}; проверьте выравнивание")
    return {"version": 1, "audio_sha256": audio_hash, "duration": duration, "scenes": timed_scenes, "words": words}


def align(scenes, language, audio_path, duration, device, download_model=False):
    # Transformers иначе запускает фоновую конвертацию чужой модели на Hub,
    # которая не нужна для выравнивания и может удерживать процесс при выходе.
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
    try:
        import torch
        import torchaudio
        from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH
        from whisperx.alignment import align as align_audio, load_align_model
    except ImportError as error:
        raise ValueError("Установите зависимости: python -m pip install -r requirements-timing.lock.txt") from error
    if device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    print(f"WhisperX: загружаю модель выравнивания для {language} ({device})...", flush=True)
    # WhisperX 3.8.6 учитывает model_cache_only только для Hugging Face.
    # Для встроенных моделей torchaudio проверяем кэш до вызова get_model.
    if not download_model and language in DEFAULT_ALIGN_MODELS_TORCH:
        bundle = getattr(torchaudio.pipelines, DEFAULT_ALIGN_MODELS_TORCH[language])
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(bundle._path).name
        if not checkpoint.is_file():
            raise ValueError("Модель отсутствует в кэше. Для первого запуска добавьте --download-model")
    try:
        model, metadata = load_align_model(language_code=language, device=device, model_cache_only=not download_model)
    except ValueError as error:
        if not download_model:
            raise ValueError("Не удалось загрузить модель из кэша. Для первого запуска добавьте --download-model") from error
        raise
    audio = load_audio(audio_path)
    text = " ".join(alignment_text(scene["text"]) for scene in scenes)
    print("WhisperX: определяю тайминги слов...", flush=True)
    result = align_audio([
        {"start": 0.0, "end": min(duration, len(audio) / 16000), "text": text},
    ], model, metadata, audio, device, return_char_alignments=False)
    words = result.get("word_segments", []) if isinstance(result, dict) else None
    if not isinstance(words, list) or any(not isinstance(word, dict) or not isinstance(word.get("word", ""), str) for word in words):
        raise ValueError("WhisperX вернул неверный список слов")
    scores = []
    for word in words:
        if "score" in word and tokens(word.get("word", "")):
            score = word["score"]
            if not isinstance(score, Real) or isinstance(score, bool):
                raise ValueError("WhisperX вернул неверную уверенность слова")
            scores.append(float(score))
    if not scores or any(not math.isfinite(score) for score in scores) or sum(scores) / len(scores) < 0.2:
        raise ValueError("Низкая уверенность выравнивания; проверьте, что story.txt соответствует voice.mp3")
    return words


def write_timeline(path, timeline, force=False):
    content = (json.dumps(timeline, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    write_bytes_atomic(path, content, overwrite=force)


def run_alignment(scenes_path: Path, script: Path, audio: Path, output: Path, *,
                  device: str = "cpu", force: bool = False, download_model: bool = False,
                  prompts_path: Path | None = None) -> dict:
    if device not in ("cpu", "cuda"):
        raise ValueError("Устройство должно быть cpu или cuda")
    inputs = [audio, script, scenes_path]
    if prompts_path is not None:
        inputs.append(prompts_path)
    for source in inputs:
        if source.resolve() == output.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("Выходной JSON не должен заменять входной файл")
    if os.path.lexists(output) and not force:
        raise ValueError(f"{output} уже существует; для обновления используйте --force")
    if output.exists() and output.is_dir():
        raise ValueError("Выходной JSON должен быть файлом, а не каталогом")
    if not os.path.lexists(scenes_path) and prompts_path is not None:
        if not script.is_file() or not prompts_path.is_file():
            raise ValueError(
                f"Не найден {scenes_path.name}; для авторазметки нужны "
                f"{script.name} и {prompts_path.name}"
            )
        from .scenes import create_scenes

        document = create_scenes(script, prompts_path, scenes_path)
        print(f"Авторазметка: создан {scenes_path}; сцен: {len(document['scenes'])}", flush=True)
    scenes, language = read_scenes(scenes_path, script)
    audio_hash = file_hash(audio)
    duration = audio_duration(audio)
    try:
        words = align(scenes, language, audio, duration, device, download_model)
    except (OSError, ValueError):
        raise
    except Exception as error:
        raise ValueError(f"Ошибка WhisperX: {error}") from error
    timeline = build_timeline(scenes, words, duration, audio_hash)
    validate_timeline(timeline, duration, len(scenes))
    if file_hash(audio) != audio_hash:
        raise ValueError("Озвучка изменилась во время выравнивания; повторите запуск")
    write_timeline(output, timeline, force=force)
    print(f"Готово: {output}; сцен: {len(scenes)}, слов: {len(timeline['words'])}")
    return timeline
