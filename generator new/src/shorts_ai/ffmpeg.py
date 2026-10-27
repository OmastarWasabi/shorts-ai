"""Запуск FFmpeg/ffprobe и параметры базового монтажа (см. NOTICE)."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess

WIDTH = 1080
HEIGHT = 1920
FPS = 30


def find_tools() -> tuple[str, str]:
    paths = []
    for name in (os.environ.get("FFMPEG_BIN") or "ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if path is None:
            raise ValueError(f"{name} не найден; установите FFmpeg и добавьте его в PATH")
        # Mux запускается из временной папки; относительный FFMPEG_BIN тоже работает.
        paths.append(os.path.abspath(path))
    return paths[0], paths[1]


def command(binary: str, *args: str, cwd: Path | None = None, timeout: float | None = None) -> bytes:
    """При Ctrl+C убивает и дожидается дочернего процесса до удаления его файлов."""
    process = subprocess.Popen(
        [binary, *args], cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise ValueError(f"Истекло время выполнения {Path(binary).name}") from None
        except BaseException:
            # Как exec.CommandContext в Go: отмена немедленно останавливает рендер.
            process.kill()
            process.communicate()
            raise
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"{Path(binary).name}: код {process.returncode}\n{detail}")
        return stdout
    finally:
        process.stdout.close()
        process.stderr.close()


def ffmpeg(binary: str, *args: str, cwd: Path | None = None) -> bytes:
    return command(
        binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-filter_threads", "1", *args, cwd=cwd,
    )


def audio_duration(path: Path, *, binary: str = "ffprobe", timeout: float | None = None) -> float:
    data = command(
        binary, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=codec_type:format=duration", "-of", "json", str(path), timeout=timeout,
    )
    try:
        info = json.loads(data)
        streams = info.get("streams", [])
        if not isinstance(streams, list) or any(not isinstance(stream, dict) for stream in streams):
            raise ValueError
        duration_text = info.get("format", {}).get("duration", "")
        if not isinstance(duration_text, str):
            raise ValueError
    except (ValueError, AttributeError, UnicodeError):
        raise ValueError("Некорректный ответ ffprobe") from None
    if not streams or streams[0].get("codec_type") != "audio":
        raise ValueError("В файле озвучки нет аудиодорожки")
    try:
        duration = float(duration_text)
    except ValueError:
        duration = math.nan
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("ffprobe не смог определить положительную длительность озвучки")
    return duration


def scene_filter(frames: int, index: int) -> str:
    progress = f"on/{max(1, frames - 1)}"
    x, y = "iw/2-iw/zoom/2", "ih/2-ih/zoom/2"
    zoom = f"1+0.12*{progress}"
    match index % 6:
        case 1:
            zoom, x = "1.12", f"(iw-iw/zoom)*{progress}"
        case 2:
            zoom = f"1.12-0.12*{progress}"
        case 3:
            zoom, x = "1.12", f"(iw-iw/zoom)*(1-{progress})"
        case 4:
            zoom, y = "1.12", f"(ih-ih/zoom)*(1-{progress})"
        case 5:
            zoom, y = "1.12", f"(ih-ih/zoom)*{progress}"
    return (
        f"scale={WIDTH * 2}:{HEIGHT * 2}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={WIDTH * 2}:{HEIGHT * 2},setsar=1,"
        f"zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p"
    )


def render_scene(binary: str, image: Path, frames: int, index: int, output: Path) -> None:
    ffmpeg(
        binary, "-i", str(image), "-vf", scene_filter(frames, index), "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-threads", "2", "-pix_fmt", "yuv420p", str(output),
    )


def require_ass(binary: str) -> None:
    filters = ffmpeg(binary, "-filters").decode("utf-8", errors="replace")
    if not any(len(fields := line.split()) >= 2 and fields[1] == "ass" for line in filters.splitlines()):
        raise ValueError("Для субтитров нужен FFmpeg с libass; укажите FFMPEG_BIN или используйте --no-subtitles")


def mux(binary: str, listing: Path, audio: Path, duration: float, output: Path, *, subtitles: bool = False) -> None:
    video_args = ["-c:v", "copy"]
    if subtitles:
        # Имя внутри фильтра постоянно; пользовательские пути в ASS-синтаксис не попадают.
        video_args = ["-vf", "ass=filename=captions.ass", "-c:v", "libx264", "-preset", "medium",
                      "-crf", "18", "-threads", "2"]
    ffmpeg(
        binary, "-f", "concat", "-safe", "1", "-i", str(listing), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", *video_args, "-c:a", "aac", "-b:a", "192k",
        "-t", f"{duration:.6f}", "-shortest", "-movflags", "+faststart", str(output),
        cwd=output.parent,
    )
