from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import signal
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from shorts_ai.cli import main
from shorts_ai.ffmpeg import audio_duration, command, ffmpeg, find_tools, render_scene, scene_filter
from shorts_ai.montage import read_images, render, split_frames

FIXTURES = Path(__file__).parent / "fixtures" / "migration"


class MontageTests(unittest.TestCase):
    def test_frame_distributions_and_all_camera_presets_match_go(self):
        baseline = json.loads((FIXTURES / "montage.json").read_text())
        for case in baseline["splits"]:
            with self.subTest(case=case):
                self.assertEqual(split_frames(case["duration"], case["scenes"]), case["frames"])
        for case in baseline["filters"]:
            with self.subTest(frames=case["frames"], index=case["index"]):
                self.assertEqual(scene_filter(case["frames"], case["index"]), case["filter"])
        for duration, scenes in ((0, 1), (-1, 1), (math.nan, 1), (math.inf, 1), (1e308, 1), (1, 0), (1, -1), (.01, 2)):
            with self.subTest(duration=duration, scenes=scenes), self.assertRaises(ValueError):
                split_frames(duration, scenes)

    def test_numeric_order_gaps_duplicates_and_ignored_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Нет изображений"):
                read_images(directory)
            names = [f"{i}.png" for i in range(1, 13)]
            names[1] = "02.PNG"
            for name in reversed(names):
                (directory / name).touch()
            for name in ("preview.png", "1.jpg", "１２.png", "-3.png", "4.0.png"):
                (directory / name).touch()
            self.assertEqual(read_images(directory), [directory / name for name in names])
            (directory / "02.PNG").unlink()
            with self.assertRaisesRegex(ValueError, "Пропущено изображение 02"):
                read_images(directory)
            (directory / "01.png").touch()
            with self.assertRaisesRegex(ValueError, "Одинаковый номер"):
                read_images(directory)

    def test_invalid_numbers_directories_and_symlinks(self):
        for name in ("0.png", f"{sys.maxsize + 1}.png"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / name).touch()
                with self.assertRaisesRegex(ValueError, "Неверный номер"):
                    read_images(directory)
        for kind in ("directory", "symlink", "dangling"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                image = directory / "01.png"
                if kind == "directory":
                    image.mkdir()
                else:
                    target = directory / "target"
                    if kind == "symlink":
                        target.touch()
                    image.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "обычным PNG"):
                    read_images(directory)

    def test_probe_validation(self):
        valid = {"streams": [{"codec_type": "audio"}], "format": {"duration": "1.2"}}
        with patch("shorts_ai.ffmpeg.command", return_value=json.dumps(valid).encode()):
            self.assertEqual(audio_duration(Path("voice.wav")), 1.2)
        for value in ("0", "-1", "NaN", "Infinity", "bad", "", None, 1.2):
            with self.subTest(value=value):
                data = {**valid, "format": {"duration": value}}
                with patch("shorts_ai.ffmpeg.command", return_value=json.dumps(data).encode()), self.assertRaises(ValueError):
                    audio_duration(Path("voice.wav"))
        for data in (b"broken", b"[]", b"null", b'{"streams":[null]}', b'{"streams":[],"format":null}', b"{}",
                     b'{"streams":[{"codec_type":"video"}],"format":{"duration":"1"}}'):
            with self.subTest(data=data), patch("shorts_ai.ffmpeg.command", return_value=data), self.assertRaises(ValueError):
                audio_duration(Path("voice.wav"))

    def test_output_preflight_including_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as temporary, patch("shorts_ai.montage.find_tools") as tools:
            directory = Path(temporary)
            for name, kind in (("bad.avi", "extension"), ("old.mp4", "file"), ("dir.mp4", "directory"), ("link.mp4", "symlink")):
                output = directory / name
                if kind == "file":
                    output.write_bytes(b"existing video")
                elif kind == "directory":
                    output.mkdir()
                elif kind == "symlink":
                    output.symlink_to(directory / "missing.mp4")
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    render(directory, directory / "missing.wav", output)
            tools.assert_not_called()
            self.assertEqual((directory / "old.mp4").read_bytes(), b"existing video")
            self.assertTrue((directory / "link.mp4").is_symlink())

    def test_missing_tools_and_relative_encoder_path(self):
        with patch.dict(os.environ, {"FFMPEG_BIN": "./tools/ffmpeg"}), patch("shorts_ai.ffmpeg.shutil.which", side_effect=lambda name: name):
            encoder, probe = find_tools()
            self.assertEqual(encoder, str(Path("tools/ffmpeg").absolute()))
            self.assertTrue(Path(probe).is_absolute())
        with patch("shorts_ai.ffmpeg.shutil.which", return_value=None), self.assertRaisesRegex(ValueError, "не найден"):
            find_tools()

    def test_publication_race_preserves_competing_result_and_cleans_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "result.mp4"
            image = directory / "01.png"
            def competing_mux(binary, listing, audio, duration, final, **kwargs):
                final.write_bytes(b"our video")
                output.write_bytes(b"other video")
            with patch("shorts_ai.montage.find_tools", return_value=("ffmpeg", "ffprobe")), \
                 patch("shorts_ai.montage.read_images", return_value=[image]), \
                 patch("shorts_ai.montage.audio_duration", return_value=1), \
                 patch("shorts_ai.montage.render_scene"), \
                 patch("shorts_ai.montage.mux", side_effect=competing_mux), redirect_stderr(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    render(directory, directory / "audio.wav", output)
            self.assertEqual(output.read_bytes(), b"other video")
            self.assertEqual(list(directory.glob(".montage-*")), [])

    def test_cli_paths_env_independence_and_timeline_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            (directory / ".env").write_text('BROKEN="must-not-read\n')
            for extra, image, audio, output in (([], directory / "images", directory / "voice.mp3", directory / "final.mp4"),
                (["--images", "кадры", "--audio", str(directory / "absolute.wav"), "--output", "nested/out.MP4", "--no-subtitles"],
                 directory / "кадры", directory / "absolute.wav", directory / "nested/out.MP4")):
                with patch("shorts_ai.montage.render") as renderer, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--project-dir", str(directory), "montage", *extra]), 0)
                    renderer.assert_called_once_with(image, audio, output, timeline=None, subtitles="--no-subtitles" not in extra)
            stderr = io.StringIO()
            with patch("shorts_ai.montage.render") as renderer, redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(directory), "montage", "--timeline", "timeline.json"]), 0)
                renderer.assert_called_once_with(directory / "images", directory / "voice.mp3", directory / "final.mp4",
                                                 timeline=directory / "timeline.json", subtitles=True)


class FFmpegIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.encoder, cls.probe = find_tools()
        except ValueError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_real_cli_video_audio_order_corrupt_input_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "видео 'test' с пробелами"
            images = directory / "images"
            images.mkdir(parents=True)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            for index in reversed(range(6)):
                Image.new("RGB", (160, 120), colors[index]).save(images / f"{index + 1:02d}.png")
            audio = directory / "voice.wav"
            ffmpeg(self.encoder, "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2", str(audio))
            # CLI не должен читать .env или импортировать HTTP/ML/Pillow для монтажа.
            (directory / ".env").write_text('BROKEN="do-not-read\n')
            script = """
import builtins, sys
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in {'httpx', 'PIL', 'torch', 'whisperx'}:
        raise AssertionError('unexpected import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from shorts_ai.cli import main
sys.exit(main(sys.argv[1:]))
"""
            args = [sys.executable, "-c", script, "--project-dir", str(directory), "montage", "--audio", "voice.wav", "--output", "готово/result.mp4"]
            result = subprocess.run(args, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = directory / "готово/result.mp4"
            info = json.loads(command(self.probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)))
            self.assertEqual(len(info["streams"]), 2)
            video, sound = info["streams"]
            self.assertEqual([video[key] for key in ("codec_type", "codec_name", "width", "height", "r_frame_rate", "nb_frames", "pix_fmt")],
                             ["video", "h264", 1080, 1920, "30/1", "36", "yuv420p"])
            self.assertEqual((sound["codec_type"], sound["codec_name"]), ("audio", "aac"))
            self.assertAlmostEqual(float(info["format"]["duration"]), 1.2, delta=.1)
            pixels = ffmpeg(self.encoder, "-i", str(output), "-vf", "select='not(mod(n,6))',crop=2:2:540:960",
                            "-frames:v", "6", "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
            self.assertEqual(len(pixels), 72)
            for index, color in enumerate(colors):
                for actual, expected in zip(pixels[index * 12:index * 12 + 3], color):
                    self.assertAlmostEqual(actual, expected, delta=12)
            pcm = ffmpeg(self.encoder, "-i", str(output), "-map", "0:a:0", "-ac", "1", "-ar", "8000", "-f", "s16le", "pipe:1")
            samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
            self.assertGreater(len(samples), 9000)
            self.assertGreater(math.sqrt(sum(sample * sample for sample in samples) / len(samples)), 1000)
            crossings = sum(a <= 0 < b for a, b in zip(samples, samples[1:]))
            self.assertAlmostEqual(crossings / (len(samples) / 8000), 440, delta=15)
            before = output.read_bytes()
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn("уже существует", result.stderr)
            self.assertEqual(output.read_bytes(), before)
            (images / "01.png").write_bytes(b"broken PNG")
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "Кадр 1"):
                render(images, audio, directory / "failed.mp4")
            self.assertFalse((directory / "failed.mp4").exists())
            self.assertEqual(list(directory.rglob(".montage-*")), [])

    def test_camera_actually_moves_for_each_preset(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image = Image.new("RGB", (180, 320), "white")
            draw = ImageDraw.Draw(image)
            for x in range(0, 180, 20):
                for y in range(0, 320, 20):
                    draw.rectangle((x, y, x + 10, y + 10), fill=(x, y % 256, (x + y) % 256))
            path = directory / "pattern.png"
            image.save(path)
            for index in range(6):
                with self.subTest(index=index):
                    output = directory / f"{index}.mp4"
                    render_scene(self.encoder, path, 6, index, output)
                    frames = ffmpeg(self.encoder, "-i", str(output), "-vf", "select='eq(n,0)+eq(n,5)',scale=54:96",
                                    "-frames:v", "2", "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
                    size = 54 * 96 * 3
                    self.assertEqual(len(frames), size * 2)
                    difference = sum(abs(a - b) for a, b in zip(frames[:size], frames[size:])) / size
                    self.assertGreater(difference, 5)

    def test_sigint_reaps_real_ffmpeg_and_removes_partial_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            images = directory / "images"
            images.mkdir()
            Image.new("RGB", (160, 120), "red").save(images / "01.png")
            audio = directory / "voice.wav"
            ffmpeg(self.encoder, "-f", "lavfi", "-i", "sine=duration=60", str(audio))
            # В обёртке лишь записываем PID и заменяем процесс настоящим FFmpeg.
            wrapper = directory / "encoder"
            pid_file = directory / "encoder.pid"
            script = f"import os, sys\nfrom pathlib import Path\nPath({str(pid_file)!r}).write_text(str(os.getpid()))\nos.execv({self.encoder!r}, [{self.encoder!r}, *sys.argv[1:]])\n"
            # Путь к Python теперь может содержать пробелы; обычный shebang его обрежет.
            wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(script) + ' "$@"\n')
            wrapper.chmod(0o700)
            env = {**os.environ, "FFMPEG_BIN": str(wrapper)}
            process = subprocess.Popen([sys.executable, "-m", "shorts_ai", "--project-dir", str(directory), "montage", "--audio", "voice.wav"],
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    clips = list(directory.glob(".montage-*/clip_0001.mp4"))
                    if clips and pid_file.exists():
                        child_pid = int(pid_file.read_text())
                        break
                    if process.poll() is not None:
                        self.fail(f"Montage exited before interrupt: {process.communicate()}")
                    time.sleep(.02)
                self.assertIsNotNone(child_pid, "FFmpeg did not begin rendering")
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 130, (stdout, stderr))
                self.assertIn("прервано", stderr)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertFalse((directory / "final.mp4").exists())
                self.assertEqual(list(directory.glob(".montage-*")), [])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.stdout.close()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
