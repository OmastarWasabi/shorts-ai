"""Приёмочная цепочка: тестовый API → файлы → JSON → настоящий FFmpeg."""

import base64
import asyncio
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from shorts_ai.cli import main
from shorts_ai.config import GenerationConfig
from shorts_ai.files import current_run
from shorts_ai.ffmpeg import command, ffmpeg, find_tools, require_ass
from shorts_ai.timeline import read_timeline
from shorts_ai.transport import create_client
from shorts_ai.pipeline import run_pipeline

FIXTURES = Path(__file__).parent / "fixtures/migration"


class PipelineTests(unittest.TestCase):
    def test_mock_generation_resume_alignment_and_both_video_modes(self):
        try:
            encoder, probe = find_tools()
            require_ass(encoder)
        except ValueError as error:
            self.skipTest(str(error))
        with tempfile.TemporaryDirectory(prefix="цепочка 'Python' ") as temporary:
            root = Path(temporary).resolve()
            scenes = json.loads((FIXTURES / "scenes.json").read_text())["scenes"]
            (root / "story.txt").write_text("\n\n".join(scene["text"] for scene in scenes), encoding="utf-8")
            (root / "prompts.txt").write_text("[01]\nred\n[02]\nblue", encoding="utf-8")
            source_audio = root / "source.mp3"
            ffmpeg(encoder, "-f", "lavfi", "-i", "sine=frequency=440:duration=4", str(source_audio))
            audio = source_audio.read_bytes()
            source_audio.unlink()
            music = root / "музыка 'фон'.mp3"
            ffmpeg(encoder, "-f", "lavfi", "-i", "sine=frequency=1000:duration=1", str(music))
            pngs = {}
            for color in ("red", "blue"):
                buffer = io.BytesIO()
                with Image.new("RGB", (160, 120), color) as image:
                    image.save(buffer, format="PNG")
                pngs[color] = buffer.getvalue()
            calls = []

            def handle(request):
                calls.append(request.url.host)
                if request.url.host == "api.fish.audio":
                    return httpx.Response(200, content=audio, headers={"Content-Type": "audio/mpeg"})
                self.assertEqual(request.url.host, "api.openai.com")
                prompt = json.loads(request.content)["prompt"]
                return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(pngs[prompt]).decode()}]})

            def client():
                return create_client(transport=httpx.MockTransport(handle))

            def run(*arguments):
                code = main(["--project-dir", str(root), *arguments])
                self.assertEqual(code, 0, (arguments, errors.getvalue()))

            env = {"FISH_API_KEY": "test-fish", "OPENAI_API_KEY": "test-openai",
                   "GENERATE_VOICE": "true", "GENERATE_IMAGES": "true"}
            words = json.loads((FIXTURES / "aligned_words.json").read_text())
            with patch.dict(os.environ, env), patch("shorts_ai.generation.create_client", side_effect=client), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as errors:
                with patch("shorts_ai.pipeline.OUTPUT_ROOT", root / "automatic"), \
                     patch("shorts_ai.alignment.align", return_value=words):
                    run("run", "--music", str(music))
                output = current_run(root)
                self.assertTrue((output / "final.mp4").is_file())
                self.assertTrue((output / "scenes.json").is_file())
                self.assertFalse((root / "scenes.json").exists())
                self.assertEqual(calls.count("api.fish.audio"), 1)
                self.assertEqual(calls.count("api.openai.com"), 2)
                with patch.dict(os.environ, {"FISH_API_KEY": "", "OPENAI_API_KEY": ""}):
                    run("generate", "--output-dir", str(output))
                    run("voice")
                    run("images")
                self.assertEqual(len(calls), 3)
                (output / "images/02.png").unlink()
                run("images")
                self.assertEqual(calls.count("api.openai.com"), 3)
                # Модель подставлена только в этом тесте; реальное выравнивание проверяется отдельно.
                with patch("shorts_ai.alignment.align", return_value=words):
                    run("align", "--force")
                duration = json.loads((output / "timeline.json").read_text())["duration"]
                _, frames = read_timeline(output / "timeline.json", output / "voice.mp3", duration, 2)
                self.assertEqual(frames[0], 66)
                run("montage", "--timeline", "timeline.json", "--music", str(music), "--output", "captioned.mp4")
                run("montage", "--timeline", "timeline.json", "--no-subtitles", "--output", "plain.mp4")
            for name in ("final.mp4", "captioned.mp4", "plain.mp4"):
                info = json.loads(command(probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output / name)))
                video, sound = info["streams"]
                self.assertEqual((video["codec_name"], video["width"], video["height"], video["r_frame_rate"]),
                                 ("h264", 1080, 1920, "30/1"))
                self.assertEqual(sound["codec_name"], "aac")
                self.assertAlmostEqual(float(info["format"]["duration"]), duration, delta=.1)
            self.assertFalse(list(root.rglob(".montage-*")))
            self.assertFalse(list(root.rglob(".shorts-ai-*")))


class PipelineControllerTests(unittest.TestCase):
    def test_each_automatic_run_has_its_own_folder_and_updates_current_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "inputs"
            source.mkdir()
            (source / "story.txt").write_text("Сценарий.")
            (source / "prompts.txt").write_text("[01]\nScene.")
            with patch("shorts_ai.pipeline.OUTPUT_ROOT", root / "videos"), \
                 patch("shorts_ai.pipeline.datetime") as clock, \
                 patch("shorts_ai.pipeline.run_generation"), \
                 patch("shorts_ai.pipeline.run_alignment"), \
                 patch("shorts_ai.pipeline.render"), redirect_stdout(io.StringIO()):
                clock.now.side_effect = [datetime(2026, 10, 28, 12, 0, 0, 1), datetime(2026, 10, 28, 12, 0, 0, 2)]
                first = run_pipeline(source, GenerationConfig())
                second = run_pipeline(source, GenerationConfig())
            self.assertNotEqual(first.parent, second.parent)
            for output in (first, second):
                self.assertEqual(output.parent.parent, root / "videos")
                self.assertEqual((output.parent / "story.txt").read_text(), "Сценарий.")
            self.assertEqual(current_run(source), second.parent)

    def test_interrupt_stops_pipeline_and_cli_returns_130(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch("shorts_ai.pipeline.OUTPUT_ROOT", root / "videos"), \
                 patch("shorts_ai.pipeline.run_generation"), \
                 patch("shorts_ai.pipeline.run_alignment", side_effect=KeyboardInterrupt), \
                 patch("shorts_ai.pipeline.render") as montage, \
                 patch.dict(os.environ, {"GENERATE_VOICE": "true", "GENERATE_IMAGES": "true"}), \
                 redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(main(["--project-dir", str(root), "run"]), 130)
            montage.assert_not_called()
            self.assertIn("прервано", stderr.getvalue())
            self.assertNotIn("Всего:", stdout.getvalue())
            self.assertNotIn("Готовый ролик:", stdout.getvalue())

    def test_waits_for_both_generators_then_aligns_and_renders_in_same_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "inputs"
            source.mkdir()
            (source / "story.txt").write_text("Исходный сценарий.")
            (source / "prompts.txt").write_text("[01]\nScene.")
            steps = []

            async def generate(directory, config):
                async def voice():
                    await asyncio.sleep(0)
                    (directory / "voice.mp3").write_bytes(b"audio")
                    steps.append("voice")

                async def images():
                    await asyncio.sleep(0)
                    await asyncio.sleep(0)
                    (directory / "images").mkdir()
                    (directory / "images/01.png").write_bytes(b"image")
                    steps.append("images")

                await asyncio.gather(voice(), images())
                (source / "story.txt").write_text("Следующий сценарий.")

            def align(scenes, script, audio, output, *, prompts_path):
                self.assertEqual(steps, ["voice", "images"])
                self.assertEqual(script.read_text(), "Исходный сценарий.")
                self.assertEqual(audio.read_bytes(), b"audio")
                self.assertEqual(prompts_path.parent, output.parent)
                output.write_text("{}")
                steps.append("align")

            def montage(images, audio, output, *, timeline, music):
                self.assertEqual(steps, ["voice", "images", "align"])
                self.assertTrue((images / "01.png").is_file())
                self.assertEqual(images.parent, output.parent)
                self.assertEqual(audio.parent, output.parent)
                self.assertEqual(timeline.parent, output.parent)
                self.assertTrue(timeline.is_file())
                self.assertEqual(music, source / "music.mp3")
                output.write_bytes(b"video")
                steps.append("montage")

            with patch("shorts_ai.pipeline.OUTPUT_ROOT", root / "videos"), \
                 patch("shorts_ai.pipeline.run_generation", side_effect=generate), \
                 patch("shorts_ai.pipeline.run_alignment", side_effect=align), \
                 patch("shorts_ai.pipeline.render", side_effect=montage), \
                 patch("shorts_ai.pipeline.perf_counter", side_effect=[0, 2, 137, 138, 156, 157, 181, 182]), \
                 redirect_stdout(io.StringIO()) as stdout:
                output = run_pipeline(source, GenerationConfig(), music=source / "music.mp3")
            self.assertEqual(steps, ["voice", "images", "align", "montage"])
            self.assertEqual(current_run(source), output.parent)
            self.assertEqual(output.parent.parent, root / "videos")
            self.assertEqual(output.name, "final.mp4")
            self.assertEqual(output.read_bytes(), b"video")
            self.assertEqual(stdout.getvalue().splitlines()[1:], [
                "Генерация начата.", "Генерация завершена: 02:15",
                "Выравнивание начато.", "Выравнивание завершено: 00:18",
                "Монтаж начат.", "Монтаж завершён: 00:24", "Всего: 03:02",
            ])

    def test_stage_error_stops_the_following_stages(self):
        for stage in ("generation", "alignment", "montage"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                with patch("shorts_ai.pipeline.OUTPUT_ROOT", root / "videos"), \
                     patch("shorts_ai.pipeline.run_generation") as generate, \
                     patch("shorts_ai.pipeline.run_alignment") as align, \
                     patch("shorts_ai.pipeline.render") as montage, redirect_stdout(io.StringIO()) as stdout:
                    failing = {"generation": generate, "alignment": align, "montage": montage}[stage]
                    failing.side_effect = ValueError("stage failed")
                    with self.assertRaisesRegex(ValueError, "stage failed"):
                        run_pipeline(root, GenerationConfig())
                    if stage != "montage":
                        montage.assert_not_called()
                    if stage == "generation":
                        align.assert_not_called()
                    completed = {"generation": "Генерация завершена", "alignment": "Выравнивание завершено",
                                 "montage": "Монтаж завершён"}[stage]
                    self.assertNotIn(completed, stdout.getvalue())
                    self.assertNotIn("Всего:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
