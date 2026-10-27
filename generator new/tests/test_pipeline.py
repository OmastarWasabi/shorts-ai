"""Приёмочная цепочка: тестовый API → файлы → JSON → настоящий FFmpeg."""

import base64
from contextlib import redirect_stderr, redirect_stdout
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
from shorts_ai.ffmpeg import command, ffmpeg, find_tools, require_ass
from shorts_ai.timeline import read_timeline
from shorts_ai.transport import create_client

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
            for name in ("story.txt", "scenes.json"):
                (root / name).write_bytes((FIXTURES / name).read_bytes())
            (root / "prompts.txt").write_text("[01]\nred\n[02]\nblue", encoding="utf-8")
            source_audio = root / "source.mp3"
            ffmpeg(encoder, "-f", "lavfi", "-i", "sine=frequency=440:duration=4", str(source_audio))
            audio = source_audio.read_bytes()
            source_audio.unlink()
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
                self.assertEqual(code, 0, arguments)

            env = {"FISH_API_KEY": "test-fish", "OPENAI_API_KEY": "test-openai",
                   "GENERATE_VOICE": "true", "GENERATE_IMAGES": "true"}
            words = json.loads((FIXTURES / "aligned_words.json").read_text())
            with patch.dict(os.environ, env), patch("shorts_ai.generation.create_client", side_effect=client), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run("generate")
                self.assertEqual(calls.count("api.fish.audio"), 1)
                self.assertEqual(calls.count("api.openai.com"), 2)
                with patch.dict(os.environ, {"FISH_API_KEY": "", "OPENAI_API_KEY": ""}):
                    run("generate")
                    run("voice")
                    run("images")
                self.assertEqual(len(calls), 3)
                (root / "images/02.png").unlink()
                run("images")
                self.assertEqual(calls.count("api.openai.com"), 3)
                # Модель подставлена только в этом тесте; реальное выравнивание проверяется отдельно.
                with patch("shorts_ai.alignment.align", return_value=words):
                    run("align")
                duration = json.loads((root / "timeline.json").read_text())["duration"]
                _, frames = read_timeline(root / "timeline.json", root / "voice.mp3", duration, 2)
                self.assertEqual(frames[0], 66)
                run("montage", "--timeline", "timeline.json", "--output", "captioned.mp4")
                run("montage", "--timeline", "timeline.json", "--no-subtitles", "--output", "plain.mp4")
            for name in ("captioned.mp4", "plain.mp4"):
                info = json.loads(command(probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(root / name)))
                video, sound = info["streams"]
                self.assertEqual((video["codec_name"], video["width"], video["height"], video["r_frame_rate"]),
                                 ("h264", 1080, 1920, "30/1"))
                self.assertEqual(sound["codec_name"], "aac")
                self.assertAlmostEqual(float(info["format"]["duration"]), duration, delta=.1)
            self.assertFalse(list(root.glob(".montage-*")))
            self.assertFalse(list(root.glob(".shorts-ai-*")))


if __name__ == "__main__":
    unittest.main()
