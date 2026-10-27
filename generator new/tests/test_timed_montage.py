from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

from shorts_ai.cli import main
from shorts_ai.ffmpeg import command, ffmpeg, find_tools, require_ass
from shorts_ai.montage import render
from shorts_ai.timeline import file_hash


def timing_for(audio):
    return {"version": 1, "duration": 1.2, "audio_sha256": file_hash(audio),
            "scenes": [{"number": 1, "text": "Первый кадр", "start": 0, "end": .35},
                       {"number": 2, "text": "Вторая сцена", "start": .35, "end": 1.2}],
            "words": [{"text": "Первый", "start": .02, "end": .1, "scene": 1},
                      {"text": "кадр", "start": .15, "end": .25, "scene": 1},
                      {"text": "Вторая", "start": .35, "end": .55, "scene": 2},
                      {"text": "сцена", "start": .75, "end": 1.1, "scene": 2}]}


class TimedMontageTests(unittest.TestCase):
    def inputs(self, root):
        audio, timeline, output = root / "voice.wav", root / "timeline.json", root / "result.mp4"
        audio.write_bytes(b"test recording")
        for name in ("01.png", "02.png"):
            (root / name).touch()
        timeline.write_text(json.dumps(timing_for(audio)))
        return audio, timeline, output

    def test_invalid_timeline_and_missing_ass_fail_before_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio, timeline, output = self.inputs(root)
            with patch("shorts_ai.montage.find_tools", return_value=("encoder", "probe")), \
                 patch("shorts_ai.montage.audio_duration", return_value=1.2), \
                 patch("shorts_ai.montage.render_scene") as scene, \
                 patch("shorts_ai.montage.require_ass", side_effect=ValueError("libass missing")) as ass:
                original = timeline.read_bytes()
                for kind in ("hash", "gap", "word"):
                    data = json.loads(original)
                    if kind == "hash":
                        data["audio_sha256"] = "wrong"
                    elif kind == "gap":
                        data["scenes"][1]["start"] = .4
                    else:
                        data["words"][0]["end"] = .6
                    timeline.write_text(json.dumps(data))
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        render(root, audio, output, timeline=timeline)
                ass.assert_not_called()
                timeline.write_bytes(original)
                with self.assertRaisesRegex(ValueError, "libass missing"):
                    render(root, audio, output, timeline=timeline)
                scene.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".montage-*")), [])

    def test_changed_audio_after_mux_is_not_published_in_either_mode(self):
        for subtitles in (False, True):
            with self.subTest(subtitles=subtitles), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                audio, timeline, output = self.inputs(root)
                def changed_mux(binary, listing, audio, duration, final, **kwargs):
                    final.write_bytes(b"partial video")
                    audio.write_bytes(b"another recording")
                with patch("shorts_ai.montage.find_tools", return_value=("encoder", "probe")), \
                     patch("shorts_ai.montage.audio_duration", return_value=1.2), \
                     patch("shorts_ai.montage.render_scene") as scene, patch("shorts_ai.montage.require_ass"), \
                     patch("shorts_ai.montage.mux", side_effect=changed_mux), redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "изменилась"):
                        render(root, audio, output, timeline=timeline, subtitles=subtitles)
                    self.assertEqual([call.args[2] for call in scene.call_args_list], [11, 25])
                self.assertFalse(output.exists())
                self.assertEqual(list(root.glob(".montage-*")), [])

    def test_no_subtitles_needs_no_ass_and_mux_failure_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio, timeline, output = self.inputs(root)
            def failing_mux(binary, listing, audio, duration, final, **kwargs):
                self.assertFalse(kwargs["subtitles"])
                self.assertFalse((final.parent / "captions.ass").exists())
                final.write_bytes(b"partial")
                raise ValueError("encoder failed")
            with patch("shorts_ai.montage.find_tools", return_value=("encoder", "probe")), \
                 patch("shorts_ai.montage.audio_duration", return_value=1.2), \
                 patch("shorts_ai.montage.render_scene"), patch("shorts_ai.montage.require_ass") as ass, \
                 patch("shorts_ai.montage.mux", side_effect=failing_mux), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "encoder failed"):
                    render(root, audio, output, timeline=timeline, subtitles=False)
                ass.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".montage-*")), [])

    def test_cli_passes_absolute_timeline_and_no_subtitles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            timeline = root / "тайминги.json"
            with patch("shorts_ai.montage.render") as renderer, redirect_stdout(io.StringIO()):
                code = main(["--project-dir", str(root), "montage", "--timeline", str(timeline), "--no-subtitles"])
            self.assertEqual(code, 0)
            renderer.assert_called_once_with(root / "images", root / "voice.mp3", root / "final.mp4",
                                             timeline=timeline, subtitles=False)


class TimedRenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.encoder, cls.probe = find_tools()
            require_ass(cls.encoder)
        except ValueError as error:
            raise unittest.SkipTest(str(error)) from error

    def media(self, root):
        for name, color in (("01.png", "red"), ("02.png", "blue")):
            Image.new("RGB", (160, 120), color).save(root / name)
        audio, timeline = root / "voice.wav", root / "timeline.json"
        ffmpeg(self.encoder, "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2", str(audio))
        timeline.write_text(json.dumps(timing_for(audio), ensure_ascii=False), encoding="utf-8")
        return audio, timeline

    def crop(self, path, timestamp):
        return ffmpeg(self.encoder, "-ss", str(timestamp), "-i", str(path), "-vf", "crop=1080:500:0:1420",
                      "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")

    def test_real_half_frame_cut_last_frame_captions_and_delayed_highlight(self):
        with tempfile.TemporaryDirectory(prefix="сцены 'тест' ") as temporary:
            root = Path(temporary)
            audio, timeline = self.media(root)
            outputs = []
            for subtitles in (False, True):
                output = root / f"result-{subtitles}.mp4"
                with redirect_stderr(io.StringIO()):
                    render(root, audio, output, timeline=timeline, subtitles=subtitles)
                outputs.append(output)
                info = json.loads(command(self.probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)))
                video, sound = info["streams"]
                self.assertEqual([video[key] for key in ("width", "height", "r_frame_rate", "nb_frames", "codec_name")],
                                 [1080, 1920, "30/1", "36", "h264"])
                self.assertEqual(sound["codec_name"], "aac")
                self.assertAlmostEqual(float(info["format"]["duration"]), 1.2, delta=1/30)
                pixels = ffmpeg(self.encoder, "-i", str(output), "-vf", "select='eq(n,10)+eq(n,11)+eq(n,35)',crop=2:2:540:960",
                                "-frames:v", "3", "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1")
                self.assertEqual(len(pixels), 36)
                for index, expected in enumerate(((255, 0, 0), (0, 0, 255), (0, 0, 255))):
                    for got, want in zip(pixels[index * 12:index * 12 + 3], expected):
                        self.assertAlmostEqual(got, want, delta=12)
            def colors(pixels):
                white = yellow = 0
                for r, g, b in zip(pixels[::3], pixels[1::3], pixels[2::3]):
                    white += r > 180 and g > 180 and b > 180
                    yellow += r > 180 and g > 130 and b < 80
                return white, yellow
            self.assertEqual(colors(self.crop(outputs[0], .6)), (0, 0))
            # Во время паузы второе слово ещё белое; после его начала становится жёлтым.
            white_before, yellow_before = colors(self.crop(outputs[1], .6))
            white_after, yellow_after = colors(self.crop(outputs[1], .85))
            self.assertGreater(white_before, 100)
            self.assertGreater(yellow_before, 100)
            self.assertLess(white_after, 20)
            self.assertGreater(yellow_after, yellow_before + 100)
            self.assertEqual(list(root.glob(".montage-*")), [])

    def test_sigint_during_caption_mux_reaps_encoder_and_cleans_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio, timeline = self.media(root)
            wrapper, marker = root / "encoder", root / "mux.pid"
            script = f"import os,sys\nfrom pathlib import Path\nargs=sys.argv[1:]\nif 'concat' in args:\n Path({str(marker)!r}).write_text(str(os.getpid()))\n args=['-re',*args]\nos.execv({self.encoder!r},[{self.encoder!r},*args])\n"
            wrapper.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(script) + ' "$@"\n')
            wrapper.chmod(0o700)
            process = subprocess.Popen([sys.executable, "-m", "shorts_ai", "--project-dir", str(root), "montage", "--images", ".",
                                        "--audio", str(audio), "--timeline", str(timeline)],
                                       env={**os.environ, "FFMPEG_BIN": str(wrapper)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            child_pid = None
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if marker.exists() and list(root.glob(".montage-*/final.mp4")):
                        child_pid = int(marker.read_text())
                        break
                    if process.poll() is not None:
                        self.fail(f"Exited before mux interrupt: {process.communicate()}")
                    time.sleep(.01)
                self.assertIsNotNone(child_pid, "Caption mux did not start")
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 130, (stdout, stderr))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertFalse((root / "final.mp4").exists())
                self.assertEqual(list(root.glob(".montage-*")), [])
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
