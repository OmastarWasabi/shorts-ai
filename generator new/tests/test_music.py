from contextlib import chdir, redirect_stdout
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from shorts_ai.cli import main
from shorts_ai.ffmpeg import command, ffmpeg, find_tools, mux
from shorts_ai.montage import render


class MusicTests(unittest.TestCase):
    def test_both_commands_resolve_music_from_terminal_not_current_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            inputs, run = root / "inputs", root / "video"
            inputs.mkdir()
            run.mkdir()
            (inputs / ".current_run").write_text(str(run))
            music = Path("музыка 'фон'.mp3")
            with chdir(root), patch("shorts_ai.pipeline.run_pipeline", return_value=run / "final.mp4") as pipeline, \
                 patch("shorts_ai.montage.render") as montage, redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(inputs), "run", "--music", str(music)]), 0)
                self.assertEqual(pipeline.call_args.kwargs["music"], root / music)
                self.assertEqual(main(["--project-dir", str(inputs), "montage", "--music", str(music)]), 0)
                self.assertEqual(montage.call_args.kwargs["music"], root / music)
                self.assertEqual(montage.call_args.args[2], run / "final.mp4")

    def test_bad_music_fails_before_rendering_any_scene(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch("shorts_ai.montage.find_tools", return_value=("encoder", "probe")), \
                 patch("shorts_ai.montage.read_images", return_value=[root / "01.png"]), \
                 patch("shorts_ai.montage.audio_duration", side_effect=[4, ValueError("bad music")]), \
                 patch("shorts_ai.montage.render_scene") as scene:
                with self.assertRaisesRegex(ValueError, "bad music"):
                    render(root, root / "voice.wav", root / "final.mp4", music=root / "missing.mp3")
            scene.assert_not_called()
            self.assertFalse((root / "final.mp4").exists())
            self.assertFalse(list(root.glob(".montage-*")))


class MusicIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.encoder, cls.probe = find_tools()
        except ValueError as error:
            raise unittest.SkipTest(str(error)) from error

    def amplitude(self, samples, frequency, start, length=.1):
        # Амплитуда конкретного тестового тона в декодированной дорожке.
        window = samples[int(start * 8000):int((start + length) * 8000)]
        real = sum(sample * math.cos(2 * math.pi * frequency * i / 8000) for i, sample in enumerate(window))
        imag = sum(sample * math.sin(2 * math.pi * frequency * i / 8000) for i, sample in enumerate(window))
        return 2 * math.hypot(real, imag) / len(window)

    def test_real_mix_loops_short_track_trims_long_track_and_fades_only_music(self):
        with tempfile.TemporaryDirectory(prefix="музыка 'тест' ") as temporary:
            root = Path(temporary).resolve()
            clip, voice = root / "clip.mp4", root / "voice.wav"
            ffmpeg(self.encoder, "-f", "lavfi", "-i", "color=c=red:s=160x120:r=30:d=4",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip))
            ffmpeg(self.encoder, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=4", str(voice))
            listing = root / "scenes.txt"
            listing.write_text("file 'clip.mp4'\n")
            for length in (1, 6):
                with self.subTest(music_length=length):
                    music = root / f"фон {length}.mp3"
                    # Длинный трек стерео, короткий моно; частота дискретизации отличается от голоса.
                    channels = ["-af", "pan=stereo|c0=c0|c1=c0"] if length == 6 else []
                    ffmpeg(self.encoder, "-f", "lavfi", "-i",
                           f"sine=frequency=1000:sample_rate=48000:duration={length}", *channels, str(music))
                    output = root / f"mixed-{length}.mp4"
                    mux(self.encoder, listing, voice, 4, output, music=music)
                    info = json.loads(command(self.probe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)))
                    self.assertEqual([stream["codec_type"] for stream in info["streams"]], ["video", "audio"])
                    self.assertEqual(info["streams"][1]["codec_name"], "aac")
                    self.assertAlmostEqual(float(info["format"]["duration"]), 4, delta=.04)
                    pcm = ffmpeg(self.encoder, "-i", str(output), "-map", "0:a:0", "-ac", "1",
                                 "-ar", "8000", "-f", "f32le", "pipe:1")
                    samples = struct.unpack(f"<{len(pcm) // 4}f", pcm)
                    # Стерео → моно меняет уровень: сравниваем с тем же преобразованием исходника.
                    source_pcm = ffmpeg(self.encoder, "-i", str(music), "-ac", "1", "-ar", "8000", "-f", "f32le", "pipe:1")
                    source_samples = struct.unpack(f"<{len(source_pcm) // 4}f", source_pcm)
                    source_music = self.amplitude(source_samples, 1000, .4)
                    early_voice = self.amplitude(samples, 440, .4)
                    early_music = self.amplitude(samples, 1000, .4)
                    self.assertGreater(early_voice, .07)
                    self.assertGreater(source_music, .1)
                    self.assertAlmostEqual(early_music, source_music * .1, delta=.001)
                    self.assertAlmostEqual(self.amplitude(samples, 1000, 1.4) / early_music, 1, delta=.1)
                    self.assertAlmostEqual(self.amplitude(samples, 1000, 3.8) / early_music, .075, delta=.04)
                    self.assertAlmostEqual(self.amplitude(samples, 440, 3.8) / early_voice, 1, delta=.05)
                    if length == 1:
                        self.assertAlmostEqual(early_music / early_voice, .1, delta=.02)


if __name__ == "__main__":
    unittest.main()
