from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from shorts_ai import alignment
from shorts_ai.cli import main
from shorts_ai.ffmpeg import command
from shorts_ai.timeline import read_timeline

FIXTURES = Path(__file__).parent / "fixtures/migration"


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.scenes, _ = alignment.read_scenes(FIXTURES / "scenes.json", FIXTURES / "story.txt")
        self.words = json.loads((FIXTURES / "aligned_words.json").read_text())

    def inputs(self, root):
        for name in ("scenes.json", "story.txt"):
            (root / name).write_bytes((FIXTURES / name).read_bytes())
        (root / "voice.mp3").write_bytes(b"test audio")
        return [root / name for name in ("scenes.json", "story.txt", "voice.mp3", "timeline.json")]

    def test_original_normalization_and_timeline_fixture(self):
        self.assertEqual(alignment.tokens("Гиппока́мп, ребёнок!"), alignment.tokens("гиппокамп ребенок"))
        self.assertEqual(alignment.alignment_text("Гиппока́мп и маленький ёжик"), "Гиппокамп и маленький ёжик")
        expected = json.loads((FIXTURES / "timeline_valid.json").read_text())
        result = alignment.build_timeline(self.scenes, self.words, 4, expected["audio_sha256"])
        self.assertEqual(result, expected)
        # Повтор «Ёжик» относится к следующей сцене по позиции в сценарии.
        self.assertEqual([word["scene"] for word in result["words"]], [1, 1, 2, 2])
        self.assertEqual(result["scenes"][0]["end"], 2.2)

    def test_bad_word_alignment_and_cross_scene_boundaries(self):
        cases = [None, [None], [{"word": 123}], self.words[:-1], [self.words[1], self.words[0], *self.words[2:]]]
        for field, value in (("start", None), ("start", math.nan), ("start", True), ("start", "0.1"),
                             ("end", 0), ("end", 5), ("word", "Другой")):
            words = copy.deepcopy(self.words)
            words[0][field] = value
            cases.append(words)
        words = copy.deepcopy(self.words)
        words[1]["end"] = 2.5
        cases.append(words)
        for words in cases:
            with self.subTest(words=words), self.assertRaises(ValueError):
                alignment.build_timeline(self.scenes, words, 4, "hash")
        with self.assertRaisesRegex(ValueError, "внутри слова"):
            alignment.build_timeline([{"number": 1, "text": "Мир"}, {"number": 2, "text": "труд"}],
                                     [{"word": "Мир-труд", "start": 0, "end": 1}], 2, "hash")

    def test_scenes_require_matching_script_and_valid_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            scenes, script, _, _ = self.inputs(Path(temporary))
            valid = json.loads(scenes.read_text())
            for value in (True, 1.0, None, "1", 2):
                document = {**valid, "version": value}
                scenes.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    alignment.read_scenes(scenes, script)
            for key, value in (("language", "RU"), ("language", 1), ("scenes", []), ("scenes", [None]),
                               ("scenes", [{"number": True, "text": "Ёжик"}]), ("scenes", [{"number": 1, "text": "..."}])):
                scenes.write_text(json.dumps({**valid, key: value}))
                with self.assertRaises(ValueError):
                    alignment.read_scenes(scenes, script)
            scenes.write_text("\ufeff" + json.dumps(valid), encoding="utf-8")
            self.assertEqual(alignment.read_scenes(scenes, script)[0], self.scenes)
            script.write_text("Совсем другой сценарий")
            with self.assertRaisesRegex(ValueError, "совпадать"):
                alignment.read_scenes(scenes, script)

    def test_cli_paths_flags_no_env_and_valid_saved_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "ролик 'с кавычками'"
            root.mkdir()
            scenes, script, audio, _ = self.inputs(root)
            (root / ".env").write_text('BROKEN="must-not-read\n')
            output = root / "nested/timing.json"
            with patch.object(alignment, "audio_duration", return_value=4), \
                 patch.object(alignment, "align", return_value=self.words) as model, redirect_stdout(io.StringIO()):
                code = main(["--project-dir", str(root), "align", "--audio", str(audio), "--output", "nested/timing.json",
                             "--device", "cuda", "--download-model"])
                self.assertEqual(code, 0)
                model.assert_called_once_with(self.scenes, "ru", audio, 4, "cuda", True)
                timing, frames = read_timeline(output, audio, 4, 2)
                self.assertEqual(frames, [66, 54])
                self.assertEqual(timing["audio_sha256"], alignment.file_hash(audio))
                before = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "уже существует"):
                    alignment.run_alignment(scenes, script, audio, output)
                self.assertEqual(output.read_bytes(), before)
                alignment.run_alignment(scenes, script, audio, output, force=True)
                self.assertEqual(output.read_bytes(), before)
            self.assertFalse(list(root.rglob(".shorts-ai-*")))

    def test_inputs_cannot_be_replaced_even_with_force_or_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes, script, audio, output = self.inputs(root)
            for source in (scenes, script, audio):
                for kind in ("same", "symlink", "hardlink"):
                    target = source
                    if kind != "same":
                        target = root / "alias.json"
                        if kind == "symlink":
                            target.symlink_to(source)
                        else:
                            os.link(source, target)
                    with patch.object(alignment, "align") as model, self.assertRaisesRegex(ValueError, "входной файл"):
                        alignment.run_alignment(scenes, script, audio, target, force=True)
                    model.assert_not_called()
                    if kind != "same":
                        target.unlink()
            output.symlink_to(root / "absent")
            with self.assertRaisesRegex(ValueError, "уже существует"):
                alignment.run_alignment(scenes, script, audio, output)

    def test_audio_replacement_during_probe_or_model_preserves_output(self):
        for phase in ("probe", "model"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as temporary:
                scenes, script, audio, output = self.inputs(Path(temporary))
                output.write_bytes(b"previous timeline")
                def replace_audio(*args):
                    audio.write_bytes(b"changed recording")
                    return 4 if phase == "probe" else self.words
                with patch.object(alignment, "audio_duration", side_effect=replace_audio if phase == "probe" else None, return_value=4), \
                     patch.object(alignment, "align", side_effect=replace_audio if phase == "model" else None, return_value=self.words):
                    with self.assertRaisesRegex(ValueError, "изменилась"):
                        alignment.run_alignment(scenes, script, audio, output, force=True)
                self.assertEqual(output.read_bytes(), b"previous timeline")

    def test_write_race_failure_and_ctrl_c_leave_no_partial_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            scenes, script, audio, output = self.inputs(Path(temporary))
            def competing_run(*args):
                output.write_bytes(b"other result")
                return self.words
            with patch.object(alignment, "audio_duration", return_value=4), patch.object(alignment, "align", side_effect=competing_run):
                with self.assertRaises(FileExistsError):
                    alignment.run_alignment(scenes, script, audio, output)
            self.assertEqual(output.read_bytes(), b"other result")
            output.unlink()
            for error in (KeyboardInterrupt(), RuntimeError("model failure")):
                with patch.object(alignment, "audio_duration", return_value=4), patch.object(alignment, "align", side_effect=error), \
                     redirect_stderr(io.StringIO()) as stderr:
                    code = main(["--project-dir", str(output.parent), "align"])
                self.assertEqual(code, 130 if isinstance(error, KeyboardInterrupt) else 1)
                self.assertNotIn("Traceback", stderr.getvalue())
                self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob(".shorts-ai-*")))

    def test_duration_limit_and_probe_timeout(self):
        with patch.object(alignment, "probe_duration", return_value=180) as probe:
            self.assertEqual(alignment.audio_duration(Path("audio.wav")), 180)
            probe.assert_called_once_with(Path("audio.wav").absolute(), timeout=30)
        with patch.object(alignment, "probe_duration", return_value=180.001), self.assertRaisesRegex(ValueError, "180"):
            alignment.audio_duration(Path("audio.wav"))
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "pid"
            script = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
            with self.assertRaisesRegex(ValueError, "Истекло"):
                command(sys.executable, "-c", script, timeout=.3)
            self.assertTrue(marker.exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)

    @contextmanager
    def model_modules(self, scores=None):
        words = copy.deepcopy(self.words)
        for word, score in zip(words, scores if scores is not None else [.8] * len(words)):
            word["score"] = score
        torch = SimpleNamespace(set_num_threads=Mock(), hub=SimpleNamespace(get_dir=Mock(return_value="/nonexistent-cache")))
        torchaudio = SimpleNamespace(pipelines=SimpleNamespace(TEST=SimpleNamespace(_path="checkpoint.pth")))
        module = ModuleType("whisperx.alignment")
        module.DEFAULT_ALIGN_MODELS_TORCH = {"en": "TEST"}
        module.load_align_model = Mock(return_value=("model", "metadata"))
        module.align = Mock(return_value={"word_segments": words})
        with patch.dict(sys.modules, {"torch": torch, "torchaudio": torchaudio, "whisperx": ModuleType("whisperx"), "whisperx.alignment": module}), \
             patch.object(alignment, "load_audio", return_value=range(64000)), redirect_stdout(io.StringIO()), patch.dict(os.environ):
            yield torch, module

    def test_model_cache_download_flag_cpu_threads_and_known_transcript(self):
        for download in (False, True):
            with self.model_modules() as (torch, model):
                words = alignment.align(self.scenes, "ru", Path("audio.mp3"), 4, "cpu", download)
                self.assertEqual(len(words), 4)
                model.load_align_model.assert_called_once_with(language_code="ru", device="cpu", model_cache_only=not download)
                torch.set_num_threads.assert_called_once_with(min(4, os.cpu_count() or 1))
                self.assertEqual(os.environ["DISABLE_SAFETENSORS_CONVERSION"], "true")
                transcript = model.align.call_args.args[0]
                self.assertEqual(transcript, [{"start": 0, "end": 4, "text": "Ёжик спит. Ёжик проснулся."}])
                self.assertFalse(model.align.call_args.kwargs["return_char_alignments"])
        with self.model_modules() as (_, model):
            model.load_align_model.side_effect = ValueError("missing cache")
            with self.assertRaisesRegex(ValueError, "--download-model"):
                alignment.align(self.scenes, "ru", Path("audio.mp3"), 4, "cpu")
        with self.model_modules() as (_, model):
            with self.assertRaisesRegex(ValueError, "--download-model"):
                alignment.align(self.scenes, "en", Path("audio.mp3"), 4, "cpu")
            model.load_align_model.assert_not_called()
            alignment.align(self.scenes, "en", Path("audio.mp3"), 4, "cpu", True)
            model.load_align_model.assert_called_once()

    def test_low_invalid_and_missing_confidence(self):
        for scores in ([.1] * 4, [math.nan] * 4, [math.inf] * 4, [True] * 4, [None] * 4, ["0.8"] * 4, []):
            with self.subTest(scores=scores), self.model_modules(scores), self.assertRaises(ValueError):
                alignment.align(self.scenes, "ru", Path("audio.mp3"), 4, "cpu")

    def test_missing_optional_dependencies_have_install_hint(self):
        with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "requirements-timing.lock.txt"):
            alignment.align(self.scenes, "ru", Path("audio.mp3"), 4, "cpu")


if __name__ == "__main__":
    unittest.main()
