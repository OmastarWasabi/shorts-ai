from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shorts_ai.cli import main
from shorts_ai.scenes import create_scenes
from shorts_ai.timeline import read_timeline

FIXTURES = Path(__file__).parent / "fixtures/migration"


class SceneTests(unittest.TestCase):
    def inputs(self, root):
        script, prompts, output = [root / name for name in ("story.txt", "prompts.txt", "scenes.json")]
        script.write_text("\ufeffЁжик спит.\r\n\r\n  \r\nЁжик\r\nпроснулся.\r\n", encoding="utf-8")
        prompts.write_text("[01]\nSleeping hedgehog.\n[02]\nWaking hedgehog.", encoding="utf-8")
        return script, prompts, output

    def test_paragraphs_preserve_words_and_match_numbered_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            script, prompts, output = self.inputs(Path(temporary))
            result = create_scenes(script, prompts, output)
            self.assertEqual(result["scenes"], [{"number": 1, "text": "Ёжик спит."},
                                                {"number": 2, "text": "Ёжик\nпроснулся."}])
            self.assertEqual(json.loads(output.read_text()), result)
            from shorts_ai.alignment import read_scenes
            self.assertEqual(read_scenes(output, script), (result["scenes"], "ru"))

    def test_mismatch_empty_text_and_invalid_prompts_do_not_create_file(self):
        for kind in ("mismatch", "empty", "punctuation", "invalid_prompts"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                script, prompts, output = self.inputs(Path(temporary))
                if kind == "mismatch":
                    script.write_text("Один абзац.")
                elif kind == "empty":
                    script.write_text(" \n ")
                elif kind == "punctuation":
                    script.write_text("...\n\nЁжик проснулся.")
                else:
                    prompts.write_text("[02]\nInvalid numbering.")
                with self.assertRaises(ValueError):
                    create_scenes(script, prompts, output)
                self.assertFalse(output.exists())

    def test_existing_markup_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            script, prompts, output = self.inputs(Path(temporary))
            output.write_bytes(b"previous markup")
            with self.assertRaises(FileExistsError):
                create_scenes(script, prompts, output)
            self.assertEqual(output.read_bytes(), b"previous markup")

    def test_align_creates_missing_markup_then_valid_timeline(self):
        with tempfile.TemporaryDirectory(prefix="авто сцены ") as temporary:
            root = Path(temporary)
            self.inputs(root)
            (root / "voice.mp3").write_bytes(b"test recording")
            words = json.loads((FIXTURES / "aligned_words.json").read_text())
            with patch("shorts_ai.alignment.audio_duration", return_value=4), \
                 patch("shorts_ai.alignment.align", return_value=words), redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(["--project-dir", str(root), "align"]), 0)
            self.assertIn("Авторазметка", stdout.getvalue())
            _, frames = read_timeline(root / "timeline.json", root / "voice.mp3", 4, 2)
            self.assertEqual(frames, [66, 54])
            self.assertFalse(list(root.glob(".shorts-ai-*")))

    def test_existing_manual_markup_needs_no_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("story.txt", "scenes.json"):
                (root / name).write_bytes((FIXTURES / name).read_bytes())
            (root / "voice.mp3").write_bytes(b"test recording")
            words = json.loads((FIXTURES / "aligned_words.json").read_text())
            before = (root / "scenes.json").read_bytes()
            with patch("shorts_ai.alignment.audio_duration", return_value=4), \
                 patch("shorts_ai.alignment.align", return_value=words), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(root), "align"]), 0)
            self.assertEqual((root / "scenes.json").read_bytes(), before)

    def test_force_cannot_replace_prompt_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, prompts, _ = self.inputs(root)
            before = prompts.read_bytes()
            with redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(main(["--project-dir", str(root), "align", "--output", "prompts.txt", "--force"]), 1)
            self.assertIn("входной файл", errors.getvalue())
            self.assertEqual(prompts.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
