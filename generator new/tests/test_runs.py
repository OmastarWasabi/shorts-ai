from contextlib import chdir, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from shorts_ai.cli import main
from shorts_ai.files import create_run, current_run
from shorts_ai.transport import create_client
from test_images import image_response

FIXTURES = Path(__file__).parent / "fixtures/migration"


class RunTests(unittest.TestCase):
    def inputs(self, root):
        for name in ("story.txt", "scenes.json"):
            (root / name).write_bytes((FIXTURES / name).read_bytes())
        (root / "prompts.txt").write_text("[01]\nFirst.\n[02]\nSecond.", encoding="utf-8")

    def test_create_run_copies_only_inputs_and_remembers_external_output(self):
        with tempfile.TemporaryDirectory(prefix="папки 'ролика' ") as temporary:
            root = Path(temporary).resolve()
            source = root / "inputs"
            source.mkdir()
            original = {"story.txt": b"story", "prompts.txt": b"[01]\nScene.",
                        ".env": b"FISH_API_KEY=test-only", "voice.mp3": b"old voice"}
            for name, content in original.items():
                (source / name).write_bytes(content)
            (source / "images").mkdir()
            (source / "images/01.png").write_bytes(b"old image")
            destination = root / "videos" / "nested" / "Новый ролик"
            run = create_run(source, destination)
            self.assertEqual(run, destination)
            self.assertEqual({path.name for path in run.iterdir()}, {"story.txt", "prompts.txt"})
            for name in ("story.txt", "prompts.txt"):
                self.assertEqual((run / name).read_bytes(), original[name])
            self.assertEqual((source / ".current_run").read_text(), str(destination))
            self.assertEqual(current_run(source), destination)
            for name, content in original.items():
                self.assertEqual((source / name).read_bytes(), content)

    def test_generate_requires_output_dir_on_every_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.inputs(root)
            create_run(root, root / "previous")
            with redirect_stderr(io.StringIO()) as errors, self.assertRaises(SystemExit) as error:
                main(["--project-dir", str(root), "generate"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("--output-dir", errors.getvalue())

    def test_relative_output_is_resolved_from_terminal_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary).resolve()
            root = work / "inputs"
            root.mkdir()
            self.inputs(root)
            with chdir(work), patch.dict(os.environ, {"GENERATE_VOICE": "true", "GENERATE_IMAGES": "true"}), \
                 patch("shorts_ai.generation.run_generation", new_callable=AsyncMock) as generate, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(root), "generate", "--output-dir", "videos/Мой ролик"]), 0)
            output = work / "videos/Мой ролик"
            self.assertEqual(current_run(root), output)
            self.assertEqual(generate.call_args.args[0], output)
            self.assertTrue((output / "story.txt").is_file())

    def test_each_generate_creates_new_run_with_its_own_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.inputs(root)
            (root / ".env").write_text("FISH_API_KEY=test-fish\nOPENAI_API_KEY=test-openai\n")
            (root / "voice.mp3").write_bytes(b"old audio")
            calls = []

            def handle(request):
                calls.append(request)
                if request.url.host == "api.fish.audio":
                    return httpx.Response(200, content=b"new audio", headers={"Content-Type": "audio/mpeg"})
                return image_response()

            def client():
                return create_client(transport=httpx.MockTransport(handle))

            with patch.dict(os.environ, {"NO_PROXY": "*"}, clear=True), \
                 patch("shorts_ai.generation.create_client", side_effect=client), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                for number in (1, 2):
                    run = root / "videos" / f"Ролик {number}"
                    self.assertEqual(main(["--project-dir", str(root), "generate", "--output-dir", str(run)]), 0)
                    self.assertEqual(current_run(root), run)
                    self.assertEqual((run / "voice.mp3").read_bytes(), b"new audio")
                    self.assertTrue((run / "images/01.png").is_file())
                    self.assertTrue((run / "images/02.png").is_file())
                    for name in ("story.txt", "prompts.txt", "scenes.json"):
                        self.assertEqual((run / name).read_bytes(), (root / name).read_bytes())
            self.assertEqual(len(calls), 6)
            image_prompts = [json.loads(request.content)["prompt"] for request in calls if request.url.host == "api.openai.com"]
            self.assertEqual(image_prompts, ["First.", "Second.", "First.", "Second."])
            self.assertEqual((root / "voice.mp3").read_bytes(), b"old audio")

    def test_align_and_montage_continue_current_run_using_copied_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.inputs(root)
            run = create_run(root, root / "output")
            original = (run / "story.txt").read_bytes()
            (root / "story.txt").write_text("Сценарий следующего ролика.")
            with patch("shorts_ai.alignment.run_alignment") as align, \
                 patch("shorts_ai.montage.render") as montage, redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(root), "align"]), 0)
                align.assert_called_once_with(
                    run / "scenes.json", run / "story.txt", run / "voice.mp3", run / "timeline.json",
                    device="cpu", force=False, download_model=False, prompts_path=run / "prompts.txt",
                )
                self.assertEqual(main(["--project-dir", str(root), "montage", "--timeline", "timeline.json"]), 0)
                montage.assert_called_once_with(run / "images", run / "voice.mp3", run / "final.mp4",
                                                timeline=run / "timeline.json", subtitles=True, music=None)
            self.assertEqual((run / "story.txt").read_bytes(), original)

    def test_explicit_run_directory_selects_an_earlier_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.inputs(root)
            first = create_run(root, root / "first")
            create_run(root, root / "second")
            with patch("shorts_ai.montage.render") as montage, redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--project-dir", str(first), "montage"]), 0)
                self.assertEqual(montage.call_args.args[2], first / "final.mp4")


if __name__ == "__main__":
    unittest.main()
