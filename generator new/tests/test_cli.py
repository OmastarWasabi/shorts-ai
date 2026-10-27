from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shorts_ai.cli import main


class CLITests(unittest.TestCase):
    def call_main(self, directory, *arguments, env=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env or {}, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["--project-dir", str(directory), *arguments])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_disabled_generators_need_no_files_keys_or_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "ролик с пробелами"
            directory.mkdir()
            with patch("httpx.AsyncClient", side_effect=AssertionError("HTTP must stay disabled")):
                code, stdout, stderr = self.call_main(
                    directory, "generate", env={"GENERATE_VOICE": "false", "GENERATE_IMAGES": "false"}
                )
            self.assertEqual(code, 0)
            self.assertIn("Все генерации отключены", stdout)
            self.assertEqual(stderr, "")
            self.assertEqual(list(directory.iterdir()), [])

    def test_env_is_loaded_from_selected_project_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".env").write_text("GENERATE_VOICE=false\nGENERATE_IMAGES=false\n", encoding="utf-8")
            code, _, stderr = self.call_main(directory, "generate")
            self.assertEqual((code, stderr), (0, ""))
            (directory / "story.txt").write_text("Тест.", encoding="utf-8")
            (directory / "voice.mp3").write_bytes(b"ID3cached")
            code, _, stderr = self.call_main(directory, "generate", env={"GENERATE_VOICE": "true"})
            self.assertEqual((code, stderr), (0, ""))

    def test_align_reports_missing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            for command in ("align",):
                with self.subTest(command=command):
                    code, stdout, stderr = self.call_main(temporary, command)
                    self.assertEqual(code, 1)
                    self.assertEqual(stdout, "")
                    self.assertIn("scenes.json", stderr)

    def test_local_commands_ignore_env_and_explicit_generators_ignore_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".env").write_text("BROKEN=\"do-not-print-this\n", encoding="utf-8")
            for arguments in (("align",), ("montage", "--timeline", "timeline.json")):
                code, _, stderr = self.call_main(directory, *arguments)
                self.assertEqual(code, 1)
                if arguments[0] == "align":
                    self.assertIn("scenes.json", stderr)
                else:
                    self.assertNotIn("этап", stderr)
                self.assertNotIn("do-not-print-this", stderr)
            (directory / ".env").write_text("GENERATE_VOICE=invalid\nGENERATE_IMAGES=invalid\n", encoding="utf-8")
            (directory / "story.txt").write_text("Тест.", encoding="utf-8")
            (directory / "voice.mp3").write_bytes(b"ID3cached")
            code, _, stderr = self.call_main(directory, "voice")
            self.assertEqual((code, stderr), (0, ""))
            code, _, stderr = self.call_main(directory, "images")
            self.assertEqual(code, 1)
            self.assertIn("prompts.txt", stderr)
            self.assertNotIn("должен быть true", stderr)

    def test_bad_flags_and_env_errors_do_not_expose_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            code, stdout, stderr = self.call_main(directory, "generate", env={"GENERATE_VOICE": "secret-value"})
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("GENERATE_VOICE", stderr)
            self.assertNotIn("secret-value", stderr)
            (directory / ".env").write_text("FISH_API_KEY=\"do-not-print-this\n", encoding="utf-8")
            code, _, stderr = self.call_main(directory, "generate")
            self.assertEqual(code, 1)
            self.assertIn("строке 1", stderr)
            self.assertNotIn("do-not-print-this", stderr)

    def test_missing_project_and_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as temporary:
            code, _, stderr = self.call_main(Path(temporary) / "missing", "generate")
            self.assertEqual(code, 1)
            self.assertIn("Каталог проекта", stderr)
        with patch("shorts_ai.cli.project_directory", side_effect=KeyboardInterrupt):
            code, _, stderr = self.call_main(".", "generate")
        self.assertEqual(code, 130)
        self.assertIn("прервано", stderr)

    def test_installed_module_and_console_command_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            project = directory / "generator"
            project.mkdir()
            env = os.environ.copy()
            env.update(GENERATE_VOICE="false", GENERATE_IMAGES="false")
            for command in ([sys.executable, "-m", "shorts_ai"], [str(Path(sys.executable).parent / "shorts-ai")]):
                result = subprocess.run([*command, "generate"], cwd=directory, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Все генерации отключены", result.stdout)
            result = subprocess.run(
                [sys.executable, "-m", "shorts_ai", "--project-dir", str(project), "generate", "--unknown"],
                cwd=directory, env=env, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 2)

    def test_help_version_and_disabled_mode_do_not_import_optional_libraries(self):
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
        with tempfile.TemporaryDirectory() as temporary:
            env = os.environ.copy()
            env.update(GENERATE_VOICE="false", GENERATE_IMAGES="false")
            for arguments in (["--help"], ["--version"], ["montage", "--help"], ["--project-dir", temporary, "generate"]):
                with self.subTest(arguments=arguments):
                    result = subprocess.run([sys.executable, "-c", script, *arguments], env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
