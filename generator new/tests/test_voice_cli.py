from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from shorts_ai.cli import main
from shorts_ai.transport import create_client


class VoiceCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "ролик с 'кавычками'"
        self.directory.mkdir()
        (self.directory / ".env").write_text(
            "FISH_API_KEY=test-fish-key\nFISH_REFERENCE_ID=test-voice\nFISH_MODEL=s2.1-pro-free\n"
            "GENERATE_VOICE=true\nGENERATE_IMAGES=false\n", encoding="utf-8",
        )
        (self.directory / "story.txt").write_text("Ёжик проснулся.", encoding="utf-8")
        self.calls = []

    def run_command(self, *args, env=None, response=None):
        def handle(request):
            self.calls.append(request)
            return response if response is not None else httpx.Response(200, content=b"ID3test-audio")

        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, env or {}, clear=True), redirect_stdout(output), redirect_stderr(errors):
            with patch("shorts_ai.generation.create_client", side_effect=lambda: create_client(transport=httpx.MockTransport(handle))):
                code = main(["--project-dir", str(self.directory), *args])
        return code, output.getvalue(), errors.getvalue()

    def test_voice_and_generate_voice_only(self):
        for command in ("voice", "generate"):
            with self.subTest(command=command):
                (self.directory / "voice.mp3").unlink(missing_ok=True)
                code, output, errors = self.run_command(command)
                self.assertEqual((code, errors), (0, ""))
                self.assertIn("Готово:", output)
                self.assertEqual((self.directory / "voice.mp3").read_bytes(), b"ID3test-audio")
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(request.url.host == "api.fish.audio" for request in self.calls))

    def test_explicit_voice_ignores_both_flags_and_custom_paths_work(self):
        script = self.directory / "custom story.txt"
        script.write_text("Другой сценарий.", encoding="utf-8")
        env = {"GENERATE_VOICE": "false", "GENERATE_IMAGES": "true"}
        code, _, errors = self.run_command("voice", "--script", script.name, "--output", "audio/result.mp3", env=env)
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual((self.directory / "audio/result.mp3").read_bytes(), b"ID3test-audio")
        self.assertEqual(json.loads(self.calls[0].content)["text"], "Другой сценарий.")
        absolute = self.directory.parent / "absolute.mp3"
        code, _, errors = self.run_command("voice", "--script", str(script), "--output", str(absolute))
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(absolute.read_bytes(), b"ID3test-audio")

    def test_missing_prompts_stop_generate_before_any_paid_request(self):
        for voice in ("true", "false"):
            with self.subTest(voice=voice):
                code, output, errors = self.run_command("generate", env={"GENERATE_VOICE": voice, "GENERATE_IMAGES": "true"})
                self.assertEqual(code, 1)
                self.assertIn("prompts.txt", errors)
                self.assertEqual(output, "")
                self.assertEqual(self.calls, [])
                self.assertFalse((self.directory / "voice.mp3").exists())

    def test_http_error_has_nonzero_exit_and_no_secrets_or_result(self):
        code, _, errors = self.run_command("voice", response=httpx.Response(401, content=b"secret-provider-message"))
        self.assertEqual(code, 1)
        self.assertIn("HTTP 401", errors)
        self.assertNotIn("secret-provider-message", errors)
        self.assertNotIn("test-fish-key", errors)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.directory / "voice.mp3").exists())

    def test_sigint_during_request_exits_130_and_preserves_inputs(self):
        script = """
import asyncio, sys
import httpx
import shorts_ai.generation as generation
from shorts_ai.cli import main
from shorts_ai.transport import create_client
class WaitingAudio(httpx.AsyncByteStream):
    async def __aiter__(self):
        print('REQUEST_STARTED', file=sys.stderr, flush=True)
        await asyncio.sleep(60)
        yield b'ID3audio'
generation.create_client = lambda: create_client(transport=httpx.MockTransport(
    lambda request: httpx.Response(200, stream=WaitingAudio())))
sys.exit(main(sys.argv[1:]))
"""
        env = os.environ.copy()
        env.update(FISH_API_KEY="test-fish-key", FISH_REFERENCE_ID="test-voice", FISH_MODEL="s2.1-pro-free")
        process = subprocess.Popen(
            [sys.executable, "-c", script, "--project-dir", str(self.directory), "voice"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stderr, selectors.EVENT_READ)
                self.assertTrue(selector.select(timeout=5), "request did not start")
                self.assertEqual(process.stderr.readline().strip(), "REQUEST_STARTED")
            process.send_signal(signal.SIGINT)
            _, errors = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130, errors)
            self.assertIn("прервано", errors)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
            process.stdout.close()
            process.stderr.close()
        self.assertFalse((self.directory / "voice.mp3").exists())
        self.assertEqual(list(self.directory.glob(".shorts-ai-*")), [])
        self.assertEqual((self.directory / "story.txt").read_text(encoding="utf-8"), "Ёжик проснулся.")


if __name__ == "__main__":
    unittest.main()
