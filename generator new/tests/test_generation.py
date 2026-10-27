import asyncio
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
from shorts_ai.config import GenerationConfig
from shorts_ai.generation import run_generation
from shorts_ai.transport import create_client
from test_images import BodyStream, image_response, png_bytes


class CombinedGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "story.txt").write_text("Тестовая озвучка.", encoding="utf-8")
        (self.root / "prompts.txt").write_text("[01]\nA scene.", encoding="utf-8")
        env = patch.dict(os.environ, {"FISH_API_KEY": "test-fish", "OPENAI_API_KEY": "test-openai"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        stdout = redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)
        self.clients = []

    def mock_api(self, handle):
        def client():
            result = create_client(transport=httpx.MockTransport(handle))
            self.clients.append(result)
            return result
        factory = patch("shorts_ai.generation.create_client", side_effect=client)
        factory.start()
        self.addCleanup(factory.stop)

    async def test_voice_and_images_really_overlap(self):
        release, ready = asyncio.Event(), asyncio.Event()
        hosts = set()

        async def handle(request):
            hosts.add(request.url.host)
            if len(hosts) == 2: ready.set()
            await release.wait()
            return httpx.Response(200, content=b"ID3audio") if request.url.host == "api.fish.audio" else image_response()

        self.mock_api(handle)
        task = asyncio.create_task(run_generation(self.root, GenerationConfig()))
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            self.assertEqual(len(self.clients), 2)
            release.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual((self.root / "voice.mp3").read_bytes(), b"ID3audio")
        self.assertEqual((self.root / "images/01.png").read_bytes(), png_bytes())
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_failed_generator_does_not_cancel_other(self):
        for failed_host in ("api.fish.audio", "api.openai.com"):
            with self.subTest(failed_host=failed_host):
                (self.root / "voice.mp3").unlink(missing_ok=True)
                (self.root / "images/01.png").unlink(missing_ok=True)

                def handle(request):
                    if request.url.host == failed_host:
                        return httpx.Response(401, json={"error": {"message": "secret-body"}})
                    return httpx.Response(200, content=b"ID3audio") if request.url.host == "api.fish.audio" else image_response()

                self.mock_api(handle)
                with self.assertRaisesRegex(ValueError, "HTTP 401") as caught:
                    await run_generation(self.root, GenerationConfig())
                self.assertNotIn("secret-body", str(caught.exception))
                self.assertEqual((self.root / "voice.mp3").exists(), failed_host != "api.fish.audio")
                self.assertEqual((self.root / "images/01.png").exists(), failed_host != "api.openai.com")

    async def test_configuration_failure_still_allows_other_generator(self):
        self.mock_api(lambda request: image_response())
        os.environ["FISH_API_KEY"] = ""
        with self.assertRaisesRegex(ValueError, "FISH_API_KEY"):
            await run_generation(self.root, GenerationConfig())
        self.assertEqual((self.root / "images/01.png").read_bytes(), png_bytes())

    async def test_errors_from_both_generators_are_reported(self):
        self.mock_api(lambda request: httpx.Response(401, json={"error": {"message": "secret-body"}}))
        with self.assertRaises(ValueError) as caught:
            await run_generation(self.root, GenerationConfig())
        message = str(caught.exception)
        self.assertIn("Fish Audio", message)
        self.assertIn("OpenAI Images", message)
        self.assertEqual(message.count("HTTP 401"), 2)
        self.assertNotIn("secret-body", message)

    async def test_all_enabled_inputs_checked_before_any_client(self):
        (self.root / "story.txt").unlink()
        (self.root / "prompts.txt").write_text("[01]\nFirst.\n[01]\nDuplicate.", encoding="utf-8")
        with patch("shorts_ai.generation.create_client", side_effect=AssertionError("no HTTP before preflight")):
            with self.assertRaises(ValueError) as caught:
                await run_generation(self.root, GenerationConfig())
        self.assertIn("story.txt", str(caught.exception))
        self.assertIn("ожидался маркер", str(caught.exception))
        self.assertFalse((self.root / "voice.mp3").exists())

    async def test_cancelling_combined_generation_drains_both_providers(self):
        (self.root / "prompts.txt").write_text("\n".join(f"[{n:02d}]\n{n}" for n in range(1, 11)), encoding="utf-8")
        ready = asyncio.Event()
        streams = []

        def handle(request):
            stream = BodyStream([b"data"], delay=60)
            streams.append(stream)
            if len(streams) == 4: ready.set()
            return httpx.Response(200, stream=stream)

        self.mock_api(handle)
        task = asyncio.create_task(run_generation(self.root, GenerationConfig()))
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(streams), 4)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertTrue(all(client.is_closed for client in self.clients))
        self.assertFalse((self.root / "voice.mp3").exists())
        self.assertEqual(list((self.root / "images").iterdir()), [])


class GenerationCLITests(unittest.TestCase):
    def test_all_four_flag_combinations_only_require_enabled_files_and_keys(self):
        for voice in (True, False):
            for images in (True, False):
                with self.subTest(voice=voice, images=images), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    env = {"GENERATE_VOICE": str(voice), "GENERATE_IMAGES": str(images)}
                    if voice:
                        (root / "story.txt").write_text("Тест.", encoding="utf-8")
                        env["FISH_API_KEY"] = "test-fish"
                    if images:
                        (root / "prompts.txt").write_text("[01]\nScene.", encoding="utf-8")
                        env["OPENAI_API_KEY"] = "test-openai"
                    calls, clients = [], []

                    def handle(request):
                        calls.append(request.url.host)
                        return httpx.Response(200, content=b"ID3audio") if request.url.host == "api.fish.audio" else image_response()

                    def client():
                        result = create_client(transport=httpx.MockTransport(handle))
                        clients.append(result)
                        return result

                    stdout, stderr = io.StringIO(), io.StringIO()
                    with patch.dict(os.environ, env, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                        with patch("shorts_ai.generation.create_client", side_effect=client):
                            code = main(["--project-dir", str(root), "generate"])
                    self.assertEqual((code, stderr.getvalue()), (0, ""))
                    self.assertEqual(calls.count("api.fish.audio"), int(voice))
                    self.assertEqual(calls.count("api.openai.com"), int(images))
                    self.assertEqual(len(clients), int(voice) + int(images))
                    self.assertEqual((root / "voice.mp3").exists(), voice)
                    self.assertEqual((root / "images/01.png").exists(), images)

    def test_images_command_ignores_flags_and_supports_other_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "проект с 'кавычками'"
            project.mkdir()
            prompts = root / "other prompts.txt"
            prompts.write_text("[01]\nScene.", encoding="utf-8")
            env = {"GENERATE_VOICE": "invalid", "GENERATE_IMAGES": "false", "OPENAI_API_KEY": "test-key"}
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, env, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                with patch("shorts_ai.generation.create_client", side_effect=lambda: create_client(transport=httpx.MockTransport(lambda request: image_response()))):
                    code = main(["--project-dir", str(project), "images", "--prompts", str(prompts), "--output-dir", "pictures/output"])
            self.assertEqual((code, stderr.getvalue()), (0, ""))
            self.assertEqual((project / "pictures/output/01.png").read_bytes(), png_bytes())
            self.assertFalse((project / "voice.mp3").exists())

    def test_cli_error_is_nonzero_and_keeps_successful_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts.txt").write_text("[01]\n1\n[02]\n2", encoding="utf-8")

            def handle(request):
                return httpx.Response(401, json={"error": {"message": "secret-provider"}}) if json.loads(request.content)["prompt"] == "1" else image_response(2)

            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), redirect_stdout(stdout), redirect_stderr(stderr):
                with patch("shorts_ai.generation.create_client", side_effect=lambda: create_client(transport=httpx.MockTransport(handle))):
                    code = main(["--project-dir", str(root), "images"])
            self.assertEqual(code, 1)
            self.assertIn("изображение 01", stderr.getvalue())
            self.assertNotIn("secret-provider", stderr.getvalue())
            self.assertFalse((root / "images/01.png").exists())
            self.assertEqual((root / "images/02.png").read_bytes(), png_bytes(2))

    def test_sigint_during_image_retry_cancels_workers_and_queue(self):
        script = """
import sys
import httpx
import shorts_ai.generation as generation
from shorts_ai.cli import main
from shorts_ai.transport import create_client
calls = 0
def handle(request):
    global calls
    calls += 1
    if calls == 3:
        print('THREE_REQUESTS_STARTED', file=sys.stderr, flush=True)
    return httpx.Response(429, headers={'Retry-After': '60'}, json={'error': {'code': 'rate_limit_exceeded'}})
generation.create_client = lambda: create_client(transport=httpx.MockTransport(handle))
sys.exit(main(sys.argv[1:]))
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts.txt").write_text("\n".join(f"[{n:02d}]\n{n}" for n in range(1, 11)), encoding="utf-8")
            env = os.environ.copy()
            env.update(OPENAI_API_KEY="test-key", OPENAI_IMAGE_MODEL="gpt-image-2.5-sunburst", OPENAI_IMAGE_SIZE="1024x1536", OPENAI_IMAGE_QUALITY="low")
            process = subprocess.Popen([sys.executable, "-c", script, "--project-dir", str(root), "images"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stderr, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=5), "image requests did not start")
                    self.assertEqual(process.stderr.readline().strip(), "THREE_REQUESTS_STARTED")
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
            self.assertEqual(list((root / "images").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
