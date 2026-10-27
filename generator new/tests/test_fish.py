import asyncio
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from shorts_ai import fish
from shorts_ai.generation import generate_voice
from shorts_ai.transport import create_client


class FishInputTests(unittest.TestCase):
    def test_shared_go_text_fixtures(self):
        path = Path(__file__).parent / "fixtures/migration/cases.json"
        cases = json.loads(path.read_text(encoding="utf-8"))["texts"]
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "story.txt"
            for case in cases:
                with self.subTest(case=case["name"]):
                    content = bytes.fromhex(case["hex"]) if "hex" in case else case["content"].encode("utf-8")
                    script.write_bytes(content)
                    if "error_contains" in case:
                        with self.assertRaisesRegex(ValueError, case["error_contains"]):
                            fish.read_text(script)
                    else:
                        self.assertEqual(fish.read_text(script), case["expected"])

    def test_config_defaults_custom_values_and_go_models(self):
        defaults = fish.read_config({"FISH_API_KEY": "test-key"})
        self.assertEqual(defaults, {
            "api_key": "test-key", "reference_id": "8336207c3431432cb60e8851e78ba437", "model": "s2.1-pro-free",
        })
        for model in ("s1", "s2-pro", "s2.1-pro", "s2.1-pro-free", "drama-3-preview"):
            with self.subTest(model=model):
                config = fish.read_config({"FISH_API_KEY": "test-key", "FISH_MODEL": model, "FISH_REFERENCE_ID": "custom-voice"})
                self.assertEqual(config["model"], model)
                self.assertEqual(config["reference_id"], "custom-voice")
        self.assertEqual(fish.read_config({"FISH_API_KEY": "test-key", "FISH_MODEL": "", "FISH_REFERENCE_ID": ""}), defaults)

    def test_bad_config_does_not_expose_values(self):
        for env in ({}, {"FISH_API_KEY": "  "}, {"FISH_API_KEY": "secret\nvalue"},
                    {"FISH_API_KEY": "secret-ключ"}, {"FISH_API_KEY": "secret-key", "FISH_MODEL": "secret-model"}):
            with self.subTest(env_names=list(env)), self.assertRaises(ValueError) as caught:
                fish.read_config(env)
            self.assertNotIn("secret", str(caught.exception))


class AudioStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, delay=0, error=None, started=None):
        self.chunks = chunks
        self.delay = delay
        self.error = error
        self.started = started
        self.closed = False
        self.read_count = 0

    async def __aiter__(self):
        if self.started is not None:
            self.started.set()
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.read_count += 1
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


class FishGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "озвучка с 'кавычками'"
        self.root.mkdir()
        self.script = self.root / "story.txt"
        self.output = self.root / "voice.mp3"
        self.text = "  Привет!\r\nОбычный текст без маркеров.\r\n"
        self.script.write_bytes(("\ufeff" + self.text).encode("utf-8"))
        environment = patch.dict(os.environ, {"FISH_API_KEY": "test-key", "FISH_REFERENCE_ID": "test-voice"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        capture = redirect_stdout(io.StringIO())
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def mock_api(self, handle):
        calls = []

        def request(request):
            calls.append(request)
            return handle(request)

        factory = patch("shorts_ai.generation.create_client", side_effect=lambda: create_client(transport=httpx.MockTransport(request)))
        factory.start()
        self.addCleanup(factory.stop)
        return calls

    def assert_no_result(self):
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".shorts-ai-*")), [])

    async def test_request_saves_raw_audio_then_reuses_it_without_key(self):
        audio = b"ID3\xff\x00raw-mp3"
        calls = self.mock_api(lambda request: httpx.Response(200, headers={"Content-Type": "audio/mpeg"}, content=audio))
        self.assertTrue(await generate_voice(self.script, self.output))
        self.assertEqual(self.output.read_bytes(), audio)
        self.assertEqual(len(calls), 1)
        request = calls[0]
        self.assertEqual((request.method, str(request.url)), ("POST", "https://api.fish.audio/v1/tts"))
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        self.assertEqual(request.headers["model"], "s2.1-pro-free")
        self.assertEqual(request.headers["Content-Type"], "application/json")
        self.assertEqual(json.loads(request.content), {"text": self.text, "reference_id": "test-voice", "format": "mp3"})
        self.script.write_text("Изменённый текст.", encoding="utf-8")
        os.environ.pop("FISH_API_KEY")
        with patch("shorts_ai.generation.create_client", side_effect=AssertionError("cached audio must not create a client")):
            self.assertFalse(await generate_voice(self.script, self.output))
        self.assertEqual(self.output.read_bytes(), audio)
        self.assertEqual(len(calls), 1)

    async def test_invalid_inputs_and_config_never_create_a_client(self):
        for kind in ("missing script", "empty text", "bad utf8", "missing key", "unknown model", "empty output", "output directory"):
            with self.subTest(kind=kind):
                self.script.write_text("Тест.", encoding="utf-8")
                os.environ.update(FISH_API_KEY="test-key", FISH_MODEL="s2.1-pro-free")
                if self.output.is_dir():
                    self.output.rmdir()
                else:
                    self.output.unlink(missing_ok=True)
                if kind == "missing script": self.script.unlink()
                if kind == "empty text": self.script.write_text("\ufeff \n\t", encoding="utf-8")
                if kind == "bad utf8": self.script.write_bytes(b"\xff")
                if kind == "missing key": os.environ["FISH_API_KEY"] = ""
                if kind == "unknown model": os.environ["FISH_MODEL"] = "secret-invalid-model"
                if kind == "empty output": self.output.write_bytes(b"")
                if kind == "output directory": self.output.mkdir()
                with patch("shorts_ai.generation.create_client", side_effect=AssertionError("preflight must finish before HTTP")):
                    with self.assertRaises((OSError, ValueError)) as caught:
                        await generate_voice(self.script, self.output)
                self.assertNotIn("secret-invalid-model", str(caught.exception))

    async def test_cached_audio_still_requires_a_valid_script(self):
        self.output.write_bytes(b"ID3cached")
        self.script.write_text(" \n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "пустым"):
            await generate_voice(self.script, self.output)
        self.assertEqual(self.output.read_bytes(), b"ID3cached")

    async def test_http_failures_are_not_retried_or_saved(self):
        for status in (401, 402, 429, 500, 503, 599):
            with self.subTest(status=status):
                calls = self.mock_api(lambda request: httpx.Response(status, content=b"secret-provider-message"))
                with self.assertRaisesRegex(ValueError, f"HTTP {status}") as caught:
                    await generate_voice(self.script, self.output)
                self.assertNotIn("secret-provider-message", str(caught.exception))
                self.assertEqual(len(calls), 1)
                self.assert_no_result()

    async def test_error_response_body_is_not_consumed(self):
        stream = AudioStream([], error=AssertionError("error body must not be read"))
        calls = self.mock_api(lambda request: httpx.Response(401, stream=stream))
        with self.assertRaisesRegex(ValueError, "HTTP 401"):
            await generate_voice(self.script, self.output)
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

    async def test_empty_or_textual_response_is_not_saved(self):
        for content_type, body in (("audio/mpeg", b""), ("application/json", b"secret-body"),
                                   ("text/plain", b"secret-body"), ("Application/Problem+JSON", b"secret-body")):
            with self.subTest(content_type=content_type):
                calls = self.mock_api(lambda request: httpx.Response(200, headers={"Content-Type": content_type}, content=body))
                with self.assertRaises(ValueError) as caught:
                    await generate_voice(self.script, self.output)
                self.assertNotIn("secret-body", str(caught.exception))
                self.assertEqual(len(calls), 1)
                self.assert_no_result()

    async def test_32_mib_boundary_and_oversized_response(self):
        chunk = b"x" * (16 * 1024 * 1024)
        stream = AudioStream([chunk, chunk, b"x"])
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaisesRegex(ValueError, "превышает"):
            await generate_voice(self.script, self.output)
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()
        stream = AudioStream([chunk, chunk])
        async with create_client(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))) as client:
            audio = await fish.request_audio(client, self.text, fish.read_config())
        self.assertEqual(len(audio), 32 * 1024 * 1024)

    async def test_interrupted_response_is_closed_without_partial_file(self):
        stream = AudioStream([b"ID3partial"], error=httpx.ReadError("secret-error"))
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaisesRegex(ValueError, "HTTP-запрос") as caught:
            await generate_voice(self.script, self.output)
        self.assertNotIn("secret-error", str(caught.exception))
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

    async def test_timeout_is_not_retried_or_saved(self):
        stream = AudioStream([b"ID3", b"audio"], delay=0.03)
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaisesRegex(ValueError, "Истекло время"):
            await generate_voice(self.script, self.output, timeout=0.05)
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

    async def test_cancellation_closes_response_without_saving(self):
        started = asyncio.Event()
        stream = AudioStream([b"audio"], delay=60, started=started)
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        task = asyncio.create_task(generate_voice(self.script, self.output))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

    async def test_cancellation_before_publication_is_respected(self):
        class CancelOnClose(AudioStream):
            async def aclose(self):
                await super().aclose()
                asyncio.current_task().cancel()

        stream = CancelOnClose([b"ID3audio"])
        self.mock_api(lambda request: httpx.Response(200, stream=stream))
        task = asyncio.create_task(generate_voice(self.script, self.output))
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_no_result()

    async def test_output_created_during_request_is_preserved(self):
        def handle(request):
            self.output.write_bytes(b"another process")
            return httpx.Response(200, content=b"ID3our-audio")

        calls = self.mock_api(handle)
        with self.assertRaises(FileExistsError):
            await generate_voice(self.script, self.output)
        self.assertEqual(self.output.read_bytes(), b"another process")
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(self.root.glob(".shorts-ai-*")), [])

    async def test_redirect_is_not_followed(self):
        calls = self.mock_api(lambda request: httpx.Response(307, headers={"Location": "https://other.test/voice"}))
        with self.assertRaisesRegex(ValueError, "HTTP 307"):
            await generate_voice(self.script, self.output)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()


if __name__ == "__main__":
    unittest.main()
