import asyncio
import base64
from contextlib import redirect_stdout
from email.utils import formatdate
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from shorts_ai import images
from shorts_ai.generation import generate_images
from shorts_ai.transport import create_client


def png_bytes(number=1, size=(2, 2), format="PNG"):
    output = io.BytesIO()
    with Image.new("RGB", size, (number, 0, 0)) as image:
        image.save(output, format=format)
    return output.getvalue()


def image_response(number=1):
    return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png_bytes(number)).decode("ascii")}]})


class BodyStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, delay=0, started=None, error=None):
        self.chunks = chunks
        self.delay = delay
        self.started = started
        self.error = error
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


class ImageDataTests(unittest.TestCase):
    def test_defaults_and_custom_model(self):
        config = images.read_config({"OPENAI_API_KEY": "test-key"})
        self.assertEqual(config, {"api_key": "test-key", "model": "gpt-image-2.5-sunburst", "size": "1024x1536", "quality": "low"})
        custom = images.read_config({"OPENAI_API_KEY": "test-key", "OPENAI_IMAGE_MODEL": "another-model", "OPENAI_IMAGE_SIZE": "1536x1024", "OPENAI_IMAGE_QUALITY": "high"})
        self.assertEqual((custom["model"], custom["size"], custom["quality"]), ("another-model", "1536x1024", "high"))
        self.assertEqual(images.read_config({"OPENAI_API_KEY": "test-key", "OPENAI_IMAGE_MODEL": "", "OPENAI_IMAGE_SIZE": "", "OPENAI_IMAGE_QUALITY": ""}), config)

    def test_invalid_config_does_not_expose_values(self):
        for fields in ({"OPENAI_API_KEY": ""}, {"OPENAI_API_KEY": "secret\nkey"}, {"OPENAI_API_KEY": "secret-ключ"},
                       {"OPENAI_IMAGE_SIZE": "secret-size"}, {"OPENAI_IMAGE_QUALITY": "secret-quality"}):
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError) as caught:
                images.read_config({"OPENAI_API_KEY": "test-key", **fields})
            self.assertNotIn("secret", str(caught.exception))

    def test_complete_png_validation(self):
        images.validate_png(png_bytes())
        images.validate_png(png_bytes(size=(4096, 1)))
        for data in (b"", b"not PNG", png_bytes(format="JPEG"), png_bytes()[:-12], png_bytes(size=(4097, 1))):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                images.validate_png(data)
        for missing in range(1, 13):
            with self.subTest(missing_tail_bytes=missing), self.assertRaises(ValueError):
                images.validate_png(png_bytes()[:-missing])
        corrupt = bytearray(png_bytes())
        index = corrupt.index(b"IDAT") + 4
        corrupt[index] ^= 1
        with self.assertRaises(ValueError):
            images.validate_png(bytes(corrupt))
        corrupt = bytearray(png_bytes())
        corrupt[-1] ^= 1
        with self.assertRaises(ValueError):
            images.validate_png(bytes(corrupt))
        images.validate_png(png_bytes() + b"trailing bytes accepted by Go")

    def test_retry_after_seconds_date_and_fallback(self):
        with patch("shorts_ai.images.random.randrange", return_value=7):
            self.assertEqual(images.retry_delay("10", 0), 10)
            self.assertEqual(images.retry_delay(formatdate(1010, usegmt=True), 0, now=1000), 10)
            self.assertAlmostEqual(images.retry_delay("invalid", 1), 2.007)
            self.assertAlmostEqual(images.retry_delay("0", 0), 1.007)
            self.assertAlmostEqual(images.retry_delay("9" * 100, 0), 1.007)


class ImageGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "картинки с 'кавычками'"
        self.root.mkdir()
        self.prompts = self.root / "prompts.txt"
        self.output = self.root / "images"
        self.write_prompts(1)
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        stdout = redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)
        self.clients = []

    def write_prompts(self, count):
        self.prompts.write_text("\n".join(f"[{number:02d}]\n{number}" for number in range(1, count + 1)), encoding="utf-8")

    def mock_api(self, handle):
        calls = []

        def call(request):
            calls.append(request)
            return handle(request)

        def client():
            result = create_client(transport=httpx.MockTransport(call))
            self.clients.append(result)
            return result

        mock = patch("shorts_ai.generation.create_client", side_effect=client)
        mock.start()
        self.addCleanup(mock.stop)
        return calls

    def assert_no_result(self):
        self.assertFalse((self.output / "01.png").exists())
        if self.output.exists():
            self.assertEqual(list(self.output.glob(".shorts-ai-*")), [])
        self.assertTrue(all(client.is_closed for client in self.clients))

    async def test_ten_prompts_three_at_a_time_names_and_resume(self):
        self.write_prompts(10)
        release, three_started = asyncio.Event(), asyncio.Event()
        active = peak = 0

        async def handle(request):
            nonlocal active, peak
            body = json.loads(request.content)
            number = int(body["prompt"])
            active += 1
            peak = max(peak, active)
            if active == 3:
                three_started.set()
            try:
                await release.wait()
                await asyncio.sleep(0.001 * (number % 3))
                return image_response(number)
            finally:
                active -= 1

        calls = self.mock_api(handle)
        task = asyncio.create_task(generate_images(self.prompts, self.output))
        try:
            await asyncio.wait_for(three_started.wait(), timeout=2)
            self.assertEqual(len(calls), 3)
            release.set()
            self.assertEqual(await asyncio.wait_for(task, timeout=5), 10)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(peak, 3)
        self.assertEqual(len(calls), 10)
        for number in range(1, 11):
            self.assertEqual((self.output / f"{number:02d}.png").read_bytes(), png_bytes(number))
        request = calls[0]
        self.assertEqual((request.method, str(request.url)), ("POST", images.IMAGES_URL))
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        body = json.loads(request.content)
        self.assertEqual({key: value for key, value in body.items() if key != "prompt"}, {
            "model": "gpt-image-2.5-sunburst", "size": "1024x1536", "quality": "low", "n": 1, "output_format": "png",
        })
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.assertEqual(await generate_images(self.prompts, self.output), 0)
        self.assertEqual(len(calls), 10)
        (self.output / "04.png").unlink()
        self.assertEqual(await generate_images(self.prompts, self.output), 1)
        self.assertEqual(len(calls), 11)
        self.assertEqual(json.loads(calls[-1].content)["prompt"], "4")
        (self.output / "01.png").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "01.png"):
            await generate_images(self.prompts, self.output)
        self.assertEqual(len(calls), 11)

    async def test_preflight_rejects_bad_inputs_settings_and_existing_files(self):
        for kind in ("prompts", "key", "size", "quality", "broken png", "large png", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                prompts, output = root / "prompts.txt", root / "images"
                prompts.write_text("[01]\nScene.", encoding="utf-8")
                env = {"OPENAI_API_KEY": "test-key"}
                if kind == "prompts": prompts.write_text("[01]\nFirst.\n[01]\nDuplicate.", encoding="utf-8")
                if kind == "key": env["OPENAI_API_KEY"] = ""
                if kind == "size": env["OPENAI_IMAGE_SIZE"] = "bad"
                if kind == "quality": env["OPENAI_IMAGE_QUALITY"] = "bad"
                if kind in ("broken png", "large png", "directory"):
                    output.mkdir()
                    if kind == "directory":
                        (output / "01.png").mkdir()
                    else:
                        with (output / "01.png").open("wb") as file:
                            file.write(b"broken")
                            if kind == "large png": file.truncate(32 * 1024 * 1024 + 1)
                with patch.dict(os.environ, env, clear=True), patch("shorts_ai.generation.create_client", side_effect=AssertionError("no client before preflight")):
                    with self.assertRaises((OSError, ValueError)):
                        await generate_images(prompts, output)

    async def test_errors_do_not_retry_or_save(self):
        cases = [
            (401, {"error": {"code": "invalid_api_key", "message": "secret-provider-message"}}),
            (429, {"error": {"code": "insufficient_quota"}}),
            (429, {"error": {"type": "insufficient_quota"}}),
            (429, {"error": {"code": "billing_hard_limit_reached"}}),
            (429, {"error": {"type": "billing_error"}}),
            (500, {}), (200, []), (200, {"data": []}), (200, {"data": [None]}),
            (200, {"data": [{"b64_json": 123}]}), (200, {"data": [{"b64_json": "!"}]}),
            (200, {"data": [{"b64_json": "bm90IGEgcG5n"}]}),
        ]
        for status, payload in cases:
            with self.subTest(status=status, payload=payload):
                calls = self.mock_api(lambda request: httpx.Response(status, json=payload))
                with self.assertRaises(ValueError) as caught:
                    await generate_images(self.prompts, self.output)
                self.assertNotIn("secret-provider-message", str(caught.exception))
                self.assertEqual(len(calls), 1)
                self.assert_no_result()

    async def test_bad_json_and_oversized_response(self):
        for body in (b"not JSON secret-body", b"\xff", b'{"data": [], "bad": NaN}'):
            with self.subTest(body=body):
                calls = self.mock_api(lambda request: httpx.Response(200, content=body))
                with self.assertRaisesRegex(ValueError, "JSON") as caught:
                    await generate_images(self.prompts, self.output)
                self.assertNotIn("secret-body", str(caught.exception))
                self.assertEqual(len(calls), 1)
                self.assert_no_result()
        stream = BodyStream([b"x" * (16 * 1024 * 1024)] * 3 + [b"x"])
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaisesRegex(ValueError, "превышает"):
            await generate_images(self.prompts, self.output)
        self.assertTrue(stream.closed)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

    async def test_transient_429_recovers_and_exhausts_after_three_attempts(self):
        for recover in (True, False):
            with self.subTest(recover=recover):
                (self.output / "01.png").unlink(missing_ok=True)
                count = 0

                def handle(request):
                    nonlocal count
                    count += 1
                    if recover and count == 2:
                        return image_response()
                    return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": {"code": "rate_limit_exceeded"}})

                calls = self.mock_api(handle)
                with patch("shorts_ai.images.retry_delay", return_value=0):
                    if recover:
                        self.assertEqual(await generate_images(self.prompts, self.output), 1)
                    else:
                        with self.assertRaisesRegex(ValueError, "HTTP 429"):
                            await generate_images(self.prompts, self.output)
                self.assertEqual(len(calls), 2 if recover else 3)

    async def test_retry_wait_holds_slots_and_cancel_stops_queue(self):
        self.write_prompts(10)
        started = asyncio.Event()
        count = 0

        def handle(request):
            nonlocal count
            count += 1
            if count == 3: started.set()
            return httpx.Response(429, headers={"Retry-After": "60"}, json={"error": {"code": "rate_limit_exceeded"}})

        calls = self.mock_api(handle)
        task = asyncio.create_task(generate_images(self.prompts, self.output))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            await asyncio.sleep(0)
            self.assertEqual(len(calls), 3)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(list(self.output.iterdir()), [])
        self.assert_no_result()

    async def test_cancel_active_requests_closes_every_stream_and_client(self):
        self.write_prompts(10)
        ready = asyncio.Event()
        streams = []

        def handle(request):
            stream = BodyStream([b"data"], delay=60)
            streams.append(stream)
            if len(streams) == 3: ready.set()
            return httpx.Response(200, stream=stream)

        self.mock_api(handle)
        task = asyncio.create_task(generate_images(self.prompts, self.output))
        try:
            await asyncio.wait_for(ready.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(len(streams), 3)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assert_no_result()

    async def test_request_timeout_network_error_and_total_retry_budget(self):
        stream = BodyStream([b"part", b"more"], delay=0.03)
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaisesRegex(ValueError, "Истекло время"):
            await generate_images(self.prompts, self.output, request_timeout=0.05)
        self.assertEqual(len(calls), 1)
        self.assertTrue(stream.closed)
        self.assert_no_result()
        stream = BodyStream([b"part"], error=httpx.ReadError("secret-network-message"))
        calls = self.mock_api(lambda request: httpx.Response(200, stream=stream))
        with self.assertRaises(ValueError) as caught:
            await generate_images(self.prompts, self.output)
        self.assertNotIn("secret-network-message", str(caught.exception))
        self.assertEqual(len(calls), 1)
        self.assert_no_result()
        calls = self.mock_api(lambda request: httpx.Response(429, headers={"Retry-After": "60"}, json={"error": {"code": "rate_limit_exceeded"}}))
        with self.assertRaisesRegex(ValueError, "время генерации"):
            await generate_images(self.prompts, self.output, total_timeout=0.03)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

    async def test_error_body_is_bounded_but_quota_still_is_not_retried(self):
        body = b'{"error":{"code":"insufficient_quota"}}'
        stream = BodyStream([body, b"x" * (64 * 1024), b"must not read"])
        calls = self.mock_api(lambda request: httpx.Response(429, stream=stream))
        with self.assertRaisesRegex(ValueError, "HTTP 429"):
            await generate_images(self.prompts, self.output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stream.read_count, 2)
        self.assertTrue(stream.closed)
        self.assert_no_result()

    async def test_failed_image_does_not_cancel_others_and_resume_only_missing(self):
        self.write_prompts(3)

        def handle(request):
            number = int(json.loads(request.content)["prompt"])
            return httpx.Response(401) if number == 1 else image_response(number)

        calls = self.mock_api(handle)
        with self.assertRaisesRegex(ValueError, "изображение 01.*HTTP 401"):
            await generate_images(self.prompts, self.output)
        self.assertEqual(len(calls), 3)
        self.assertFalse((self.output / "01.png").exists())
        for number in (2, 3):
            self.assertEqual((self.output / f"{number:02d}.png").read_bytes(), png_bytes(number))
        calls = self.mock_api(lambda request: image_response(int(json.loads(request.content)["prompt"])))
        self.assertEqual(await generate_images(self.prompts, self.output), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0].content)["prompt"], "1")

    async def test_redirect_is_not_followed_and_competing_output_is_preserved(self):
        calls = self.mock_api(lambda request: httpx.Response(307, headers={"Location": "https://other.test/"}))
        with self.assertRaisesRegex(ValueError, "HTTP 307"):
            await generate_images(self.prompts, self.output)
        self.assertEqual(len(calls), 1)
        self.assert_no_result()

        def handle(request):
            (self.output / "01.png").write_bytes(png_bytes(9))
            return image_response(1)

        self.mock_api(handle)
        with self.assertRaises(ValueError):
            await generate_images(self.prompts, self.output)
        self.assertEqual((self.output / "01.png").read_bytes(), png_bytes(9))
        self.assertEqual(list(self.output.glob(".shorts-ai-*")), [])


if __name__ == "__main__":
    unittest.main()
