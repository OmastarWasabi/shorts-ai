import asyncio
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from shorts_ai.transport import create_client, request_limited


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0, started=None):
        self.chunks = chunks
        self.delay = delay
        self.started = started
        self.read_count = 0
        self.closed = False

    async def __aiter__(self):
        if self.started is not None:
            self.started.set()
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.read_count += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_and_no_automatic_retry(self):
        calls = []

        async def handle(request):
            calls.append(request)
            return httpx.Response(429, headers={"Retry-After": "60"}, content=b"rate limited")

        async with create_client(transport=httpx.MockTransport(handle)) as client:
            result = await request_limited(client, "POST", "https://example.test/api", max_bytes=100, json={"text": "Тест"})
        self.assertEqual(result.status_code, 429)
        self.assertEqual(result.headers["Retry-After"], "60")
        self.assertEqual(result.body, b"rate limited")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].method, "POST")

    async def test_redirect_is_returned_without_sending_another_request(self):
        calls = []

        async def handle(request):
            calls.append(request.url.host)
            return httpx.Response(307, headers={"Location": "https://other.test/"})

        async with create_client(transport=httpx.MockTransport(handle)) as client:
            result = await request_limited(client, "POST", "https://example.test/", max_bytes=100, headers={"Authorization": "Bearer test-only"})
        self.assertEqual(result.status_code, 307)
        self.assertEqual(calls, ["example.test"])

    async def test_response_limit_stops_reading_and_closes_stream(self):
        stream = TrackedStream([b"12", b"345", b"must not be read"])
        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
        async with create_client(transport=transport) as client:
            with self.assertRaisesRegex(ValueError, "превышает"):
                await request_limited(client, "GET", "https://example.test/", max_bytes=4)
        self.assertEqual(stream.read_count, 2)
        self.assertTrue(stream.closed)

    async def test_total_timeout_includes_reading_body(self):
        stream = TrackedStream([b"a", b"b", b"c"], delay=0.03)
        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
        async with create_client(transport=transport) as client:
            with self.assertRaisesRegex(ValueError, "Истекло время"):
                await request_limited(client, "GET", "https://example.test/", max_bytes=10, timeout=0.05)
        self.assertLess(stream.read_count, 3)
        self.assertTrue(stream.closed)

    async def test_cancellation_propagates_and_closes_response(self):
        started = asyncio.Event()
        stream = TrackedStream([b"data"], delay=60, started=started)
        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
        async with create_client(transport=transport) as client:
            task = asyncio.create_task(request_limited(client, "GET", "https://example.test/", max_bytes=100))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(stream.closed)

    async def test_compressed_response_is_decoded_once_and_limited(self):
        raw = b"decoded body" * 10
        transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=gzip.compress(raw)))
        async with create_client(transport=transport) as client:
            result = await request_limited(client, "GET", "https://example.test/", max_bytes=len(raw))
            self.assertEqual(result.body, raw)
            with self.assertRaisesRegex(ValueError, "превышает"):
                await request_limited(client, "GET", "https://example.test/", max_bytes=10)

    async def test_connection_error_does_not_expose_request_data(self):
        def handle(request):
            raise httpx.ConnectError("secret-value in https://private.test/?key=secret-value", request=request)

        async with create_client(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaisesRegex(ValueError, "Не удалось выполнить") as caught:
                await request_limited(client, "GET", "https://private.test/?key=secret-value", max_bytes=10)
        self.assertNotIn("secret-value", str(caught.exception))
        self.assertNotIn("private.test", str(caught.exception))

    async def test_tls_rejects_untrusted_certificate_and_accepts_explicit_ca(self):
        openssl = shutil.which("openssl")
        if openssl is None:
            self.skipTest("openssl is needed for a local TLS verification test")
        with tempfile.TemporaryDirectory() as temporary:
            key, cert = Path(temporary) / "key.pem", Path(temporary) / "cert.pem"
            subprocess.run(
                [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                 "-days", "1", "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15,
            )

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"trusted local server")

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            server.socket = context.wrap_socket(server.socket, server_side=True)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
            thread.start()
            url = f"https://127.0.0.1:{server.server_port}/"
            try:
                with patch.dict(os.environ, {"NO_PROXY": "*"}, clear=True):
                    async with create_client() as client:
                        with self.assertRaisesRegex(ValueError, "Не удалось выполнить"):
                            await request_limited(client, "GET", url, max_bytes=100, timeout=2)
                with patch.dict(os.environ, {"SSL_CERT_FILE": str(cert), "NO_PROXY": "*"}, clear=True):
                    async with create_client() as client:
                        result = await request_limited(client, "GET", url, max_bytes=100, timeout=2)
                        self.assertEqual(result.status_code, 200)
                        self.assertEqual(result.body, b"trusted local server")
            finally:
                await asyncio.to_thread(server.shutdown)
                server.server_close()
                thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
