"""Основа HTTP для провайдеров: TLS, таймаут, ограниченное чтение ответа."""

import asyncio
from dataclasses import dataclass
import math
from typing import Any

import httpx


REQUEST_TIMEOUT = 180.0


@dataclass(frozen=True)
class HTTPResponse:
    status_code: int
    headers: httpx.Headers
    body: bytes


def create_client(*, transport: httpx.AsyncBaseTransport | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=False,
        verify=True,
        transport=transport,
    )


async def request_limited(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    max_bytes: int,
    timeout: float = REQUEST_TIMEOUT,
    read_error_body: bool = True,
    error_max_bytes: int | None = None,
    **kwargs: Any,
) -> HTTPResponse:
    """Общий таймаут включает соединение и чтение тела. Повторов здесь нет."""
    if max_bytes < 1 or not math.isfinite(timeout) or timeout <= 0 or (error_max_bytes is not None and error_max_bytes < 1):
        raise ValueError("Нужны положительные ограничение ответа и таймаут")
    try:
        async with asyncio.timeout(timeout):
            async with client.stream(method, url, follow_redirects=False, **kwargs) as response:
                if response.status_code != 200 and not read_error_body:
                    return HTTPResponse(response.status_code, response.headers, b"")
                body = bytearray()
                error_limit = error_max_bytes if response.status_code != 200 else None
                async for chunk in response.aiter_bytes():
                    if error_limit is not None:
                        body.extend(chunk[:error_limit - len(body)])
                        if len(body) == error_limit:
                            break
                        continue
                    if len(body) + len(chunk) > max_bytes:
                        raise ValueError("HTTP-ответ превышает допустимый размер")
                    body.extend(chunk)
                return HTTPResponse(response.status_code, response.headers, bytes(body))
    except (TimeoutError, httpx.TimeoutException):
        raise ValueError("Истекло время HTTP-запроса") from None
    except httpx.HTTPError:
        # Текст исключения HTTPX может содержать URL и другие данные запроса.
        raise ValueError("Не удалось выполнить HTTP-запрос") from None
