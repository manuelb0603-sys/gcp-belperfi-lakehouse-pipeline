"""Synchronous stdio MCP client for the local finance tool server (``finance/mcp_server.py``).

``McpToolClient(project_name).call_tool(name, arguments)`` returns the tool's JSON result as a
dict and raises ``ValueError`` with the server's message for a tool error (the harness turns it
into tool-error feedback, as for an in-process call). The server process starts on the first call
(so a run without data-tool calls never launches it) and is reused until ``close()``. The MCP SDK
is async: one background thread runs an event loop holding the stdio session, and calls are
queued to it, so the session opens and closes in the same task as anyio requires.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
from typing import Any

from .log import get_logger
from .settings import PROJECT_ROOT

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 120.0  # seconds per tool call (BigQuery queries) and for server start-up


def server_parameters(project_name: str) -> Any:
    """Launch ``python -m finance.mcp_server`` with this interpreter from ``Scripts/``."""
    from mcp import StdioServerParameters
    from .mcp_server import PROJECT_ENV

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "finance.mcp_server"],
        cwd=str(PROJECT_ROOT / "Scripts"),
        # The full environment keeps Google credentials (APPDATA, GOOGLE_APPLICATION_CREDENTIALS) visible.
        env={**os.environ, PROJECT_ENV: project_name},
    )


class McpToolClient:
    """One stdio session with the finance MCP server, used synchronously."""

    def __init__(self, project_name: str, *, timeout: float = DEFAULT_TIMEOUT, server_params: Any = None):
        self.project_name = project_name
        self.timeout = float(timeout)
        self._server_params = server_params
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._requests: asyncio.Queue | None = None

    @property
    def started(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ----- lifecycle -----

    def _start(self) -> None:
        ready: concurrent.futures.Future = concurrent.futures.Future()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="finance-mcp", daemon=True)
        self._thread.start()
        try:
            ready.result(timeout=self.timeout)
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Could not start the finance MCP server: {type(exc).__name__}: {exc}") from exc

    def _run(self, ready: concurrent.futures.Future) -> None:
        loop = self._loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve(ready))
        finally:
            loop.close()

    async def _serve(self, ready: concurrent.futures.Future) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        params = self._server_params or server_parameters(self.project_name)
        self._requests = asyncio.Queue()
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    ready.set_result(None)
                    while True:
                        item = await self._requests.get()
                        if item is None:
                            return
                        name, arguments, future = item
                        try:
                            result = await session.call_tool(name, arguments, read_timeout_seconds=self.timeout)
                        except Exception as exc:
                            if not future.done():
                                future.set_exception(exc)
                        else:
                            if not future.done():
                                future.set_result(result)
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning(f"Finance MCP session ended: {type(exc).__name__}: {exc}")

    def close(self) -> None:
        """Stop the session and the server process (safe to call more than once)."""
        loop, thread = self._loop, self._thread
        if loop is not None and thread is not None and thread.is_alive() and self._requests is not None:
            loop.call_soon_threadsafe(self._requests.put_nowait, None)
            thread.join(timeout=10)
        self._loop = self._thread = self._requests = None

    def __enter__(self) -> "McpToolClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ----- calls -----

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self.started:
                self._start()
            loop, requests = self._loop, self._requests
        future: concurrent.futures.Future = concurrent.futures.Future()
        loop.call_soon_threadsafe(requests.put_nowait, (name, dict(arguments), future))
        try:
            result = future.result(timeout=self.timeout + 5)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{name} did not answer within {self.timeout:g} seconds") from None
        text = "".join(getattr(block, "text", "") for block in getattr(result, "content", None) or [])
        if getattr(result, "is_error", False):
            raise ValueError(text or f"{name} failed")
        return json.loads(text)
