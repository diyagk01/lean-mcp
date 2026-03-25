from __future__ import annotations

import json
import os
import re
import select
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

from .types import McpTool


class McpClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class McpCallResult:
    raw: Dict[str, Any]
    text: str
    is_error: bool


class McpStdioClient:
    """
    Minimal MCP client over stdio JSON-RPC.
    """

    def __init__(
        self,
        *,
        command: str,
        args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.env = env
        self.timeout_seconds = timeout_seconds
        self._proc: Optional[subprocess.Popen[str]] = None
        self._next_id = 1
        self._stderr_lines: Deque[str] = deque(maxlen=200)
        self._stderr_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._proc is not None:
            return
        merged_env = os.environ.copy()
        if self.env:
            merged_env.update(self.env)
        self._proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=merged_env,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()
        self._initialize_protocol()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    def list_tools(self) -> List[McpTool]:
        result = self._send_request("tools/list", {})
        raw_tools = result.get("tools", [])
        tools: List[McpTool] = []
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            description = str(raw.get("description", "")).strip() or f"{name} tool"
            input_schema = (
                raw.get("inputSchema")
                or raw.get("input_schema")
                or raw.get("arguments")
                or {"type": "object", "properties": {}}
            )
            tools.append(
                McpTool(
                    name=name,
                    description=description,
                    input_schema=dict(input_schema),  # type: ignore[arg-type]
                    server=self.command,
                    metadata={"source": "mcp_live"},
                )
            )
        return tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> McpCallResult:
        result = self._send_request("tools/call", {"name": name, "arguments": arguments})
        text = _extract_tool_text(result)
        is_error = bool(result.get("isError", False))
        return McpCallResult(raw=result, text=text, is_error=is_error)

    def stderr_tail(self) -> List[str]:
        return list(self._stderr_lines)

    def _initialize_protocol(self) -> None:
        _ = self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lean-mcp-chat", "version": "0.1.0"},
            },
        )
        self._send_notification("notifications/initialized", {})

    def _send_notification(self, method: str, params: Dict[str, Any]) -> None:
        self._write_json({"jsonrpc": "2.0", "method": method, "params": params})

    def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._write_json(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )
        return self._wait_for_response(request_id)

    def _wait_for_response(self, request_id: int) -> Dict[str, Any]:
        while True:
            msg = self._read_json_line_with_timeout()
            if msg is None:
                raise McpClientError(
                    f"Timeout waiting for MCP response to request id={request_id}. "
                    f"stderr_tail={self.stderr_tail()[-5:]}"
                )
            if "id" not in msg:
                continue
            if msg.get("id") != request_id:
                continue
            if "error" in msg:
                raise McpClientError(f"MCP error: {msg['error']}")
            result = msg.get("result", {})
            if not isinstance(result, dict):
                raise McpClientError("MCP response result must be an object")
            return result

    def _write_json(self, payload: Dict[str, Any]) -> None:
        proc = self._require_proc()
        if proc.stdin is None:
            raise McpClientError("MCP stdin is not available")
        serialized = json.dumps(payload, separators=(",", ":"))
        proc.stdin.write(serialized + "\n")
        proc.stdin.flush()

    def _read_json_line_with_timeout(self) -> Optional[Dict[str, Any]]:
        proc = self._require_proc()
        if proc.stdout is None:
            raise McpClientError("MCP stdout is not available")
        ready, _, _ = select.select([proc.stdout], [], [], self.timeout_seconds)
        if not ready:
            return None
        line = proc.stdout.readline()
        if not line:
            raise McpClientError("MCP process closed stdout unexpectedly")
        candidate = line.strip()
        if not candidate:
            return {}
        # Some servers can log plaintext to stdout; skip non-JSON lines safely.
        if not re.match(r"^\s*\{", candidate):
            return {}
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        return parsed

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self._stderr_lines.append(line.rstrip())

    def _require_proc(self) -> subprocess.Popen[str]:
        if self._proc is None:
            raise McpClientError("MCP process is not started")
        if self._proc.poll() is not None:
            raise McpClientError(
                f"MCP process exited with code {self._proc.returncode}. "
                f"stderr_tail={self.stderr_tail()[-10:]}"
            )
        return self._proc


def _extract_tool_text(result: Dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return json.dumps(result)
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block and isinstance(block["text"], str):
            parts.append(block["text"])
            continue
        if "json" in block:
            parts.append(json.dumps(block["json"]))
            continue
        parts.append(json.dumps(block))
    return "\n".join(parts).strip()
