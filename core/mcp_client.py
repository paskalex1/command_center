import json
import logging
import subprocess
import uuid
from typing import Any, Dict, List, Tuple

import requests
from django.conf import settings

from .models import MCPServer, MCPTool

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 60


class MCPClientError(Exception):
    def __init__(self, message: str, code: int | None = None, data: Any | None = None):
        super().__init__(message)
        self.code = code
        self.data = data


def _build_command(server: MCPServer) -> List[str]:
    args = server.command_args or []
    if not isinstance(args, list):
        logger.warning("MCPServer.command_args is not a list, got %r", args)
        args = []
    return [server.command, *[str(a) for a in args]]


def _run_stdio_request(server: MCPServer, payload: Dict[str, Any]) -> Dict[str, Any]:
    cmd = _build_command(server)
    timeout = DEFAULT_TIMEOUT_SECONDS

    try:
        completed = subprocess.run(
            cmd,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("MCP server %s timed out: %s", server.name, exc)
        raise MCPClientError(f"MCP server '{server.name}' timed out") from exc
    except OSError as exc:
        logger.error("Failed to start MCP server %s: %s", server.name, exc)
        raise MCPClientError(
            f"Failed to start MCP server '{server.name}': {exc}"
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        logger.error(
            "MCP server %s exited with %s: %s",
            server.name,
            completed.returncode,
            stderr,
        )
        raise MCPClientError(
            f"MCP server '{server.name}' exited with code {completed.returncode}",
        )

    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise MCPClientError(f"MCP server '{server.name}' returned empty response")

    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON from MCP server %s: %s", server.name, stdout)
        raise MCPClientError("Invalid JSON response from MCP server") from exc

    if "error" in response and response["error"] is not None:
        err = response["error"]
        raise MCPClientError(
            message=str(err.get("message") or "MCP server error"),
            code=err.get("code"),
            data=err.get("data"),
        )

    return response


def _run_http_request(
    server: MCPServer,
    method: str,
    params: Dict[str, Any] | None = None,
    session_id: str | None = None,
) -> tuple[Dict[str, Any], str | None]:
    """
    HTTP-клиент для MCP-серверов.

    - Требует server.base_url.
    - Добавляет Origin из настроек (MCP_HTTP_ORIGIN), чтобы пройти CORS/Origin-проверку
      у HTTP MCP-серверов (например, cyanheads/filesystem-mcp-server).
    """
    if not server.base_url:
        raise MCPClientError(
            f"MCP server '{server.name}' has HTTP transport but base_url is not configured"
        )

    origin = getattr(settings, "MCP_HTTP_ORIGIN", "http://command-center")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Origin": origin,
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params or {},
    }

    try:
        resp = requests.post(
            server.base_url,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:  # noqa: PERF203
        logger.error("HTTP request to MCP server %s failed: %s", server.name, exc)
        raise MCPClientError(
            f"Failed to call MCP server '{server.name}' via HTTP: {exc}"
        ) from exc

    if resp.status_code < 200 or resp.status_code >= 300:
        logger.error(
            "MCP HTTP server %s returned status %s: %s",
            server.name,
            resp.status_code,
            resp.text,
        )
        raise MCPClientError(
            f"MCP HTTP server '{server.name}' returned status {resp.status_code}"
        )

    content_type = resp.headers.get("Content-Type", "")
    try:
        if "text/event-stream" in content_type:
            response = _parse_sse_json(resp.text)
        else:
            response = resp.json()
    except ValueError as exc:  # JSONDecodeError is subclass of ValueError
        logger.error("Invalid JSON from MCP HTTP server %s: %s", server.name, resp.text)
        raise MCPClientError("Invalid JSON response from MCP HTTP server") from exc

    response_session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
    if response_session_id:
        session_id = response_session_id

    if "error" in response and response["error"] is not None:
        err = response["error"]
        raise MCPClientError(
            message=str(err.get("message") or "MCP HTTP server error"),
            code=err.get("code"),
            data=err.get("data"),
        )

    return response, session_id


def _parse_sse_json(body: str) -> Dict[str, Any]:
    """
    Разбирает ответ SSE и возвращает первый валидный JSON из строк data: ...
    """
    events = body.split("\n\n")
    for event in events:
        data_lines: list[str] = []
        for line in event.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        payload = "\n".join(data_lines).strip()
        if not payload:
            continue
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON data found in SSE response")


def list_tools_for_server(server: MCPServer) -> List[Dict[str, Any]]:
    if server.transport == MCPServer.TRANSPORT_HTTP:
        session_id = None
        initialize_params = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "command-center",
                "version": "0.1.0",
            },
        }
        _, session_id = _run_http_request(
            server,
            method="initialize",
            params=initialize_params,
            session_id=None,
        )
        response, _ = _run_http_request(
            server,
            method="tools/list",
            params={},
            session_id=session_id,
        )
    else:
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/list",
            "params": {},
        }
        response = _run_stdio_request(server, payload)
    result = response.get("result") or {}
    tools = result.get("tools") or []

    if not isinstance(tools, list):
        raise MCPClientError("Invalid tools format from MCP server")

    return tools


def call_tool(server: MCPServer, tool: MCPTool, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if server.transport == MCPServer.TRANSPORT_HTTP:
        initialize_params = {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "command-center",
                "version": "0.1.0",
            },
        }
        _, session_id = _run_http_request(
            server,
            method="initialize",
            params=initialize_params,
            session_id=None,
        )
        response, _ = _run_http_request(
            server,
            method="tools/call",
            params={
                "name": tool.name,
                "arguments": arguments or {},
            },
            session_id=session_id,
        )
    else:
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool.name,
                "arguments": arguments or {},
            },
        }
        response = _run_stdio_request(server, payload)
    return response.get("result") or {}
