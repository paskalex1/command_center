import json
import logging
import subprocess
import uuid
from typing import Any, Dict, List

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


def _run_request(server: MCPServer, payload: Dict[str, Any]) -> Dict[str, Any]:
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


def list_tools_for_server(server: MCPServer) -> List[Dict[str, Any]]:
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/list",
        "params": {},
    }

    response = _run_request(server, payload)
    result = response.get("result") or {}
    tools = result.get("tools") or []

    if not isinstance(tools, list):
        raise MCPClientError("Invalid tools format from MCP server")

    return tools


def call_tool(server: MCPServer, tool: MCPTool, arguments: Dict[str, Any]) -> Dict[str, Any]:
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

    response = _run_request(server, payload)
    return response.get("result") or {}


def sync_tools_for_server(server: MCPServer) -> int:
    tools_data = list_tools_for_server(server)

    seen_names: set[str] = set()

    for tool_info in tools_data:
        name = tool_info.get("name")
        if not name:
            continue

        seen_names.add(name)

        description = tool_info.get("description", "") or ""
        input_schema = tool_info.get("input_schema") or {}
        output_schema = tool_info.get("output_schema") or {}

        MCPTool.objects.update_or_create(
            server=server,
            name=name,
            defaults={
                "description": description,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "is_active": True,
            },
        )

    if seen_names:
        MCPTool.objects.filter(server=server).exclude(name__in=seen_names).update(
            is_active=False
        )

    return MCPTool.objects.filter(server=server, is_active=True).count()

