# core/dummy_mcp_server.py
import sys
import json
import uuid

def read_json_line():
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    return json.loads(line)

def write_json(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    while True:
        req = read_json_line()
        if req is None:
            break

        jsonrpc = req.get("jsonrpc", "2.0")
        req_id = req.get("id") or str(uuid.uuid4())
        method = req.get("method")
        params = req.get("params") or {}

        if method == "tools/list":
            # Отдаём один простой инструмент echo
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo tool: возвращает переданные аргументы",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"}
                            },
                            "required": ["message"]
                        },
                        "output_schema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"}
                            }
                        },
                    }
                ]
            }
            write_json({
                "jsonrpc": jsonrpc,
                "id": req_id,
                "result": result,
            })

        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}

            if name == "echo":
                message = arguments.get("message", "")
                result = {"message": f"Echo: {message}"}
                write_json({
                    "jsonrpc": jsonrpc,
                    "id": req_id,
                    "result": result,
                })
            else:
                write_json({
                    "jsonrpc": jsonrpc,
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {name}"
                    }
                })
        else:
            write_json({
                "jsonrpc": jsonrpc,
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            })


if __name__ == "__main__":
    main()
