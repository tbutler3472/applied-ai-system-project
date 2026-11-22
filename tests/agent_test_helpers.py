import json
from unittest.mock import Mock


def _response(selection):
    if "tool" in selection:
        selection = {
            "tools": [
                {
                    "tool": selection["tool"],
                    "arguments": selection.get("arguments", {}),
                    "reason": selection.get("reason", ""),
                }
            ]
        }
    elif selection.get("done") is True:
        selection = {"summary": selection["summary"]}
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": json.dumps(selection)}
    return response


def _plan(*tools, summary=None):
    response = {"tools": list(tools)}
    if summary is not None:
        response["summary"] = summary
    return response


def _tool(name, reason="This is useful next.", arguments=None):
    return {
        "tool": name,
        "reason": reason,
        "arguments": {} if arguments is None else arguments,
    }


def _done(summary):
    return {"done": True, "summary": summary}
