"""Small prompt and Ollama-response helpers for the agent."""

import json
from typing import Any

import requests

from agent_tools import TOOL_DESCRIPTIONS
from agent_validation import AgentError
from rag_utils import OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS, OLLAMA_URL, StrategyAnalysis


TOOL_PLAN_MAX_TOKENS = 128
FINAL_EXPLANATION_MAX_TOKENS = 160

def _agent_prompt(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    analysis: StrategyAnalysis,
) -> str:
    available = {
        name: description
        for name, description in TOOL_DESCRIPTIONS.items()
        if not (name == "review_guess_history" and not history)
    }
    return (
        "You select read-only tools for a number guessing game. Choose only "
        "tools that add useful information for this state; do not select every "
        "tool automatically. Select one to three distinct tools, in execution "
        "order. Give each a brief reason grounded in the state. Tools take no "
        "arguments. Do not explain the game, invent results, or infer the "
        "secret. Return only the tools key in this JSON shape, without copying "
        "the state into the response. Include an empty arguments object for "
        "every selection: "
        "{\"tools\":[{\"tool\":\"tool_name\",\"arguments\":{},"
        "\"reason\":\"brief reason\"}]}.\n"
        "Tool meanings: analyze_range=narrow possibilities; "
        "review_guess_history=check repeats and progress; "
        "evaluate_attempts=check tries and win guarantee; "
        "retrieve_strategy=find a useful strategy; "
        "recommend_next_guess=choose an unused midpoint.\n"
        f"State: {json.dumps({'difficulty': difficulty, 'allowed_range': allowed_range, 'history': history}, separators=(',', ':'))}\n"
        f"Validated attempts remaining: {analysis.attempts_remaining}.\n"
        f"Available tools: {', '.join(available)}.\n"
    )

def _compact_prior_results(
    prior_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compacted = []
    for item in prior_results:
        if item.get("event") == "finish_validation":
            compacted.append(
                {"event": "finish_validation", "reason": item.get("reason")}
            )
            continue

        tool = item.get("tool")
        result = item.get("result")
        if tool == "review_guess_history" and isinstance(result, dict):
            compacted.append(
                {
                    "tool": tool,
                    "result": {
                        key: result.get(key)
                        for key in (
                            "guesses_made",
                            "repeated_guesses",
                            "inefficient_guesses",
                        )
                    },
                }
            )
        elif tool == "retrieve_strategy" and isinstance(result, list):
            compacted.append(
                {
                    "tool": tool,
                    "result": [
                        {
                            key: str(strategy.get(key, ""))[:240]
                            for key in ("id", "title", "text")
                            if key in strategy
                        }
                        for strategy in result[:3]
                        if isinstance(strategy, dict)
                    ],
                }
            )
        elif isinstance(result, dict):
            compacted.append(
                {
                    "tool": tool,
                    "result": {
                        key: value
                        for key, value in result.items()
                        if key
                        in {
                            "completed",
                            "remaining_range",
                            "possible_count",
                            "attempts_remaining",
                            "can_guarantee_win",
                            "recommended_guess",
                        }
                    },
                }
            )
    return compacted

def _model_request(
    prompt: str,
    timeout_seconds: float,
    final_explanation: bool = False,
) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": (
                    FINAL_EXPLANATION_MAX_TOKENS
                    if final_explanation
                    else TOOL_PLAN_MAX_TOKENS
                )
            },
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    content = response.json().get("response")
    if not isinstance(content, str) or not content.strip():
        raise AgentError("The agent returned an empty response.")
    return content

def _final_explanation_prompt(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    analysis: StrategyAnalysis,
    tool_results: list[dict[str, Any]],
) -> str:
    facts = {
        "difficulty": difficulty,
        "original_range": allowed_range,
        "current_range": analysis.remaining_range,
        "possible_count": analysis.possible_count,
        "recommended_guess": analysis.recommended_guess,
        "attempts_remaining": analysis.attempts_remaining,
        "can_guarantee_win": analysis.can_guarantee_win,
        "history": history,
        "executed_tool_results": _compact_prior_results(tool_results),
    }
    return (
        "Write a natural, personalized game-coach explanation in 3–6 sentences. "
        "Use only these validated public facts and actual executed tool results. "
        "Explain progress, attempts and risk, recommendation, and useful history "
        "efficiency when relevant. Speak directly to the player. Never mention "
        "tools, models, prompts, or implementation details. When describing "
        "past guesses, match each guess to its recorded Too High or Too Low "
        "outcome exactly; do not invent outcome counts. Do not claim the "
        "recommended guess is too high or too low based on the hidden answer. "
        "Explicitly state whether a win can be guaranteed, using the validated "
        "can_guarantee_win fact. Do not add numbers, infer the hidden answer, "
        "or contradict the facts. "
        "For a final attempt, do not say midpoint is more likely. "
        "Return only JSON: {\"summary\":\"...\"}.\n\n"
        f"Validated facts and executed results: {json.dumps(facts)}"
    )

def _parse_summary(raw: str) -> str:
    try:
        response = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        raise AgentError("The final explanation was malformed JSON.") from error
    if not isinstance(response, dict) or set(response) != {"summary"}:
        raise AgentError("The final explanation response was invalid.")
    if not isinstance(response["summary"], str):
        raise AgentError("The final explanation must be text.")
    return response["summary"]
