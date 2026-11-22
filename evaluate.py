"""Small reproducible Project 4 evaluation checks.

The required checks are deterministic and do not need Ollama. Use --ollama
only to add an optional local-model smoke test.
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import Mock, patch

import requests

import agent_utils
from agent_validation import AgentError, _parse_selection
from logic_utils import get_range_for_difficulty
from rag_utils import analyze_strategy, calculate_next_guess, retrieve_strategies


passed = 0
failed = 0


def check(name: str, condition: bool) -> None:
    global passed, failed
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {name}")
    if condition:
        passed += 1
    else:
        failed += 1


def run_evaluation(include_ollama: bool) -> int:
    global passed, failed

    remaining, guess = calculate_next_guess(
        (1, 100),
        [{"guess": 60, "outcome": "Too High"}, {"guess": 30, "outcome": "Too Low"}],
    )
    check("correct remaining range", remaining == (31, 59))

    opening_history = []
    opening_range, recommended = calculate_next_guess((1, 20), opening_history)
    guessed_numbers = {entry["guess"] for entry in opening_history}
    check(
        "recommended guess is in range and unused",
        opening_range is not None
        and recommended is not None
        and opening_range[0] <= recommended <= opening_range[1]
        and recommended not in guessed_numbers
        and recommended == 10,
    )

    remaining, repeated_safe_guess = calculate_next_guess(
        (1, 10),
        [{"guess": 5, "outcome": "Too High"}, {"guess": 5, "outcome": "Too High"}],
    )
    check(
        "avoids repeated guesses",
        remaining == (1, 4) and repeated_safe_guess == 2,
    )

    analysis = analyze_strategy(
        "Normal",
        (1, 50),
        [{"guess": 30, "outcome": "Too High"}],
    )
    check("attempts remaining is correct", analysis.attempts_remaining == 7)

    easy = analyze_strategy("Easy", (1, 20), [])
    hard = analyze_strategy("Hard", (1, 100), [])
    check(
        "guarantee calculations are correct",
        easy.can_guarantee_win and not hard.can_guarantee_win,
    )

    plan = _parse_selection(
        '{"tools":[{"tool":"analyze_range","arguments":{},"reason":"Narrow the range."}]}',
        set(agent_utils.TOOL_DESCRIPTIONS),
    )
    check("valid tool-selection parsing", plan["tools"][0]["tool"] == "analyze_range")

    try:
        _parse_selection(
            '{"tools":[{"tool":"run_shell","reason":"Execute a command."}]}',
            set(agent_utils.TOOL_DESCRIPTIONS),
        )
        unknown_rejected = False
    except AgentError:
        unknown_rejected = True
    check("unknown tools are rejected", unknown_rejected)

    try:
        _parse_selection(
            '{"tools":[{"tool":"analyze_range","arguments":{"command":"bad"},"reason":"Unsafe."}]}',
            set(agent_utils.TOOL_DESCRIPTIONS),
        )
        unsafe_rejected = False
    except AgentError:
        unsafe_rejected = True
    malformed_rejected = False
    try:
        _parse_selection("not json", set(agent_utils.TOOL_DESCRIPTIONS))
    except AgentError:
        malformed_rejected = True
    check("unsafe and malformed requests are rejected", unsafe_rejected and malformed_rejected)

    with patch.object(agent_utils.requests, "post", Mock(side_effect=requests.Timeout("offline"))):
        fallback = agent_utils.run_strategy_agent("Easy", (1, 20), [])
    check(
        "deterministic fallback is used on Ollama failure",
        fallback.explanation_source == "fallback" and fallback.analysis.recommended_guess == 10,
    )

    ranges = {
        difficulty: get_range_for_difficulty(difficulty)
        for difficulty in ("Easy", "Normal", "Hard")
    }
    sources = retrieve_strategies("Normal", (1, 50), [])
    check(
        "difficulty settings and retrieval remain distinct",
        ranges == {"Easy": (1, 20), "Normal": (1, 50), "Hard": (1, 100)}
        and any(source["id"] == "normal_difficulty" for source in sources),
    )

    if include_ollama:
        print("OPTIONAL: Ollama smoke test starting (local llama3.2:3b).")
        try:
            result = agent_utils.run_strategy_agent("Easy", (1, 20), [])
            print(
                "OPTIONAL RESULT: agent returned successfully; "
                f"explanation_source={result.explanation_source}."
            )
        except (requests.RequestException, AgentError, ValueError) as error:
            print(f"OPTIONAL RESULT: Ollama smoke test unavailable or rejected: {error}")

    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Project 4 evaluation checks")
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="attempt an optional real llama3.2:3b smoke test",
    )
    args = parser.parse_args()

    print("Project 4 Evaluation")
    print("Required checks: deterministic; Ollama is mocked or not used.")
    if args.ollama:
        print("Optional real-Ollama mode: enabled")
    else:
        print("Optional real-Ollama mode: disabled")
    print()
    return run_evaluation(args.ollama)


if __name__ == "__main__":
    sys.exit(main())
