import inspect
import json
from unittest.mock import Mock

import requests

import agent_utils
from agent_test_helpers import _done, _plan, _response, _tool
from agent_utils import (
    MAX_TOOL_CALLS,
    analyze_range,
    recommend_next_guess,
    run_strategy_agent,
)

def test_prior_tool_results_are_compact_and_keep_useful_facts():
    large_strategy_text = "Useful strategy. " * 100
    prior_results = [
        {
            "tool": "review_guess_history",
            "result": {
                "guesses_made": 3,
                "repeated_guesses": 0,
                "inefficient_guesses": 2,
                "reviews": [{"details": "unneeded verbose detail"}],
            },
        },
        {
            "tool": "retrieve_strategy",
            "result": [
                {
                    "id": "midpoint",
                    "title": "Choose a midpoint",
                    "text": large_strategy_text,
                    "unused_metadata": "omit",
                }
            ],
        },
        {
            "event": "finish_validation",
            "model_summary": "unneeded long response",
            "reason": "The summary did not report attempts.",
        },
    ]

    compact = agent_utils._compact_prior_results(prior_results)
    encoded = json.dumps(compact)

    assert compact[0]["result"]["inefficient_guesses"] == 2
    assert len(compact[1]["result"][0]["text"]) == 240
    assert "unneeded verbose detail" not in encoded
    assert "unused_metadata" not in encoded
    assert "unneeded long response" not in encoded


def test_public_state_only_and_deterministic_recommendation(monkeypatch):
    hidden = "secret-number-never-send"
    selections = [
        _tool("analyze_range"),
        _tool("evaluate_attempts"),
        _tool("recommend_next_guess"),
        _tool("review_guess_history"),
        _done(
            "Your Too Low clue leaves 33 possible numbers, from 68 to 100. "
            "That clue removed more than half of the original choices. "
            "Try 84 next to split the remaining options again. You have 4 "
            "attempts and cannot guarantee a win from here, so make every clue count."
        ),
    ]
    post = Mock(side_effect=[_response(item) for item in selections])
    monkeypatch.setattr(agent_utils.requests, "post", post)

    assert list(inspect.signature(run_strategy_agent).parameters) == [
        "difficulty",
        "allowed_range",
        "history",
    ]
    result = run_strategy_agent(
        "Hard",
        (1, 100),
        [{"guess": 67, "outcome": "Too Low"}],
    )
    all_requests = json.dumps([call.kwargs["json"] for call in post.call_args_list])
    assert hidden not in all_requests
    assert result.analysis.remaining_range == (68, 100)
    assert result.analysis.recommended_guess == 84
    assert analyze_range((1, 100), [{"guess": 67, "outcome": "Too Low"}]) == {
        "completed": False,
        "remaining_range": [68, 100],
        "possible_count": 33,
    }
    assert recommend_next_guess(
        (1, 100), [{"guess": 67, "outcome": "Too Low"}]
    )["recommended_guess"] == 84


def test_completed_game_does_not_call_model(monkeypatch):
    post = Mock()
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent(
        "Normal", (1, 50), [{"guess": 25, "outcome": "Win"}]
    )

    assert result.analysis.completed
    assert result.analysis.recommended_guess is None
    assert "found the number" in result.explanation
    assert result.trace == []
    post.assert_not_called()


def test_inconsistent_history_is_rejected_before_model_call(monkeypatch):
    post = Mock()
    monkeypatch.setattr(agent_utils.requests, "post", post)

    try:
        run_strategy_agent(
            "Normal",
            (1, 50),
            [
                {"guess": 25, "outcome": "Too High"},
                {"guess": 25, "outcome": "Too Low"},
            ],
        )
    except agent_utils.AgentError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("Inconsistent history must be rejected.")
    post.assert_not_called()


def test_tool_arguments_must_be_empty(monkeypatch):
    post = Mock(
        return_value=_response(
            _tool("analyze_range", arguments={"guess": 12})
        )
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert "empty arguments" in result.stopped_reason
