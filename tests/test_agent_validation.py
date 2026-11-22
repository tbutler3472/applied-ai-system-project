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

def test_contradictory_tool_reason_is_flagged_without_rewriting(monkeypatch):
    reason = "You're out of attempts, so I need to decide whether the game is over."
    post = Mock(
        side_effect=[
            _response(_tool("evaluate_attempts", reason)),
            requests.Timeout("stop after verifying the trace"),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent(
        "Hard",
        (1, 100),
        [
            {"guess": 60, "outcome": "Too High"},
            {"guess": 30, "outcome": "Too Low"},
            {"guess": 45, "outcome": "Too Low"},
            {"guess": 42, "outcome": "Too Low"},
        ],
    )

    assert result.analysis.attempts_remaining == 1
    assert result.trace[0].reason == reason
    assert "validated count is 1" in result.trace[0].warning


def test_tool_reason_with_wrong_prior_outcomes_is_flagged():
    reason = (
        "62 and 68 were both too high and too low, respectively, "
        "so review those clues."
    )
    history = [
        {"guess": 62, "outcome": "Too Low"},
        {"guess": 68, "outcome": "Too High"},
    ]

    warning = agent_utils._reason_inconsistency(reason, 3, history)

    assert "opposite higher-or-lower outcome" in warning


def test_one_attempt_fallback_explains_final_guess_probability_and_history(monkeypatch):
    monkeypatch.setattr(
        agent_utils.requests,
        "post",
        Mock(side_effect=requests.Timeout("too slow")),
    )
    history = [
        {"guess": 20, "outcome": "Too Low"},
        {"guess": 50, "outcome": "Too High"},
        {"guess": 35, "outcome": "Too High"},
        {"guess": 28, "outcome": "Too Low"},
    ]

    result = run_strategy_agent("Hard", (1, 100), history)

    assert result.analysis.attempts_remaining == 1
    assert result.analysis.remaining_range == (29, 34)
    assert result.analysis.recommended_guess == 31
    assert "one attempt left" in result.explanation
    assert "final informed guess" in result.explanation
    assert "does not make this number more likely" in result.explanation
    assert "eliminated fewer than half" in result.explanation


def test_fallback_identifies_efficient_and_inefficient_history(monkeypatch):
    monkeypatch.setattr(
        agent_utils.requests,
        "post",
        Mock(side_effect=requests.Timeout("too slow")),
    )

    inefficient = run_strategy_agent(
        "Hard", (1, 100), [{"guess": 20, "outcome": "Too Low"}]
    )
    efficient = run_strategy_agent(
        "Hard",
        (1, 100),
        [
            {"guess": 50, "outcome": "Too Low"},
            {"guess": 75, "outcome": "Too High"},
        ],
    )

    assert "eliminated fewer than half" in inefficient.explanation
    assert "cut the remaining choices by at least half" in efficient.explanation


def test_missing_tool_reason_is_rejected(monkeypatch):
    selection = {"tool": "analyze_range", "arguments": {}}
    post = Mock(return_value=_response(selection))
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert result.trace[0].tool == "Model response rejected"
    assert "valid name and reason" in result.trace[0].result


def test_contradictory_model_explanation_uses_personalized_fallback(monkeypatch):
    post = Mock(
        side_effect=[
            _response(
                _plan(
                    _tool("evaluate_attempts"),
                    _tool("recommend_next_guess"),
                    _tool("review_guess_history"),
                )
            ),
            _response(
                _done(
                    "You have 6 attempts left and can guarantee a win. "
                    "The remaining range has 20 possible numbers, from 1 to 20. "
                    "Start with 10 to split the choices. "
                    "You are close to the answer, so keep going."
                )
            ),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Hard", (1, 100), [
        {"guess": 60, "outcome": "Too High"},
        {"guess": 30, "outcome": "Too Low"},
        {"guess": 45, "outcome": "Too Low"},
        {"guess": 42, "outcome": "Too Low"},
    ])

    assert result.explanation_source == "fallback"
    assert result.analysis.attempts_remaining == 1
    assert result.analysis.recommended_guess == 52
    assert "final informed guess" in result.explanation
    assert "You are close" not in result.explanation
    assert result.trace[-1].tool == "Validation fallback"


def test_empty_history_does_not_accept_claim_to_review_past_guesses(monkeypatch):
    post = Mock(
        side_effect=[
            _response(
                _plan(
                    _tool("recommend_next_guess"),
                    _tool("evaluate_attempts"),
                    summary=(
                        "With 20 possible numbers, it is a good time to review your "
                        "guess history. The recommended next guess is 10. "
                        "You have 6 attempts, so use each clue to narrow the range. "
                        "A win is still guaranteed with balanced guesses."
                    ),
                )
            ),
            _response(
                _done(
                    "With 20 possible numbers, it is a good time to review your "
                    "guess history. The recommended next guess is 10. "
                    "You have 6 attempts, so use each clue to narrow the range. "
                    "A win is still guaranteed with balanced guesses."
                )
            ),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert result.explanation_source == "fallback"
    assert "Start with 10" in result.explanation
    assert result.trace[2].tool == "Model summary rejected"
    assert result.trace[3].tool == "Validation fallback"


def test_reported_one_sentence_response_has_specific_rejection_reason():
    history = [
        {"guess": 15, "outcome": "Too Low"},
        {"guess": 60, "outcome": "Too High"},
        {"guess": 45, "outcome": "Too High"},
    ]
    analysis = agent_utils.analyze_strategy("Hard", (1, 100), history)
    model_response = (
        "Given the history, let's see how much each clue narrowed the range "
        "and find a useful strategy to help you narrow down the remaining range."
    )

    reason = agent_utils._summary_rejection_reason(
        model_response,
        analysis,
        (1, 100),
        history,
        {"evaluate_attempts"},
    )

    assert reason == (
        "The summary did not report the validated remaining-attempt count (2)."
    )


def test_fallback_describes_inefficient_history_naturally(monkeypatch):
    monkeypatch.setattr(
        agent_utils.requests,
        "post",
        Mock(side_effect=requests.Timeout("too slow")),
    )
    history = [
        {"guess": 15, "outcome": "Too Low"},
        {"guess": 60, "outcome": "Too High"},
        {"guess": 45, "outcome": "Too High"},
    ]

    result = run_strategy_agent("Hard", (1, 100), history)

    assert "Three of your earlier guesses eliminated fewer than half" in result.explanation
    assert "3 of your 3 earlier clues" not in result.explanation


def test_summary_validation_rejects_false_history_efficiency_claims():
    history = [
        {"guess": 15, "outcome": "Too Low"},
        {"guess": 60, "outcome": "Too High"},
        {"guess": 45, "outcome": "Too High"},
    ]
    analysis = agent_utils.analyze_strategy("Hard", (1, 100), history)
    summary = (
        "Your clues narrowed the range to 29 possible answers. "
        "Only two of your earlier guesses eliminated fewer than half the choices. "
        "I recommend 30 for your next guess. With 2 attempts left, a win isn't guaranteed."
    )

    reason = agent_utils._summary_rejection_reason(
        summary,
        analysis,
        (1, 100),
        history,
        {"analyze_range", "evaluate_attempts", "recommend_next_guess", "review_guess_history"},
    )

    assert reason == "The summary gave an incorrect count of inefficient guesses."


def test_empty_history_does_not_offer_history_review_tool():
    prompt = agent_utils._agent_prompt(
        "Easy",
        (1, 20),
        [],
        agent_utils.analyze_strategy("Easy", (1, 20), []),
    )
    assert "review_guess_history" not in prompt.split("Available tools: ", 1)[1]


def test_argumentless_model_plan_is_normalized_to_safe_tool_arguments():
    selection = agent_utils._parse_selection(
        '{"tools":[{"tool":"analyze_range","reason":"Use the recorded clues."}],'
        '"state":{"ignored":"untrusted model echo"}}',
        set(agent_utils.TOOL_DESCRIPTIONS),
    )

    assert selection["tools"] == [
        {
            "tool": "analyze_range",
            "arguments": {},
            "reason": "Use the recorded clues.",
        }
    ]


def test_repeated_tools_json_fields_keep_each_model_selection():
    selection = agent_utils._parse_selection(
        '{"tools":[{"tool":"analyze_range","reason":"Narrow the range."}],'
        '"tools":[{"tool":"evaluate_attempts","reason":"Check remaining tries."}]}',
        set(agent_utils.TOOL_DESCRIPTIONS),
    )

    assert [item["tool"] for item in selection["tools"]] == [
        "analyze_range",
        "evaluate_attempts",
    ]


def test_missing_arguments_are_only_normalized_for_known_tools_with_a_reason():
    allowed_tools = set(agent_utils.TOOL_DESCRIPTIONS)

    for raw in (
        '{"tools":[{"tool":"unknown_tool","reason":"Still useful."}]}',
        '{"tools":[{"tool":"analyze_range","reason":"  "}]}',
        '{"tools":[{"tool":"analyze_range","reason":""}]}',
        '{"tools":[{"tool":"analyze_range","arguments":{"range":[1,20]},'
        '"reason":"Use the recorded clues."}]}',
    ):
        try:
            agent_utils._parse_selection(raw, allowed_tools)
        except agent_utils.AgentError:
            pass
        else:
            raise AssertionError(f"Unsafe tool selection was accepted: {raw}")


def test_tool_plan_prompt_requests_canonical_arguments_field():
    prompt = agent_utils._agent_prompt(
        "Hard",
        (1, 100),
        [{"guess": 40, "outcome": "Too Low"}],
        agent_utils.analyze_strategy(
            "Hard", (1, 100), [{"guess": 40, "outcome": "Too Low"}]
        ),
    )

    assert '"tool":"tool_name","arguments":{},"reason":"brief reason"' in prompt


def test_model_summary_that_mentions_internal_tools_is_rejected():
    history = [
        {"guess": 60, "outcome": "Too High"},
        {"guess": 30, "outcome": "Too Low"},
        {"guess": 45, "outcome": "Too Low"},
    ]
    analysis = agent_utils.analyze_strategy("Hard", (1, 100), history)
    summary = (
        "You've made three attempts so far, and the tool recommends guessing "
        "52, which is still too high given the remaining range of 46-59. "
        "It's risky to continue guessing too high, as indicated by the 'Too "
        "High' outcome with a guess of 60. The tool also suggests reviewing "
        "your history, where you've made three guesses with one 'Too Low' "
        "outcome and two 'Too High' outcomes. It's efficient to stick to a "
        "strategy that avoids repeating guesses. Given two attempts remaining, "
        "it's wise to try a guess within the lower end of the range to minimize risk."
    )

    reason = agent_utils._summary_rejection_reason(
        summary,
        analysis,
        (1, 100),
        history,
        {"evaluate_attempts", "recommend_next_guess", "review_guess_history"},
    )

    assert reason == "The summary included technical or hidden-information claims."


def test_final_explanation_prompt_prohibits_internal_tool_and_false_history_claims():
    history = [
        {"guess": 60, "outcome": "Too High"},
        {"guess": 30, "outcome": "Too Low"},
        {"guess": 45, "outcome": "Too Low"},
    ]
    analysis = agent_utils.analyze_strategy("Hard", (1, 100), history)
    prompt = agent_utils._final_explanation_prompt(
        "Hard",
        (1, 100),
        history,
        analysis,
        [
            {"tool": "review_guess_history", "result": {"guesses_made": 3}},
        ],
    )

    assert "Never mention tools, models, prompts, or implementation details." in prompt
    assert "match each guess to its recorded Too High or Too Low outcome exactly" in prompt
    assert "Do not claim the recommended guess is too high or too low" in prompt
    assert "Explicitly state whether a win can be guaranteed" in prompt
