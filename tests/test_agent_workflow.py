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

def test_agent_selects_executes_tools_and_passes_results(monkeypatch):
    selections = [
        _plan(
            _tool("analyze_range", "Find what is still possible."),
            _tool("evaluate_attempts", "Check the actual attempt limit and win risk."),
            _tool("recommend_next_guess", "Choose a balanced unused guess."),
            _tool("review_guess_history", "Check how much the previous clue narrowed the choices."),
            summary=(
            "Your 30 guess left 29 possible answers, from 1 to 29. "
            "That clue eliminated fewer than half the choices, so aim nearer "
            "the middle next time. I recommend 15 to divide the remaining "
            "options more evenly. You have 7 attempts, and a win is still "
            "guaranteed if you keep using the clues."
            ),
        )
    ]
    post = Mock(side_effect=[_response(item) for item in selections])
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent(
        "Normal",
        (1, 50),
        [{"guess": 30, "outcome": "Too High"}],
    )

    assert [step.tool for step in result.trace] == [
        "analyze_range",
        "evaluate_attempts",
        "recommend_next_guess",
        "review_guess_history",
        "Model-requested finish",
    ]
    assert result.analysis.remaining_range == (1, 29)
    assert result.analysis.recommended_guess == 15
    assert "29 possible answers" in result.explanation
    assert result.explanation_source == "ollama"
    prompts = [call.kwargs["json"]["prompt"] for call in post.call_args_list]
    assert len(prompts) == 1
    assert post.call_args.kwargs["json"]["options"]["num_predict"] == 128
    assert '"allowed_range":[1,50]' in prompts[0]
    assert "brief reason grounded in the state" in prompts[0]
    assert "Do not explain the game" in prompts[0]
    assert "Validated attempts remaining: 7" in prompts[0]
    assert post.call_args.kwargs["json"]["format"] == "json"
    assert result.trace[0].reason == "Find what is still possible."
    assert "secret-number-never-send" not in "\n".join(prompts)


def test_agent_selects_different_sequences_for_different_states(monkeypatch):
    scenario_responses = [
        _plan(
            _tool("analyze_range", "Check the opening range before choosing a first guess."),
            _tool("retrieve_strategy", "Find an opening tip suited to Easy mode."),
            _tool("recommend_next_guess", "Use the opening range to select a balanced guess."),
            _tool("evaluate_attempts", "Check whether the six attempts are enough for this range."),
        ),
        _done(
            "Easy starts with 20 possible numbers, from 1 to 20. "
            "Begin with 10 to split those choices into smaller groups. "
            "You have 6 attempts, so use each higher-or-lower clue. "
            "A win is still guaranteed if you keep making balanced guesses."
        ),
        _plan(
            _tool("review_guess_history"),
            _tool("analyze_range"),
            _tool("recommend_next_guess"),
            _tool("evaluate_attempts"),
        ),
        _done(
            "Your two lower clues leave 70 possible answers, from 31 to 100. "
            "Both guesses removed fewer than half the choices, so a more "
            "balanced pick can narrow them faster. Try 65 next to divide the "
            "remaining range. You have 3 attempts left, and a win isn't guaranteed."
        ),
    ]
    post = Mock(side_effect=[_response(item) for item in scenario_responses])
    monkeypatch.setattr(agent_utils.requests, "post", post)

    new_game = run_strategy_agent("Easy", (1, 20), [])
    inefficient_history = run_strategy_agent(
        "Hard",
        (1, 100),
        [
            {"guess": 20, "outcome": "Too Low"},
            {"guess": 30, "outcome": "Too Low"},
        ],
    )

    assert [step.tool for step in new_game.trace] == [
        "analyze_range",
        "retrieve_strategy",
        "recommend_next_guess",
        "evaluate_attempts",
        "Model-requested finish",
    ]
    assert [step.tool for step in inefficient_history.trace] == [
        "review_guess_history",
        "analyze_range",
        "recommend_next_guess",
        "evaluate_attempts",
        "Model-requested finish",
    ]
    first_prompt, later_prompt = [
        call.kwargs["json"]["prompt"] for call in post.call_args_list
    ][0], [call.kwargs["json"]["prompt"] for call in post.call_args_list][2]
    assert '"history":[]' in first_prompt
    assert '"guess":20' in later_prompt
    final_prompts = [
        post.call_args_list[index].kwargs["json"]["prompt"]
        for index in (1, 3)
    ]
    assert len(post.call_args_list) == 4
    assert '"tool": "analyze_range"' in final_prompts[0]
    assert '"recommended_guess": 10' in final_prompts[0]
    assert '"attempts_remaining": 6' in final_prompts[0]
    assert '"inefficient_guesses": 2' in final_prompts[1]
    assert '"tool": "review_guess_history"' in final_prompts[1]


def test_maximum_tool_calls_stops_loop(monkeypatch):
    names = [
        "analyze_range",
        "review_guess_history",
        "evaluate_attempts",
        "retrieve_strategy",
        "recommend_next_guess",
    ]
    post = Mock(
        side_effect=[
            _response(
                _plan(
                    *[_tool(name, "This result is needed.") for name in names],
                    summary=(
                        "Your Too Low clue leaves 25 possible numbers, from 26 to 50. "
                        "That first clue cut the choices in half, with no repeats. "
                        "Try 38 next to split the remaining range. You have 7 "
                        "attempts, so a win is still guaranteed if you use the clues."
                    ),
                )
            ),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent(
        "Normal", (1, 50), [{"guess": 25, "outcome": "Too Low"}]
    )

    assert post.call_count == 1
    assert sum(step.tool in names for step in result.trace) == MAX_TOOL_CALLS
    assert result.stopped_reason is None
    assert result.analysis.recommended_guess == 38
    assert post.call_args_list[0].kwargs["json"]["format"] == "json"


def test_unknown_tool_stops_safely(monkeypatch):
    post = Mock(
        return_value=_response(_tool("run_shell", "It might be helpful."))
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert len(result.trace) == 1
    assert result.trace[0].tool == "Model response rejected"
    assert "Unknown tool name" in result.trace[0].result
    assert result.analysis.recommended_guess == 10


def test_live_shape_without_arguments_executes_every_valid_selection(monkeypatch):
    history = [
        {"guess": 40, "outcome": "Too Low"},
        {"guess": 65, "outcome": "Too High"},
        {"guess": 49, "outcome": "Too Low"},
    ]
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": (
            '{"tools":['
            '{"tool":"review_guess_history","reason":"Track earlier clues."},'
            '{"tool":"analyze_range","reason":"Narrow the possible values."},'
            '{"tool":"evaluate_attempts","reason":"Check remaining attempts."}'
            ']}'
        )
    }
    post = Mock(
        side_effect=[
            response,
            requests.Timeout("Skip final explanation in this regression test."),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Hard", (1, 100), history)

    assert result.analysis.remaining_range == (50, 64)
    assert result.analysis.attempts_remaining == 2
    assert result.analysis.recommended_guess == 57
    assert [step.tool for step in result.trace[:3]] == [
        "review_guess_history",
        "analyze_range",
        "evaluate_attempts",
    ]
    assert all(step.result for step in result.trace[:3])
    assert post.call_count == 2


def test_malformed_json_uses_deterministic_fallback(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": "{not json"}
    monkeypatch.setattr(agent_utils.requests, "post", Mock(return_value=response))

    result = run_strategy_agent("Easy", (1, 20), [])

    assert result.analysis.recommended_guess == 10
    assert "Start with" in result.explanation
    assert "malformed JSON" in result.stopped_reason


def test_repeated_tool_selection_is_ignored_and_traced(monkeypatch):
    post = Mock(
        return_value=_response(
            _plan(
                _tool("analyze_range", "Calculate the opening range."),
                _tool("analyze_range", "Double-check the same range."),
                _tool("recommend_next_guess", "Select the validated opening guess."),
                _tool("evaluate_attempts", "Check whether a win can be guaranteed."),
                summary=(
                    "Start with 10 to split Easy's 20 possible numbers, from 1 "
                    "to 20. You have 6 attempts left, and a win is still "
                    "guaranteed if you use each clue to divide the range. "
                    "Follow every higher-or-lower clue to keep reducing the "
                    "possible choices."
                ),
            )
        )
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert post.call_count == 1
    assert [step.tool for step in result.trace] == [
        "analyze_range",
        "analyze_range",
        "recommend_next_guess",
        "evaluate_attempts",
        "Model-requested finish",
    ]
    assert result.trace[1].reason == "Double-check the same range."
    assert "Repeated tool request ignored" in result.trace[1].result
    assert result.explanation_source == "ollama"


def test_model_timeout_uses_local_fallback(monkeypatch):
    post = Mock(side_effect=requests.Timeout("too slow"))
    monkeypatch.setattr(
        agent_utils.requests,
        "post",
        post,
    )

    result = run_strategy_agent("Easy", (1, 20), [])

    assert agent_utils.OLLAMA_TIMEOUT_SECONDS == 60
    assert 0 < post.call_args.kwargs["timeout"] <= 24
    assert result.analysis.recommended_guess == 10
    assert "Start with 10" in result.explanation
    assert "timed out" in result.stopped_reason.lower()
    assert result.trace[0].tool == "Model request timed out"


def test_agent_deadline_stops_requests_and_preserves_completed_tools(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(agent_utils.time, "monotonic", lambda: clock[0])
    post = Mock(
        return_value=_response(
            _plan(
                _tool("analyze_range"),
                _tool("evaluate_attempts"),
                _tool("recommend_next_guess"),
            )
        )
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)
    original_tool_result = agent_utils._tool_result

    def finish_tool_then_expire(*args, **kwargs):
        result = original_tool_result(*args, **kwargs)
        clock[0] = 131.0
        return result

    monkeypatch.setattr(agent_utils, "_tool_result", finish_tool_then_expire)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert post.call_count == 1
    assert post.call_args.kwargs["timeout"] == 18
    assert [step.tool for step in result.trace] == [
        "analyze_range",
        "Agent time budget exhausted",
    ]
    assert result.analysis.recommended_guess == 10
    assert "Start with 10" in result.explanation
    assert result.explanation_source == "fallback"


def test_timeout_during_tool_selection_returns_personalized_fallback(monkeypatch):
    post = Mock(side_effect=requests.Timeout("selection request timed out"))
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert post.call_count == 1
    assert "Start with 10" in result.explanation
    assert result.explanation_source == "fallback"
    assert result.trace[0].tool == "Model request timed out"
    assert result.trace[0].result == "selection request timed out"


def test_each_model_request_uses_current_remaining_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(agent_utils.time, "monotonic", lambda: clock[0])

    def post_request(*args, **kwargs):
        if post.call_count == 1:
            clock[0] += 5.0
            return _response(
                _plan(
                    _tool("evaluate_attempts"),
                    _tool("recommend_next_guess"),
                )
            )
        raise requests.Timeout("stop after verifying reduced timeout")

    post = Mock(side_effect=post_request)
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert [call.kwargs["timeout"] for call in post.call_args_list] == [18, 21]
    assert result.trace[0].tool == "evaluate_attempts"
    assert result.explanation_source == "fallback"


def test_timeout_during_final_explanation_preserves_tool_trace(monkeypatch):
    clock = [100.0]
    deadline = clock[0] + agent_utils.AGENT_TIME_BUDGET_SECONDS
    monkeypatch.setattr(agent_utils.time, "monotonic", lambda: clock[0])
    responses = iter(
        [
            _response(
                _plan(
                    _tool("evaluate_attempts"),
                    _tool("recommend_next_guess"),
                    _tool("review_guess_history"),
                )
            ),
        ]
    )

    def post_request(*args, **kwargs):
        try:
            return next(responses)
        except StopIteration:
            clock[0] = deadline
            raise requests.Timeout("final request timed out")

    post = Mock(side_effect=post_request)
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent(
        "Hard",
        (1, 100),
        [{"guess": 50, "outcome": "Too Low"}],
    )

    assert post.call_count == 2
    assert [step.tool for step in result.trace] == [
        "evaluate_attempts",
        "recommend_next_guess",
        "review_guess_history",
        "Final explanation timed out",
    ]
    assert result.analysis.recommended_guess == 75
    assert "75" in result.explanation
    assert "attempts left" in result.explanation
    assert result.explanation_source == "fallback"
    assert [
        call.kwargs["json"]["options"]["num_predict"]
        for call in post.call_args_list
    ] == [128, 160]


def test_agent_can_stop_early_when_selected_results_are_enough(monkeypatch):
    post = Mock(
        return_value=_response(
            _plan(
                _tool(
                    "recommend_next_guess",
                    "You are starting a new round, so calculate a balanced opening guess.",
                ),
                _tool(
                    "evaluate_attempts",
                    "Check that the remaining tries can handle this starting range.",
                ),
                summary=(
                    "Easy has 20 possible numbers, from 1 to 20. "
                    "Start with 10 to split those choices into smaller groups. "
                    "You have 6 attempts, so use the next clue to narrow the range. "
                    "A win is still guaranteed with balanced guesses."
                ),
            )
        )
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert [step.tool for step in result.trace] == [
        "recommend_next_guess",
        "evaluate_attempts",
        "Model-requested finish",
    ]
    assert result.explanation_source == "ollama"
    assert post.call_count == 1


def test_reported_hard_mode_premature_finish_continues_needed_analysis(monkeypatch):
    history = [
        {"guess": 15, "outcome": "Too Low"},
        {"guess": 60, "outcome": "Too High"},
        {"guess": 45, "outcome": "Too High"},
    ]
    first_summary = (
        "Given the history, let's see how much each clue narrowed the range "
        "and find a useful strategy to help you narrow down the remaining range."
    )
    final_summary = (
        "Your clues have narrowed 100 possible numbers to 29, leaving 16 to 44. "
        "Each of your three guesses eliminated fewer than half the choices, "
        "so aim closer to the middle to narrow the range faster. "
        "I recommend 30 as your next guess. With 2 attempts left and more "
        "possible answers than attempts, a win isn't guaranteed."
    )
    post = Mock(
        side_effect=[
            _response(
                _plan(
                    _tool("analyze_range", "Find the possible range from the clues."),
                    _tool("review_guess_history", "Measure how much each clue narrowed the choices."),
                    _tool("recommend_next_guess", "Choose a legal, unused midpoint guess."),
                    _tool("evaluate_attempts", "Check the remaining attempts and win risk."),
                    summary=first_summary,
                )
            ),
            _response(_done(final_summary)),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Hard", (1, 100), history)

    assert result.analysis.remaining_range == (16, 44)
    assert result.analysis.possible_count == 29
    assert result.analysis.attempts_remaining == 2
    assert result.analysis.recommended_guess == 30
    assert result.explanation == final_summary
    assert result.explanation_source == "ollama"
    assert [step.tool for step in result.trace] == [
        "analyze_range",
        "review_guess_history",
        "recommend_next_guess",
        "evaluate_attempts",
        "Model summary rejected",
        "Model-requested finish",
    ]
    assert len(post.call_args_list) == 2
    final_prompt = post.call_args_list[1].kwargs["json"]["prompt"]
    assert '"tool": "review_guess_history"' in final_prompt
    assert '"attempts_remaining": 2' in final_prompt


def test_long_model_generated_reason_is_preserved(monkeypatch):
    reason = "The existing public clues make this analysis useful. " * 7
    post = Mock(
        side_effect=[
            _response(_tool("analyze_range", reason)),
            requests.Timeout("stop after verifying the trace"),
        ]
    )
    monkeypatch.setattr(agent_utils.requests, "post", post)

    result = run_strategy_agent("Easy", (1, 20), [])

    assert result.trace[0].reason == reason
    assert json.loads(result.trace[0].result)["possible_count"] == 20
