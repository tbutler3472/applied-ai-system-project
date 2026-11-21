import inspect
import json
from pathlib import Path
from unittest.mock import Mock

import requests
from streamlit.testing.v1 import AppTest

import rag_utils
from rag_utils import (
    StrategyError,
    analyze_strategy,
    calculate_next_guess,
    fallback_explanation,
    generate_strategy,
    retrieve_strategies,
)


def test_remaining_range_narrows_from_public_feedback():
    remaining, recommendation = calculate_next_guess(
        (1, 100),
        [
            {"guess": 60, "outcome": "Too High"},
            {"guess": 30, "outcome": "Too Low"},
        ],
    )
    assert remaining == (31, 59)
    assert recommendation == 45


def test_empty_history_recommends_unused_midpoint():
    assert calculate_next_guess((1, 10), []) == ((1, 10), 5)


def test_recommendation_skips_previously_guessed_numbers():
    remaining, recommendation = calculate_next_guess(
        (1, 10),
        [
            {"guess": 8, "outcome": "Too High"},
            {"guess": 3, "outcome": "Too Low"},
        ],
    )
    assert remaining == (4, 7)
    assert recommendation == 5
    assert recommendation not in {8, 3}

    analysis = analyze_strategy(
        "Normal",
        (1, 10),
        [
            {"guess": 8, "outcome": "Too High"},
            {"guess": 3, "outcome": "Too Low"},
            {"guess": 8, "outcome": "Too High"},
        ],
    )
    assert analysis.repeated_guesses == 1
    assert analysis.recommended_guess == 5


def test_inconsistent_feedback_is_rejected():
    try:
        calculate_next_guess(
            (1, 10),
            [
                {"guess": 7, "outcome": "Too High"},
                {"guess": 7, "outcome": "Too Low"},
            ],
        )
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("Contradictory feedback must not produce a range.")


def test_completed_game_has_no_recommendation():
    assert calculate_next_guess(
        (1, 10), [{"guess": 6, "outcome": "Win"}]
    ) == (None, None)


def test_analysis_reports_attempts_possible_count_and_guarantee_feasibility():
    analysis = analyze_strategy(
        "Hard",
        (1, 100),
        [{"guess": 67, "outcome": "Too Low"}],
    )
    assert analysis.remaining_range == (68, 100)
    assert analysis.possible_count == 33
    assert analysis.recommended_guess == 84
    assert analysis.attempts_remaining == 4
    assert not analysis.can_guarantee_win
    assert analysis.higher_range == (68, 83)
    assert analysis.lower_range == (85, 100)

    one_guess_left = analyze_strategy(
        "Easy",
        (1, 20),
        [
            {"guess": 10, "outcome": "Too Low"},
            {"guess": 15, "outcome": "Too High"},
            {"guess": 12, "outcome": "Too Low"},
            {"guess": 13, "outcome": "Too Low"},
            {"guess": 20, "outcome": "Too High"},
        ],
    )
    assert one_guess_left.attempts_remaining == 1
    assert one_guess_left.possible_count == 1
    assert one_guess_left.can_guarantee_win


def test_no_attempts_and_won_games_do_not_recommend():
    finished_by_attempts = analyze_strategy(
        "Hard",
        (1, 100),
        [
            {"guess": 90, "outcome": "Too High"},
            {"guess": 80, "outcome": "Too High"},
            {"guess": 70, "outcome": "Too High"},
            {"guess": 60, "outcome": "Too High"},
            {"guess": 50, "outcome": "Too High"},
        ],
    )
    assert finished_by_attempts.attempts_remaining == 0
    assert finished_by_attempts.recommended_guess is None

    won = analyze_strategy("Normal", (1, 50), [{"guess": 25, "outcome": "Win"}])
    assert won.completed
    assert won.recommended_guess is None


def test_coach_displays_last_attempt_and_feasibility_guidance(monkeypatch):
    monkeypatch.setattr(
        rag_utils.requests,
        "post",
        Mock(side_effect=requests.Timeout("offline")),
    )
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    app.selectbox[0].select("Hard").run(timeout=15)
    app.session_state["history"] = [67, 80, 90, 95]
    app.session_state["guess_outcomes"] = [
        "Too Low",
        "Too Low",
        "Too Low",
        "Too Low",
    ]
    app.session_state["attempts"] = 4
    app.button[2].click().run(timeout=15)

    assert any("one guess left" in item.value.lower() for item in app.warning)
    assert any("can guarantee" in item.value.lower() for item in app.info)
    assert any(
        element.value == "AI Strategy Coach" for element in app.subheader
    )


def test_fallback_is_natural_and_adapts_to_empty_history():
    analysis = analyze_strategy("Easy", (1, 20), [])
    explanation = fallback_explanation("Easy", analysis)
    assert "Welcome to Easy" in explanation
    assert "middle" in explanation
    assert "Python" not in explanation


def test_retrieval_prioritizes_feedback_and_difficulty_strategies():
    too_high = retrieve_strategies(
        "Hard",
        (1, 100),
        [{"guess": 70, "outcome": "Too High"}],
    )
    source_ids = {source["id"] for source in too_high}
    assert "narrow_after_too_high" in source_ids
    assert "hard_difficulty" in source_ids

    normal = retrieve_strategies(
        "Normal",
        (1, 50),
        [{"guess": 30, "outcome": "Too High"}],
    )
    normal_ids = {source["id"] for source in normal}
    assert "normal_difficulty" in normal_ids
    assert "hard_difficulty" not in normal_ids

    empty_history = retrieve_strategies("Easy", (1, 20), [])
    assert "easy_difficulty" in {source["id"] for source in empty_history}


def test_prompt_contains_retrieved_text_ids_and_public_history(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": (
            "The last clue gives you a smaller range to work with. "
            "Keep choosing from the smaller range they leave behind. "
            "A balanced next choice can make a big difference. "
            "Keep using each result to make the next choice count."
        )
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(rag_utils.requests, "post", post)

    history = [{"guess": 30, "outcome": "Too High"}]
    analysis, answer, sources = generate_strategy("Normal", (1, 50), history)

    assert "balanced next choice" in answer
    assert analysis.recommended_guess == 15
    assert sources
    payload = post.call_args.kwargs["json"]
    prompt = payload["prompt"]
    assert payload["model"] == "llama3.2:3b"
    assert payload["stream"] is False
    assert "Difficulty: Normal" in prompt
    assert "Original range: 1 to 50" in prompt
    assert "guess 30: Too High" in prompt
    assert "Possible numbers remaining: 29" in prompt
    assert "Recommended next guess: 15" in prompt
    assert "do not propose alternatives" in prompt
    for source in sources:
        assert f"Source ID: {source['id']}" in prompt
        assert source["text"] in prompt
    assert "friendly, thoughtful game coach" in prompt
    assert "Relevant strategy advice:" in prompt


def test_strategy_api_accepts_only_public_game_inputs(monkeypatch):
    hidden_number = "4821"
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": (
            "Use the clues to keep narrowing your choices. "
            "A balanced guess can help you make steady progress. "
            "Keep your focus on the range still in play. "
            "You have a useful path forward."
        )
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(rag_utils.requests, "post", post)

    parameters = inspect.signature(generate_strategy).parameters
    assert list(parameters) == ["difficulty", "allowed_range", "history"]
    helper_parameters = inspect.signature(calculate_next_guess).parameters
    assert list(helper_parameters) == ["allowed_range", "history"]
    retrieval_parameters = inspect.signature(retrieve_strategies).parameters
    assert "secret" not in retrieval_parameters
    analysis_parameters = inspect.signature(analyze_strategy).parameters
    assert "secret" not in analysis_parameters

    generate_strategy(
        "Normal",
        (1, 50),
        [{"guess": 18, "outcome": "Too Low"}],
    )
    request_payload = json.dumps(post.call_args.kwargs["json"])
    assert hidden_number not in request_payload
    assert set(post.call_args.kwargs["json"]) == {"model", "prompt", "stream"}


def test_retrieval_failure_is_reported_without_calling_ollama(monkeypatch):
    monkeypatch.setattr(rag_utils, "KNOWLEDGE_BASE_PATH", Path("missing-strategies.json"))
    post = Mock()
    monkeypatch.setattr(rag_utils.requests, "post", post)

    try:
        generate_strategy("Normal", (1, 50), [])
    except StrategyError as error:
        assert "knowledge base" in str(error)
    else:
        raise AssertionError("A missing knowledge base should raise StrategyError.")
    post.assert_not_called()


def test_ollama_connection_failure_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        rag_utils.requests,
        "post",
        Mock(side_effect=requests.ConnectionError("offline")),
    )

    analysis, explanation, _ = generate_strategy("Easy", (1, 20), [])
    assert analysis.recommended_guess == 10
    assert "Start with 10" in explanation


def test_ollama_timeout_has_actionable_error(monkeypatch):
    monkeypatch.setattr(
        rag_utils.requests,
        "post",
        Mock(side_effect=requests.Timeout("slow response")),
    )

    analysis, explanation, _ = generate_strategy("Easy", (1, 20), [])
    assert analysis.recommended_guess == 10
    assert "Start with 10" in explanation


def test_contradictory_model_response_uses_natural_fallback(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": (
            "You have 999 possibilities left. "
            "You are close to finding the answer. "
            "You should guarantee a win with your remaining attempts. "
            "Guess 999 next and ignore the clues."
        )
    }
    monkeypatch.setattr(rag_utils.requests, "post", Mock(return_value=response))

    analysis, explanation, _ = generate_strategy(
        "Hard", (1, 100), [{"guess": 67, "outcome": "Too Low"}]
    )
    assert analysis.recommended_guess == 84
    assert "84 is a balanced next choice" in explanation
    assert "more possible numbers" not in explanation
    assert "close to finding" not in explanation


def test_false_guarantee_claim_uses_local_coaching_fallback(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": (
            "You can guarantee a win by staying focused. "
            "The clues will always lead you to the answer. "
            "Keep going and you will be successful. "
            "There is no need to change your approach."
        )
    }
    monkeypatch.setattr(rag_utils.requests, "post", Mock(return_value=response))

    _, explanation, _ = generate_strategy(
        "Hard", (1, 100), [{"guess": 67, "outcome": "Too Low"}]
    )
    assert "84 is a balanced next choice" in explanation
    assert "guarantee" not in explanation


def test_technical_model_response_uses_player_friendly_fallback(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "response": (
            "According to the retrieved sources, you should use the midpoint. "
            "The model thinks this is a strong approach. "
            "The recommendation is validated and authoritative. "
            "The prompt explains why this strategy works."
        )
    }
    monkeypatch.setattr(rag_utils.requests, "post", Mock(return_value=response))

    _, explanation, _ = generate_strategy(
        "Easy", (1, 20), [{"guess": 5, "outcome": "Too Low"}]
    )
    assert "15 possible numbers left" in explanation
    assert "retrieved" not in explanation.lower()
