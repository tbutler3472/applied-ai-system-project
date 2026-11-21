from pathlib import Path
import random

from streamlit.testing.v1 import AppTest

from logic_utils import check_guess, parse_guess, update_score


def test_winning_guess():
    # If the secret is 50 and guess is 50, it should be a win
    result = check_guess(50, 50)
    assert result == ("Win", "🎉 Correct!")


def test_guess_too_high():
    # If secret is 50 and guess is 60, hint should be "Too High"
    result = check_guess(60, 50)
    assert result == ("Too High", "📉 Go LOWER!")


def test_guess_too_low():
    # If secret is 50 and guess is 40, hint should be "Too Low"
    result = check_guess(40, 50)
    assert result == ("Too Low", "📈 Go HIGHER!")


def test_higher_lower_hint_bug_regression():
    # Regression check for reversed comparisons: a lower guess against a high secret
    # should say "Too Low" with a "Go HIGHER!" hint, and vice versa.
    assert check_guess(4, 90) == ("Too Low", "📈 Go HIGHER!")
    assert check_guess(90, 4) == ("Too High", "📉 Go LOWER!")


def test_boundary_exact_match_edge_case():
    # Exact-match wins should still work at the lowest and highest numbers in range.
    assert check_guess(1, 1) == ("Win", "🎉 Correct!")
    assert check_guess(100, 100) == ("Win", "🎉 Correct!")


def test_extreme_comparison_hint_edge_case():
    # At the far ends of the range, the direction check must remain consistent.
    assert check_guess(1, 99) == ("Too Low", "📈 Go HIGHER!")
    assert check_guess(99, 1) == ("Too High", "📉 Go LOWER!")


def test_near_boundary_guess_edge_case():
    # One-number-off guesses should still produce the correct side and hint.
    assert check_guess(99, 100) == ("Too Low", "📈 Go HIGHER!")
    assert check_guess(2, 1) == ("Too High", "📉 Go LOWER!")


def test_check_guess_compares_string_secrets_numerically():
    assert check_guess(9, "10") == ("Too Low", "📈 Go HIGHER!")
    assert check_guess(10, "9") == ("Too High", "📉 Go LOWER!")
    assert check_guess(10, "10") == ("Win", "🎉 Correct!")


def test_parse_guess_accepts_integers_but_rejects_decimals():
    assert parse_guess(" 12 ") == (True, 12, None)
    assert parse_guess("12.0") == (False, None, "That is not a number.")
    assert parse_guess("") == (False, None, "Enter a guess.")


def test_update_score_preserves_existing_scoring_rules():
    assert update_score(0, "Win", 1) == 80
    assert update_score(0, "Win", 20) == 10
    assert update_score(10, "Too High", 2) == 15
    assert update_score(10, "Too High", 1) == 5
    assert update_score(10, "Too Low", 1) == 5


def _app_test():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(str(app_path)).run(timeout=10)


def test_invalid_and_out_of_range_guesses_do_not_consume_attempts():
    app = _app_test()
    app.text_input(key="guess_input_Normal").set_value("12.5")
    app.button[0].click().run(timeout=10)
    assert app.session_state["attempts"] == 0
    assert app.session_state["history"] == []

    app.text_input(key="guess_input_Normal").set_value("51")
    app.button[0].click().run(timeout=10)
    assert app.session_state["attempts"] == 0
    assert app.session_state["history"] == []


def test_new_game_resets_all_game_state(monkeypatch):
    app = _app_test()
    app.session_state["attempts"] = 3
    app.session_state["score"] = 42
    app.session_state["status"] = "won"
    app.session_state["history"] = [5, 10, 15]
    app.session_state["secret"] = 7
    monkeypatch.setattr(random, "randint", lambda low, high: 23)

    app.button[1].click().run(timeout=10)

    assert app.session_state["attempts"] == 0
    assert app.session_state["score"] == 0
    assert app.session_state["status"] == "playing"
    assert app.session_state["history"] == []
    assert app.session_state["secret"] == 23
