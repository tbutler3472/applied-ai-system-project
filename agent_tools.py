"""Read-only Python tools available to the strategy agent."""

import json
from typing import Any

from agent_validation import AgentError
from rag_utils import StrategyError, analyze_strategy, calculate_next_guess, retrieve_strategies


TOOL_DESCRIPTIONS = {
    "analyze_range": "Find the remaining possible range using higher/lower clues.",
    "review_guess_history": "Summarize repeats and how much each clue narrowed the range.",
    "evaluate_attempts": "Check remaining attempts and whether a win is still guaranteed.",
    "retrieve_strategy": "Find relevant strategy advice from the local guessing knowledge base.",
    "recommend_next_guess": "Find an unused integer nearest the remaining range midpoint.",
}

def analyze_range(
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining_range, _ = calculate_next_guess(allowed_range, history)
    if remaining_range is None:
        return {"completed": True, "remaining_range": None, "possible_count": 0}
    return {
        "completed": False,
        "remaining_range": list(remaining_range),
        "possible_count": remaining_range[1] - remaining_range[0] + 1,
    }

def review_guess_history(
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    seen: set[int] = set()
    reviews = []
    for index, entry in enumerate(history, start=1):
        guess = entry["guess"]
        outcome = entry["outcome"]
        repeated = guess in seen
        seen.add(guess)
        before_range, _ = calculate_next_guess(
            allowed_range, history[: index - 1]
        )
        after_range, _ = calculate_next_guess(allowed_range, history[:index])
        before_count = (
            0 if before_range is None else before_range[1] - before_range[0] + 1
        )
        after_count = (
            0 if after_range is None else after_range[1] - after_range[0] + 1
        )
        reviews.append(
            {
                "attempt": index,
                "outcome": outcome,
                "repeated": repeated,
                "possibilities_before": before_count,
                "possibilities_after": after_count,
                "narrowed_by_at_least_half": (
                    after_count <= before_count // 2
                    if outcome != "Win" and before_count
                    else None
                ),
            }
        )
    return {
        "guesses_made": len(history),
        "repeated_guesses": sum(review["repeated"] for review in reviews),
        "inefficient_guesses": sum(
            review["narrowed_by_at_least_half"] is False for review in reviews
        ),
        "reviews": reviews,
    }

def evaluate_attempts(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    analysis = analyze_strategy(difficulty, allowed_range, history)
    return {
        "attempts_remaining": analysis.attempts_remaining,
        "possible_count": analysis.possible_count,
        "can_guarantee_win": analysis.can_guarantee_win,
        "completed": analysis.completed,
    }

def retrieve_strategy(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return retrieve_strategies(difficulty, allowed_range, history)

def recommend_next_guess(
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining, guess = calculate_next_guess(allowed_range, history)
    return {
        "remaining_range": list(remaining) if remaining else None,
        "recommended_guess": guess,
    }

def _tool_result(
    name: str,
    arguments: dict[str, Any],
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> Any:
    if arguments != {}:
        raise AgentError("This tool takes no model-supplied arguments.")
    if name == "analyze_range":
        return analyze_range(allowed_range, history)
    if name == "review_guess_history":
        if not history:
            raise AgentError("History review is not useful before the first guess.")
        return review_guess_history(allowed_range, history)
    if name == "evaluate_attempts":
        return evaluate_attempts(difficulty, allowed_range, history)
    if name == "retrieve_strategy":
        try:
            return retrieve_strategy(difficulty, allowed_range, history)
        except StrategyError as error:
            raise AgentError(str(error)) from error
    if name == "recommend_next_guess":
        return recommend_next_guess(allowed_range, history)
    raise AgentError("The selected tool is not allowed.")
