import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


KNOWLEDGE_BASE_PATH = Path(__file__).parent / "data" / "guessing_strategies.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT_SECONDS = 30
ATTEMPT_LIMITS = {"Easy": 6, "Normal": 8, "Hard": 5}


class StrategyError(Exception):
    """Raised when strategy retrieval cannot complete."""


@dataclass(frozen=True)
class StrategyAnalysis:
    remaining_range: tuple[int, int] | None
    possible_count: int
    recommended_guess: int | None
    attempts_remaining: int
    can_guarantee_win: bool
    higher_range: tuple[int, int] | None
    lower_range: tuple[int, int] | None
    efficient_guesses: int
    repeated_guesses: int
    completed: bool
    guesses_made: int


def calculate_next_guess(
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> tuple[tuple[int, int] | None, int | None]:
    """Narrow public bounds and return an unused integer nearest the midpoint."""
    allowed_low, allowed_high = allowed_range
    if allowed_low > allowed_high:
        raise ValueError("The allowed range is invalid.")

    lower, upper = allowed_range
    used_guesses: set[int] = set()
    for entry in history:
        if "guess" not in entry or "outcome" not in entry:
            raise ValueError("Each history entry needs a guess and outcome.")
        guess = entry["guess"]
        outcome = entry["outcome"]
        if not isinstance(guess, int) or isinstance(guess, bool):
            raise ValueError("History guesses must be integers.")
        if not allowed_low <= guess <= allowed_high:
            raise ValueError("A history guess is outside the allowed range.")

        used_guesses.add(guess)
        if outcome == "Win":
            return None, None
        if outcome == "Too High":
            upper = min(upper, guess - 1)
        elif outcome == "Too Low":
            lower = max(lower, guess + 1)
        else:
            raise ValueError(f"Unknown guess outcome: {outcome}")
        if lower > upper:
            raise ValueError("Guess feedback is inconsistent with the allowed range.")

    remaining = (lower, upper)
    midpoint_twice = lower + upper
    candidates = sorted(
        range(lower, upper + 1),
        key=lambda guess: (abs(2 * guess - midpoint_twice), guess),
    )
    recommended = next((guess for guess in candidates if guess not in used_guesses), None)
    return remaining, recommended


def analyze_strategy(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> StrategyAnalysis:
    """Summarize the public game state and deterministic coaching facts."""
    if difficulty not in ATTEMPT_LIMITS:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    remaining_range, guess = calculate_next_guess(allowed_range, history)
    attempts_remaining = max(0, ATTEMPT_LIMITS[difficulty] - len(history))
    completed = any(entry.get("outcome") == "Win" for entry in history)

    if remaining_range is None:
        return StrategyAnalysis(
            None,
            0,
            None,
            attempts_remaining,
            False,
            None,
            None,
            0,
            0,
            True,
            len(history),
        )

    lower, upper = remaining_range
    possible_count = upper - lower + 1
    if attempts_remaining == 0:
        guess = None

    efficient = 0
    repeated = 0
    prior_bounds = allowed_range
    seen: set[int] = set()
    for entry in history:
        prior_low, prior_high = prior_bounds
        current_guess = entry["guess"]
        if current_guess in seen:
            repeated += 1
        else:
            seen.add(current_guess)
        prior_count = prior_high - prior_low + 1
        if entry["outcome"] == "Too High":
            prior_bounds = (prior_low, min(prior_high, current_guess - 1))
        elif entry["outcome"] == "Too Low":
            prior_bounds = (max(prior_low, current_guess + 1), prior_high)
        if entry["outcome"] in {"Too High", "Too Low"}:
            new_count = prior_bounds[1] - prior_bounds[0] + 1
            if new_count <= prior_count // 2:
                efficient += 1

    can_guarantee = (
        attempts_remaining > 0
        and possible_count <= 2**attempts_remaining - 1
    )
    higher_range = (lower, guess - 1) if guess is not None and guess > lower else None
    lower_range = (guess + 1, upper) if guess is not None and guess < upper else None

    return StrategyAnalysis(
        remaining_range,
        possible_count,
        guess,
        attempts_remaining,
        can_guarantee,
        higher_range,
        lower_range,
        efficient,
        repeated,
        completed,
        len(history),
    )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _history_text(history: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"guess {entry['guess']}: {entry['outcome']}"
        for entry in history
        if "guess" in entry and "outcome" in entry
    ) or "No guesses yet."


def retrieve_strategies(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    top_k: int = 3,
    knowledge_base: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Find strategy documents relevant to the difficulty and public feedback."""
    if knowledge_base is None:
        try:
            with KNOWLEDGE_BASE_PATH.open(encoding="utf-8") as file:
                knowledge_base = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise StrategyError(
                "The guessing-strategy knowledge base could not be loaded."
            ) from error

    query = _tokens(
        f"{difficulty} {allowed_range[0]} {allowed_range[1]} "
        f"{_history_text(history)}"
    )
    if not history:
        query.update(_tokens("opening midpoint binary search"))
    if any(entry.get("outcome") == "Too High" for entry in history):
        query.update(_tokens("narrow range lower upper"))
    if any(entry.get("outcome") == "Too Low" for entry in history):
        query.update(_tokens("narrow range higher lower"))
    if len(history) > len({entry.get("guess") for entry in history}):
        query.update(_tokens("repeat previous guesses history"))

    difficulty_ids = {
        "Easy": "easy_difficulty",
        "Normal": "normal_difficulty",
        "Hard": "hard_difficulty",
    }
    ranked = []
    for entry in knowledge_base:
        if entry.get("id") in difficulty_ids.values() and entry.get("id") != difficulty_ids.get(difficulty):
            continue
        searchable = " ".join(
            [entry.get("title", ""), entry.get("text", "")]
            + entry.get("keywords", [])
        )
        score = len(query & _tokens(searchable))
        if entry.get("id") == difficulty_ids.get(difficulty):
            score += 5
        if score:
            ranked.append(
                (
                    score,
                    {
                        "id": entry["id"],
                        "title": entry["title"],
                        "text": entry["text"],
                    },
                )
            )
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [entry for _, entry in ranked[:max(0, top_k)]]


def fallback_explanation(
    difficulty: str,
    analysis: StrategyAnalysis,
) -> str:
    if analysis.completed:
        return "You already found the number—nice work! Start a new game whenever you’re ready for another round."
    if analysis.attempts_remaining == 0:
        return "That round is over, so there are no guesses left to make. Check the game result above, then start a new round and use each clue to cut the range roughly in half."
    if analysis.recommended_guess is None or analysis.remaining_range is None:
        return "There isn’t an unused number left in the possible range. Start a new game to get a fresh set of clues."

    count = analysis.possible_count
    attempts = analysis.attempts_remaining
    if analysis.guesses_made == 0:
        opening = (
            f"Welcome to {difficulty}! Start with {analysis.recommended_guess}; "
            "a middle-of-the-range guess gives you a useful first clue."
        )
    elif analysis.repeated_guesses:
        opening = (
            "You’ve already tried a number more than once, so switch to an "
            "unused choice and let each clue shrink the possibilities."
        )
    elif analysis.efficient_guesses == analysis.guesses_made:
        opening = (
            "You’re using the clues well; each useful result is narrowing "
            "the choices instead of leaving you to guess blindly."
        )
    else:
        opening = (
            "The earlier clues are still useful, and choosing nearer the "
            "middle of the remaining range will cut away more possibilities."
        )

    if attempts == 1:
        urgency = "This is your last try, so make this balanced guess count."
    elif attempts <= 2:
        urgency = "You’re low on tries, so use the next clue to eliminate as many choices as possible."
    else:
        urgency = f"You still have {attempts} tries to work with."

    recommendation = (
        f"There are {count} possible numbers left, and "
        f"{analysis.recommended_guess} is a balanced next choice."
    )
    next_step = (
        "Only one number fits the clues, so this guess can finish the game."
        if attempts == 1 and count == 1
        else "Use the next clue to cut the choices down again."
    )
    return f"{opening} {recommendation} {urgency} {next_step}"


def _valid_model_explanation(
    answer: str,
    analysis: StrategyAnalysis,
) -> bool:
    if not answer or re.search(r"\d", answer):
        return False
    if re.search(
        r"\b(close to the answer|close to winning|close to finding)\b",
        answer,
        re.I,
    ):
        return False
    if re.search(
        r"\b(python|ollama|rag|ai|model|prompt|source|sources|retrieval|"
        r"retrieved|knowledge base|documents?|implementation|validated|"
        r"authoritative|according to|instead|alternative|different guess)\b",
        answer,
        re.I,
    ):
        return False
    if re.search(
        r"\b(ignore the clues|guess randomly|random guess|avoid the midpoint|"
        r"do not use the midpoint|don't use the midpoint|pick another)\b",
        answer,
        re.I,
    ):
        return False
    if re.search(r"\b(guarantee\w*|ensure)\b", answer, re.I):
        return False
    if analysis.attempts_remaining <= 1 and re.search(
        r"\b(plenty|lots of|many|ample|time to spare)\b", answer, re.I
    ):
        return False
    if (
        analysis.guesses_made
        and analysis.efficient_guesses < analysis.guesses_made
        and re.search(
            r"\b(efficient|using the clues well|good use of the clues|"
            r"great job with your guesses)\b",
            answer,
            re.I,
        )
    ):
        return False
    return 4 <= len(re.findall(r"[.!?](?:\s|$)", answer)) <= 7


def _build_prompt(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    analysis: StrategyAnalysis,
    sources: list[dict[str, str]],
) -> str:
    source_context = "\n\n".join(
        f"Source ID: {source['id']}\n{source['title']}: {source['text']}"
        for source in sources
    )
    status = (
        f"The player has made {analysis.efficient_guesses} efficient guesses "
        f"and repeated {analysis.repeated_guesses} guesses."
    )
    return (
        "Speak directly to the player as a friendly, thoughtful game coach. "
        "Write four to seven natural sentences that apply the strategy context "
        "to this exact situation. Be encouraging but honest. Never pretend to "
        "know the hidden answer or claim the player is close to it. Do not list "
        "metrics, repeat numeric values, name source IDs, or mention tools or "
        "implementation. Do not invent mistakes or praise inefficient play. "
        "Explain how to improve when useful. Do not make claims about whether "
        "a win is guaranteed; that is shown separately. "
        "The displayed recommendation and bounds are fixed; "
        "do not propose alternatives.\n\n"
        f"Difficulty: {difficulty}\n"
        f"Original range: {allowed_range[0]} to {allowed_range[1]}\n"
        f"Public guess history: {_history_text(history)}\n"
        f"Possible numbers remaining: {analysis.possible_count}\n"
        f"Attempts remaining: {analysis.attempts_remaining}\n"
        f"Recommended next guess: {analysis.recommended_guess}\n"
        f"Remaining range: {analysis.remaining_range}\n"
        f"{status}\n\n"
        f"Relevant strategy advice:\n{source_context}"
    )


def generate_explanation(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    sources: list[dict[str, str]],
) -> str:
    """Generate an explanation grounded in public state, with a local fallback."""
    analysis = analyze_strategy(difficulty, allowed_range, history)
    fallback = fallback_explanation(difficulty, analysis)
    if (
        analysis.completed
        or analysis.attempts_remaining == 0
        or analysis.recommended_guess is None
        or not sources
    ):
        return fallback

    prompt = _build_prompt(difficulty, allowed_range, history, analysis, sources)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
    except (requests.RequestException, ValueError, AttributeError):
        return fallback
    return answer if _valid_model_explanation(answer, analysis) else fallback


def generate_strategy(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> tuple[StrategyAnalysis, str, list[dict[str, str]]]:
    """Analyze public game state and return coach text with retrieved evidence."""
    analysis = analyze_strategy(difficulty, allowed_range, history)
    sources = retrieve_strategies(difficulty, allowed_range, history)
    explanation = generate_explanation(
        difficulty, allowed_range, history, sources
    )
    return analysis, explanation, sources
