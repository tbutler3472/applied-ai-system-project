"""Input, selection, and player-facing explanation validation."""

import json
import re
from typing import Any

from rag_utils import (
    ATTEMPT_LIMITS,
    StrategyAnalysis,
    analyze_strategy,
    calculate_next_guess,
)


MAX_TOOL_CALLS = 5


class AgentError(Exception):
    """Raised when public agent state is unusable."""

def _validate_public_state(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> None:
    if difficulty not in ATTEMPT_LIMITS:
        raise AgentError("Unknown difficulty.")
    if (
        not isinstance(allowed_range, tuple)
        or len(allowed_range) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in allowed_range)
    ):
        raise AgentError("The allowed range is invalid.")
    try:
        analyze_strategy(difficulty, allowed_range, history)
    except (AttributeError, TypeError, ValueError, KeyError) as error:
        raise AgentError(f"Public game history is unusable: {error}") from error

def _reason_inconsistency(
    reason: str,
    attempts_remaining: int,
    history: list[dict[str, Any]],
) -> str | None:
    if not reason:
        return None

    exhausted_claim = re.search(
        r"\b(?:out of|no more|no|zero)\s+(?:guesses?|attempts?)\b|"
        r"\b(?:no|zero)\s+(?:guesses?|attempts?)\s+(?:left|remaining)\b",
        reason,
        re.I,
    )
    if exhausted_claim and attempts_remaining > 0:
        return (
            "Inconsistent reason: it says no attempts remain, but the "
            f"validated count is {attempts_remaining}."
        )
    if attempts_remaining == 0 and re.search(
        r"\b(?:one|two|three|four|five|six|seven|eight|\d+)\s+"
        r"(?:more\s+)?(?:guesses?|attempts?)\s+(?:left|remaining)\b",
        reason,
        re.I,
    ):
        return "Inconsistent reason: it claims attempts remain, but the validated count is zero."

    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
    }
    claims = re.findall(
        r"\b(?:you have|with|only|there are)\s+"
        r"(\d+|zero|one|two|three|four|five|six|seven|eight)\s+"
        r"(?:more\s+)?(?:guesses?|attempts?)\b",
        reason,
        re.I,
    )
    for claim in claims:
        count = int(claim) if claim.isdigit() else number_words[claim.lower()]
        if count != attempts_remaining:
            return (
                "Inconsistent reason: it gives a different remaining-attempt "
                f"count than the validated count of {attempts_remaining}."
            )

    outcome_by_guess = {
        entry["guess"]: entry["outcome"]
        for entry in history
        if entry["outcome"] in {"Too High", "Too Low"}
    }
    for guess, outcome in outcome_by_guess.items():
        claimed_outcome = re.search(
            rf"\b(?:guess\s+)?{guess}\s+"
            rf"(?:was|is|came back as|turned out to be)\s+"
            r"too\s+(high|low)\b",
            reason,
            re.I,
        )
        if (
            claimed_outcome
            and claimed_outcome.group(1).lower()
            != outcome.lower().replace("too ", "")
        ):
            return (
                f"Inconsistent reason: it gives the wrong outcome for guess "
                f"{guess}; the recorded clue was {outcome}."
            )

    respective_claim = re.search(
        r"\b(\d+)\s+and\s+(\d+)\s+(?:were|are)\s+(?:both\s+)?"
        r"too\s+(high|low)\s+and\s+too\s+(high|low),?\s+respectively\b",
        reason,
        re.I,
    )
    if respective_claim:
        claimed_pairs = (
            (respective_claim.group(1), respective_claim.group(3)),
            (respective_claim.group(2), respective_claim.group(4)),
        )
        for guess_text, outcome_text in claimed_pairs:
            expected = outcome_by_guess.get(int(guess_text))
            if expected and expected.lower() != f"too {outcome_text.lower()}":
                return (
                    "Inconsistent reason: it assigns a recorded guess the "
                    "opposite higher-or-lower outcome."
                )
    return None

def _parse_selection(
    raw: str,
    allowed_tools: set[str],
) -> dict[str, Any]:
    def merge_tool_lists(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key not in result:
                result[key] = value
            elif (
                key == "tools"
                and isinstance(result[key], list)
                and isinstance(value, list)
            ):
                result[key].extend(value)
            else:
                raise AgentError("The agent response repeated a JSON field.")
        return result

    try:
        selection = json.loads(raw, object_pairs_hook=merge_tool_lists)
    except (json.JSONDecodeError, TypeError) as error:
        raise AgentError("The agent returned malformed JSON.") from error
    if not isinstance(selection, dict):
        raise AgentError("The agent response must be a JSON object.")
    if "state" in selection:
        selection = {key: value for key, value in selection.items() if key != "state"}
    if set(selection) == {"tool", "reason"}:
        selection = {
            "tools": [
                {
                    "tool": selection["tool"],
                    "arguments": {},
                    "reason": selection["reason"],
                }
            ]
        }
    if "tools" not in selection or set(selection) - {"tools", "summary"}:
        raise AgentError("The agent response must provide a tool plan.")
    if not isinstance(selection["tools"], list) or len(selection["tools"]) > MAX_TOOL_CALLS:
        raise AgentError("The agent tool plan is invalid.")
    if "summary" in selection and not isinstance(selection["summary"], str):
        raise AgentError("The optional agent summary must be text.")
    for index, item in enumerate(selection["tools"]):
        if (
            not isinstance(item, dict)
            or set(item) not in (
                {"tool", "reason"},
                {"tool", "arguments", "reason"},
            )
            or not isinstance(item["tool"], str)
            or not isinstance(item["reason"], str)
            or not item["reason"].strip()
            or len(item["reason"]) > 1000
        ):
            raise AgentError(
                "Each tool selection needs a valid name and reason, with empty arguments."
            )
        if item["tool"] not in allowed_tools:
            raise AgentError(f"Unknown tool name: {item['tool']}.")
        arguments = item.get("arguments", {})
        if not isinstance(arguments, dict) or arguments != {}:
            raise AgentError(
                "Each tool selection needs a valid name and reason, with empty arguments."
            )
        selection["tools"][index] = {
            "tool": item["tool"],
            "arguments": {},
            "reason": item["reason"],
        }
    return selection

def _summary_rejection_reason(
    summary: str,
    analysis: StrategyAnalysis,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    used_tools: set[str],
) -> str | None:
    if not summary or len(summary) > 1000:
        return "The summary was empty or too long."
    if re.search(
        r"\b(secret|hidden answer|source ids?|python|ollama|model|tool|prompt|"
        r"json|agent|implementation|retrieval|knowledge base|the player|"
        r"binary search|"
        r"analyze_range|review_guess_history|evaluate_attempts|"
        r"retrieve_strategy|recommend_next_guess)\b",
        summary,
        re.I,
    ):
        return "The summary included technical or hidden-information claims."
    if re.search(
        r"\b(?:close|near) (?:to )?(?:the )?(?:secret|hidden number|answer)\b|"
        r"\b(?:you're|you are) close\b",
        summary,
        re.I,
    ):
        return "The summary speculated about closeness to the hidden answer."
    positive_guarantee = re.search(
        r"\b(?:can|could|will|still can)\s+(?:still\s+)?guarantee\b|"
        r"\b(?:win|success) is (?:still )?guaranteed\b|"
        r"\bguaranteed (?:to )?win\b|"
        r"\benough (?:tries|attempts|guesses) to (?:guarantee|win)\b|"
        r"\bsure to win\b",
        summary,
        re.I,
    )
    negative_guarantee = re.search(
        r"\b(?:cannot|can't|can’t|won't|won’t|isn't|isn’t|not)\s+"
        r"(?:still\s+)?(?:be\s+)?(?:guarantee|guaranteed|certain)\b|"
        r"\bnot guaranteed\b|\bno guaranteed win\b|"
        r"\bnot (?:enough|possible) (?:tries|attempts|guesses|to guarantee)\b|"
        r"\bmore (?:possible )?(?:numbers|answers|choices|possibilities) "
        r"than (?:you have )?(?:attempts|guesses|tries)\b",
        summary,
        re.I,
    )
    if positive_guarantee and not analysis.can_guarantee_win:
        return "The summary incorrectly claimed a guaranteed win."
    if negative_guarantee and analysis.can_guarantee_win:
        return "The summary incorrectly claimed a win was not guaranteed."

    nonwinning_guesses = [
        entry for entry in history if entry["outcome"] != "Win"
    ]
    inefficient = max(0, len(nonwinning_guesses) - analysis.efficient_guesses)
    inefficiency_claim = re.search(
        r"\b(?:inefficient|fewer than half|less than half|"
        r"eliminated little|barely narrowed|did not eliminate|didn't eliminate)\b",
        summary,
        re.I,
    )
    if inefficiency_claim and not inefficient:
        return "The summary claimed inefficient guesses, but the clues narrowed the range efficiently."
    if inefficient and inefficiency_claim:
        count_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
        }
        inefficiency_counts = re.findall(
            r"\b(\d+|one|two|three|four|five|six|seven|eight)\s+"
            r"(?:of\s+(?:your\s+)?(?:\d+\s+)?|your\s+)?"
            r"(?:earlier\s+)?(?:guesses|clues)\b",
            summary,
            re.I,
        )
        if any(
            (int(value) if value.isdigit() else count_words[value.lower()])
            != inefficient
            for value in inefficiency_counts
        ):
            return "The summary gave an incorrect count of inefficient guesses."
    if analysis.repeated_guesses == 0:
        repeated_claim = re.search(
            r"\b(?:you repeated|you have repeated|there (?:was|were) "
            r"(?:a )?repeated|repeated the same|guessed the same)\b",
            summary,
            re.I,
        )
        if repeated_claim:
            return "The summary claimed repeated guesses, but none occurred."
    elif re.search(
        r"\b(?:no|none|zero)\s+(?:repeated|duplicate)\s+guesses?\b|"
        r"\b(?:didn't|did not|never)\s+repeat(?:ed)?\s+(?:a\s+)?guess\b",
        summary,
        re.I,
    ):
        return "The summary denied repeated guesses that appear in the history."
    if inefficient and re.search(
        r"\b(?:all|every|each)\s+(?:of )?(?:your )?(?:guesses|clues)"
        r"\s+(?:were|have been)\s+(?:efficient|effective)\b",
        summary,
        re.I,
    ):
        return "The summary incorrectly described every guess as efficient."

    valid_numbers = {
        allowed_range[0],
        allowed_range[1],
        allowed_range[1] - allowed_range[0] + 1,
        analysis.possible_count,
        analysis.attempts_remaining,
    }
    if analysis.recommended_guess is not None:
        valid_numbers.add(analysis.recommended_guess)
    valid_numbers.update(entry["guess"] for entry in history)
    for index in range(len(history) + 1):
        bounds, _ = calculate_next_guess(allowed_range, history[:index])
        if bounds:
            valid_numbers.update(bounds)
            valid_numbers.add(bounds[1] - bounds[0] + 1)
    mentioned_numbers = [int(value) for value in re.findall(r"\b\d+\b", summary)]
    if any(value not in valid_numbers for value in mentioned_numbers):
        return "The summary included a number not supported by public game facts."
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
    }
    attempt_words_pattern = (
        r"\d+|zero|one|two|three|four|five|six|seven|eight"
    )
    attempt_claims = re.findall(
        rf"\b(?:you have|with|only|there are|remaining)\s+"
        rf"({attempt_words_pattern})\s+(?:more\s+)?(?:guesses?|attempts?)"
        rf"(?:\s+(?:left|remaining))?\b|"
        rf"\b({attempt_words_pattern})\s+(?:more\s+)?(?:guesses?|attempts?)"
        rf"\s+(?:left|remaining)\b",
        summary,
        re.I,
    )
    if any(
        (int(value) if value.isdigit() else number_words[value.lower()])
        != analysis.attempts_remaining
        for claim in attempt_claims
        for value in claim
        if value
    ):
        return "The summary gave an incorrect remaining-attempt count."
    if analysis.attempts_remaining != 1 and re.search(
        r"\b(?:last|final)\s+(?:guess|attempt)\b", summary, re.I
    ):
        return "The summary incorrectly described the next guess as the last."
    if re.search(r"\b(?:answer|secret number|hidden number) is \d+\b", summary, re.I):
        return "The summary claimed to know the hidden answer."
    next_guess_claims = re.findall(
        r"\b(?:try|choose|pick|recommend|go with|start with)\s+"
        r"(?:number\s+)?(\d+)\b|"
        r"\b(?:next guess|recommendation)(?: is|:)?\s+(\d+)\b",
        summary,
        re.I,
    )
    if analysis.recommended_guess is not None and any(
        int(value) != analysis.recommended_guess
        for claim in next_guess_claims
        for value in claim
        if value
    ):
        return "The summary recommended a different number than the validated guess."

    if analysis.remaining_range is not None:
        low, high = analysis.remaining_range
        range_mentions = re.findall(r"\b(\d+)\s*(?:-|–|to)\s*(\d+)\b", summary)
        if any((int(first), int(last)) != (low, high) for first, last in range_mentions):
            valid_ranges = {
                tuple(calculate_next_guess(allowed_range, history[:index])[0])
                for index in range(len(history) + 1)
                if calculate_next_guess(allowed_range, history[:index])[0] is not None
            }
            if any((int(first), int(last)) not in valid_ranges for first, last in range_mentions):
                    return "The summary stated a range inconsistent with the public clues."

    if "recommend_next_guess" in used_tools and analysis.recommended_guess is not None:
        if str(analysis.recommended_guess) not in summary:
                return "The summary did not include the validated next guess."
    if not history and "recommend_next_guess" in used_tools and not re.search(
        r"\b(?:start|begin|first|opening)\b", summary, re.I
    ):
        return "The opening advice did not explain how to start."
    if not history and re.search(
        r"\b(?:review|revisit|analyze)\b.{0,35}"
        r"\b(?:guess )?history\b|\b(?:previous|past) guesses\b",
        summary,
        re.I,
    ):
        return "The summary referred to guesses that have not been made."
    if "analyze_range" in used_tools and analysis.remaining_range is not None:
        low, high = analysis.remaining_range
        if not (
            str(analysis.possible_count) in summary
            or (str(low) in summary and str(high) in summary)
        ):
            return "The summary did not use the analyzed range or possible count."
    if "evaluate_attempts" in used_tools:
        attempt_words = {
            0: ("zero", "no"),
            1: ("one", "a single", "last"),
            2: ("two",),
            3: ("three",),
            4: ("four",),
            5: ("five",),
            6: ("six",),
            7: ("seven",),
            8: ("eight",),
        }.get(analysis.attempts_remaining, ())
        if str(analysis.attempts_remaining) not in summary and not any(
            re.search(rf"\b{re.escape(word)}\b", summary, re.I)
            for word in attempt_words
        ):
            return (
                "The summary did not report the validated remaining-attempt "
                f"count ({analysis.attempts_remaining})."
            )
        if analysis.can_guarantee_win and not positive_guarantee:
            return "The summary did not explain that a win can still be guaranteed."
        if not analysis.can_guarantee_win and not negative_guarantee:
            return "The summary did not explain that a win is not guaranteed."
    if "review_guess_history" in used_tools and history:
        if not re.search(r"\b(?:guess|clue|history|range)\b", summary, re.I):
            return "The summary did not discuss the reviewed guess history."
        if inefficient and not re.search(
            r"\b(?:inefficient|fewer|half|narrow|cut)\b", summary, re.I
        ):
            return "The summary omitted the supported history-efficiency finding."
        if analysis.repeated_guesses and not re.search(
            r"\b(?:repeat|repeated|again)\b", summary, re.I
        ):
            return "The summary omitted the supported repeated-guess finding."

    sentence_count = len(re.findall(r"[.!?](?:\s|$)", summary))
    if not 3 <= sentence_count <= 6:
        return (
            "The summary must be 3–6 complete sentences; "
            f"it contained {sentence_count}."
        )
    return None

def _required_summary_tools(
    analysis: StrategyAnalysis,
    history: list[dict[str, Any]],
) -> set[str]:
    required = {"evaluate_attempts"}
    if analysis.recommended_guess is not None:
        required.add("recommend_next_guess")
    inefficient = len([entry for entry in history if entry["outcome"] != "Win"]) - analysis.efficient_guesses
    if history and (analysis.repeated_guesses or inefficient):
        required.add("review_guess_history")
    return required
