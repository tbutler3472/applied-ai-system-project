import json
import time
from dataclasses import dataclass
from typing import Any

import requests

from rag_utils import OLLAMA_TIMEOUT_SECONDS, StrategyAnalysis, analyze_strategy
from agent_tools import (
    TOOL_DESCRIPTIONS,
    _tool_result,
    analyze_range,
    evaluate_attempts,
    recommend_next_guess,
    retrieve_strategy,
    review_guess_history,
)
from agent_validation import (
    MAX_TOOL_CALLS,
    AgentError,
    _parse_selection,
    _reason_inconsistency,
    _summary_rejection_reason,
    _validate_public_state,
)
from agent_prompts import (
    _agent_prompt,
    _compact_prior_results,
    _final_explanation_prompt,
    _model_request,
    _parse_summary,
)


AGENT_TIME_BUDGET_SECONDS = 30
FINAL_EXPLANATION_RESERVE_SECONDS = 12
AGENT_RETURN_MARGIN_SECONDS = 4





@dataclass(frozen=True)
class AgentTrace:
    tool: str
    reason: str
    result: str
    warning: str | None = None


@dataclass(frozen=True)
class AgentResult:
    analysis: StrategyAnalysis
    explanation: str
    trace: list[AgentTrace]
    stopped_reason: str | None = None
    explanation_source: str = "fallback"


































def run_strategy_agent(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
) -> AgentResult:
    """Select a safe tool plan, execute it locally, then explain the results."""
    deadline = time.monotonic() + AGENT_TIME_BUDGET_SECONDS
    _validate_public_state(difficulty, allowed_range, history)
    analysis = analyze_strategy(difficulty, allowed_range, history)
    fallback = _fallback_agent_summary(
        difficulty, allowed_range, history, analysis
    )
    if analysis.completed or analysis.attempts_remaining == 0:
        return AgentResult(analysis, fallback, [])

    trace: list[AgentTrace] = []
    remaining = deadline - time.monotonic()
    selection_timeout = min(
        remaining - FINAL_EXPLANATION_RESERVE_SECONDS,
        OLLAMA_TIMEOUT_SECONDS,
    )
    if selection_timeout <= 0:
        return _deadline_fallback(analysis, fallback, trace)

    try:
        plan_response = _model_request(
            _agent_prompt(difficulty, allowed_range, history, analysis),
            selection_timeout,
        )
        if time.monotonic() >= deadline:
            return _deadline_fallback(analysis, fallback, trace)
        plan = _parse_selection(plan_response, set(TOOL_DESCRIPTIONS))
    except requests.Timeout as error:
        detail = "Tool selection timed out; used a personalized deterministic explanation."
        trace.append(AgentTrace("Model request timed out", "Ollama did not select tools in time.", str(error)))
        return AgentResult(analysis, fallback, trace, detail)
    except (requests.RequestException, AgentError, ValueError) as error:
        trace.append(
            AgentTrace(
                "Model response rejected",
                "The tool plan could not be used safely.",
                str(error),
            )
        )
        return AgentResult(analysis, fallback, trace, str(error))

    if not plan["tools"]:
        detail = "Ollama selected no tools; used a personalized deterministic explanation."
        trace.append(
            AgentTrace(
                "No tools selected",
                "Ollama returned an empty tool plan.",
                "No tool was executed.",
            )
        )
        return AgentResult(analysis, fallback, trace, detail)

    tool_results: list[dict[str, Any]] = []
    used_tools: set[str] = set()
    for selected in plan["tools"]:
        if time.monotonic() >= deadline:
            return _deadline_fallback(analysis, fallback, trace)
        name = selected["tool"]
        reason = selected["reason"] or "Not provided"
        if name not in TOOL_DESCRIPTIONS:
            detail = f"Unknown tool selected: {name}"
            trace.append(AgentTrace(name, reason, detail))
            return AgentResult(analysis, fallback, trace, detail)
        warning = _reason_inconsistency(
            reason, analysis.attempts_remaining, history
        )
        if name in used_tools:
            trace.append(
                AgentTrace(
                    name,
                    reason,
                    "Repeated tool request ignored; the earlier result was retained.",
                    warning,
                )
            )
            continue
        if len(reason) > 1000:
            detail = "The selected tool did not include a valid short reason."
            trace.append(AgentTrace(name, reason, detail))
            return AgentResult(analysis, fallback, trace, detail)

        try:
            result = _tool_result(
                name,
                selected["arguments"],
                difficulty,
                allowed_range,
                history,
            )
        except (AgentError, ValueError, TypeError, KeyError) as error:
            trace.append(AgentTrace(name, reason, f"Tool failed safely: {error}", warning))
            return AgentResult(analysis, fallback, trace, str(error))

        used_tools.add(name)
        tool_results.append({"tool": name, "result": result})
        trace.append(
            AgentTrace(name, reason, json.dumps(result, ensure_ascii=False), warning)
        )
        if time.monotonic() >= deadline:
            return _deadline_fallback(analysis, fallback, trace)

    initial_summary = plan.get("summary", "")
    if initial_summary:
        problem = _summary_rejection_reason(
            initial_summary, analysis, allowed_range, history, used_tools
        )
        if not problem:
            trace.append(
                AgentTrace(
                    "Model-requested finish",
                    "The tool plan included a summary that passed public-fact checks.",
                    json.dumps({"summary": initial_summary, "validation": "accepted"}),
                )
            )
            return AgentResult(
                analysis, initial_summary, trace, explanation_source="ollama"
            )
        trace.append(
            AgentTrace(
                "Model summary rejected",
                "The initial explanation did not pass safety checks.",
                problem,
            )
        )

    remaining = deadline - time.monotonic()
    final_timeout = min(
        remaining - AGENT_RETURN_MARGIN_SECONDS,
        OLLAMA_TIMEOUT_SECONDS,
    )
    if final_timeout <= 0:
        return _deadline_fallback(analysis, fallback, trace)

    try:
        final_text = _model_request(
            _final_explanation_prompt(
                difficulty,
                allowed_range,
                history,
                analysis,
                tool_results,
            ),
            final_timeout,
            final_explanation=True,
        )
        if time.monotonic() >= deadline:
            return _deadline_fallback(analysis, fallback, trace)
        final_response = _parse_summary(final_text)
    except requests.Timeout as error:
        detail = "Final explanation timed out; used a personalized deterministic explanation."
        trace.append(
            AgentTrace("Final explanation timed out", "Ollama did not finish in time.", str(error))
        )
        return AgentResult(analysis, fallback, trace, detail)
    except (requests.RequestException, AgentError, ValueError) as error:
        trace.append(
            AgentTrace(
                "Model response rejected",
                "The final explanation could not be used safely.",
                str(error),
            )
        )
        return AgentResult(analysis, fallback, trace, str(error))

    problem = _summary_rejection_reason(
        final_response, analysis, allowed_range, history, used_tools
    )
    if problem:
        trace.append(
            AgentTrace(
                "Validation fallback",
                problem,
                "Displayed a personalized deterministic explanation.",
            )
        )
        return AgentResult(
            analysis, fallback, trace, problem, explanation_source="fallback"
        )

    trace.append(
        AgentTrace(
            "Model-requested finish",
            "The final explanation passed public-fact checks.",
            json.dumps({"summary": final_response, "validation": "accepted"}),
        )
    )
    return AgentResult(
        analysis, final_response, trace, explanation_source="ollama"
    )


def _deadline_fallback(
    analysis: StrategyAnalysis,
    fallback: str,
    trace: list[AgentTrace],
) -> AgentResult:
    detail = (
        f"The {AGENT_TIME_BUDGET_SECONDS}-second agent time budget was "
        "exhausted; used a deterministic explanation."
    )
    trace.append(
        AgentTrace(
            "Agent time budget exhausted",
            "Stopped before another model request.",
            detail,
        )
    )
    return AgentResult(analysis, fallback, trace, detail)




def _fallback_agent_summary(
    difficulty: str,
    allowed_range: tuple[int, int],
    history: list[dict[str, Any]],
    analysis: StrategyAnalysis,
) -> str:
    if analysis.completed:
        return "You found the number—well played! Start a new game whenever you’re ready for another round."
    if analysis.attempts_remaining == 0:
        return (
            f"That {difficulty} round is over, with no guesses left. "
            f"Your clues narrowed {allowed_range[1] - allowed_range[0] + 1} "
            f"possible numbers to {analysis.possible_count}. "
            "Next time, aim near the middle early and let every higher-or-lower clue "
            "shrink the choices."
        )
    if analysis.recommended_guess is None:
        return "There isn’t an unused number left among the possibilities. Start a new game to get a fresh round."
    initial_count = allowed_range[1] - allowed_range[0] + 1
    range_text = (
        f"{analysis.remaining_range[0]}–{analysis.remaining_range[1]}"
        if analysis.remaining_range
        else "no remaining range"
    )
    if history:
        progress = (
            f"On {difficulty}, your clues have narrowed {initial_count} possible "
            f"numbers to {analysis.possible_count}, leaving {range_text}."
        )
    else:
        progress = (
            f"On {difficulty}, there are {initial_count} possible numbers, "
            f"from {allowed_range[0]} to {allowed_range[1]}."
        )
    nonwinning_guesses = [entry for entry in history if entry["outcome"] != "Win"]
    inefficient = max(0, len(nonwinning_guesses) - analysis.efficient_guesses)
    history_note = ""
    if analysis.repeated_guesses:
        history_note = (
            f"You repeated {analysis.repeated_guesses} guess"
            f"{'es' if analysis.repeated_guesses != 1 else ''}, so avoid those "
            "numbers from here."
        )
    elif inefficient:
        amount = {
            1: "One",
            2: "Two",
            3: "Three",
            4: "Four",
            5: "Five",
            6: "Six",
            7: "Seven",
            8: "Eight",
        }.get(inefficient, str(inefficient))
        history_note = (
            f"{amount} of your earlier guesses eliminated fewer than half "
            "the possibilities. Aim closer to the middle of the current "
            "range on your next guess to narrow it faster."
        )
    elif nonwinning_guesses:
        history_note = (
            f"All {analysis.efficient_guesses} of your higher-or-lower clues "
            "cut the remaining choices by at least half."
        )
    if analysis.guesses_made == 0:
        guarantee_note = (
            "You still have enough tries to guarantee a win with balanced guesses."
            if analysis.can_guarantee_win
            else "Because this range is wider than your remaining attempts can guarantee, make each clue count."
        )
        return (
            f"{progress} Start with {analysis.recommended_guess} to split the "
            "choices into smaller groups. Use each higher-or-lower clue to "
            f"narrow the range; you have {analysis.attempts_remaining} attempts "
            f"and should use them carefully. {guarantee_note}"
        )
    if analysis.attempts_remaining == 1:
        final_risk = (
            "Only one possible answer remains, so this guess can still guarantee a win."
            if analysis.can_guarantee_win
            else "A win is not guaranteed with one attempt left."
        )
        sentences = [
            progress,
            history_note or "You have avoided repeating a guess so far.",
            f"I recommend {analysis.recommended_guess} as your final informed guess.",
        ]
        if analysis.can_guarantee_win:
            sentences.append(final_risk)
        else:
            sentences.extend(
                [
                    final_risk,
                    "Because no clue will follow, the midpoint does not make this number more likely than any other remaining choice.",
                ]
            )
        return " ".join(sentence for sentence in sentences if sentence)
    recommendation = (
        f"I recommend {analysis.recommended_guess}, near the middle of the "
        "remaining choices, so the next clue can narrow the range efficiently."
    )
    if analysis.can_guarantee_win:
        risk = (
            f"You have {analysis.attempts_remaining} attempts left, enough to "
            "guarantee a win if each clue is used to split the choices."
        )
    else:
        risk = (
            f"You have {analysis.attempts_remaining} attempts left, so a win "
            "isn't guaranteed; make each guess count by using the clue to cut "
            "away as many choices as possible."
        )
    return " ".join(
        sentence
        for sentence in (progress, history_note, recommendation, risk)
        if sentence
    )
