# AI Interactions

This file records how I used AI while building the project. I am separating real local-model behavior from mocked tests and Copilot-assisted coding help. It does not describe any interaction as successful if the code or the model run did not show that result.

## Real agent traces

The [agent trace log](logs/agent_traces.md) contains captured Ollama tool
selections, tool results, and examples where the final explanation used a
deterministic fallback.

## Copilot-assisted development

### Guess history

I asked for a Guess History feature that would keep valid guesses, reset for a new game or difficulty change, and ignore invalid input. The assistant changed `app.py` to use the existing session state and list valid guesses in the sidebar.

I verified this myself with Streamlit tests and input checks. Invalid decimal and out-of-range guesses leave the attempt and history unchanged. I kept the implementation because the tests pass and it fits the existing game flow.

### Higher/lower hint bug

Copilot helped identify and fix the higher/lower hint problem by moving the
comparison into `logic_utils.py`. The expected behavior is that a guess lower
than the secret returns Too Low, and a guess higher than the secret returns
Too High.

My original reflection records that an AI suggestion was to change the tests.
When I ran them, three out of four failed. I then used AI to revise the old
tests to match the `check_guess` function, and four tests passed. That experience
taught me to check that a test still represents the intended behavior, not just
whether it passes.

### RAG and agent implementation

I used Copilot to draft and revise the RAG retrieval helpers, agent tool
definitions, Ollama prompts, and response validation. The current workflow
passes public game state and only exposes the fixed read-only tools.

The parts I kept now appear in:

- `rag_utils.py` for public-state analysis, retrieval, and fallback explanations;
- `agent_tools.py` for fixed read-only tools;
- `agent_validation.py` for JSON and public-fact checks;
- `agent_prompts.py` for bounded tool and explanation prompts; and
- `agent_utils.py` for trace collection and timeout handling.

I revised the workflow so tool arguments are empty and Python calculates all
numeric results rather than asking the model to supply them.

### Tests

Copilot helped identify boundary cases for the game and suggested tests for exact matches, far-away guesses, and one-number-off guesses. I reviewed the tests, kept the valid cases in `tests/test_game_logic.py`, and ran them under pytest.

It also helped write mocked Ollama tests for success, malformed JSON, unknown tools, unsafe arguments, connection failures, and timeouts. Those tests verify code paths but are not evidence of a real model response.

### Style and documentation

Copilot suggestions were used to add docstrings and clean PEP 8 formatting in `logic_utils.py`. I ran the test suite after the changes and kept the behavior-focused code.

## RAG retrieval examples

The custom JSON strategy knowledge base gives the coach strategy notes that
match the current difficulty and clues. For an Easy opening, retrieval can
return Easy, midpoint, and binary-search tips. After a Too High clue, it can
return the matching advice to lower the upper bound. This gives Ollama
situation-specific reference text in addition to generic guessing advice. It
is a qualitative reason retrieval may make advice more relevant; I did not run
a baseline comparison or measure an improvement.

### Too High clue

Public input: Hard, range 1–100, and 70 → Too High.

Retrieval returned the Hard difficulty source and the Too High narrowing source. The retrieved text says to lower the upper bound to 69 and choose within the smaller range.

The deterministic result leaves range 1–69 and recommends 35. Ollama is allowed
to explain the clue only after Python has calculated these facts.

### Easy opening

Public input: Easy, range 1–20, no guesses.

Retrieval returned Easy difficulty, midpoint, and binary-search sources. The deterministic result recommends 10.

## Successful Ollama-generated explanations

The mock tests contain examples where Ollama-style text passes validation and is shown as `explanation_source="ollama"`. These are automated test examples, not live model claims.

The live traces in `logs/agent_traces.md` describe real local-model tool runs. In captured runs, Ollama often executed tools successfully but did not always produce a usable final explanation.

## Successful tool execution followed by fallback

Real captured Easy and Hard runs show Ollama selecting and executing tools. In at least one Easy live run, the model's final text mentioned implementation tools and was rejected by the player-facing response checks. The app kept the deterministic range and guess and showed a local fallback.

The live runs selected different combinations and orders of tools depending on
the game state. Some used four tools and others used five. This demonstrates
model-selected tool execution, while the final explanation may still be
rejected or replaced.

## Complete model request failures

A timeout during selection is covered by mocked automated tests; I do not have
a separate captured live selection-timeout trace. In a captured live Hard run,
the final explanation timed out after the tools had run, and the app kept the
tool trace while using fallback text.

The existing agent tests also cover a 30-second overall budget, malformed JSON, empty plans, unknown tools, and repeated tool selections. These test cases are not presented as successful live model conversations.

## Mocked or automated examples

Automated pytest examples prove that:

- valid tool plans execute and pass results to a final prompt;
- unknown tools stop safely;
- arguments must be empty;
- malformed JSON uses fallback;
- incorrect final text is rejected; and
- request timeouts produce fallback explanations.

These tests use `unittest.mock.Mock` or `requests.Timeout`. They show that the app handles expected paths but do not prove Ollama's live output for those exact scenarios.
