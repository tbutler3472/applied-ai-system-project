# Model Card: Game Glitch Investigator

## Model name and deployment

The language-model component is the locally deployed **llama3.2:3b** model served by Ollama at `http://localhost:11434/api/generate`. The app does not send game data to a hosted API. Python uses Ollama as an optional explanation and tool-selection component; Ollama is not the source of truth for game calculations.

The RAG Strategy Coach and AI Strategy Agent use the same model, but with different prompts. The coach asks for a short explanation after Python retrieval. The agent asks the model to choose a limited set of tool names and later asks for a validated summary.

## Intended use

This system is designed for a player using a number-guessing game and for a CodePath project demonstration. It can:

- recommend an unused balanced guess;
- explain the remaining range and attempts;
- retrieve a small set of strategy documents; and
- record which agent tools it selected and why.

It is not designed to handle unrelated questions, real-world decision-making, or hidden game information. It should be run locally by a student or evaluator with Ollama available if AI features are being tested.

## Inputs and outputs

Public inputs are the difficulty, allowed range, guess history, and outcomes (`Too High`, `Too Low`, or `Win`). The app does not pass the secret number to Ollama or to the strategy tools.

Outputs include:

- remaining range and possible-count values;
- recommended guess;
- attempts remaining;
- guarantee status;
- retrieved strategy sources;
- an Ollama-generated or deterministic fallback explanation; and
- an ordered agent decision trace.

## RAG use

The local knowledge base is `data/guessing_strategies.json`. It contains entries for midpoint strategy, binary search, Too High and Too Low narrowing, repeated guesses, and Easy, Normal, and Hard play. Retrieval combines keyword matching with difficulty filtering and returns up to three ranked entries.

Retrieved text is included in the coach prompt. The model may explain how to apply it, but it does not define the numeric analysis or selected guess.

## Agent tool selection and execution

The agent gives Ollama five known tool names. Ollama returns JSON containing tool names, empty argument objects, and reasons. Python parses the plan and rejects unknown tools, malformed JSON, disallowed arguments, duplicate fields other than repeated `tools` lists, or empty/unsafe reasons.

Each tool is a fixed read-only Python function. The app does not allow Ollama to write code or execute system commands. Tool results are returned in a separate final-explanation request if the initial response does not already contain an acceptable summary. The agent can run at most five tools and has a 30-second overall time budget.

## Response validation

For the RAG coach, Ollama text is rejected if it contains numbers or internal implementation terms, speculates that the player is close to the answer, or uses unsupported guarantees. For the agent, a second validator checks attempt counts, ranges, history outcomes, repeated guesses, guarantee claims, efficiency claims, and recommended guesses.

A valid Ollama plan or summary is not trusted just because it is well formatted. It must agree with the public game facts calculated by Python.

## Failures and timeouts

If Ollama is unavailable, the app uses a deterministic fallback. The RAG coach catches request and response errors. The agent separately records tool-selection timeout, final-explanation timeout, malformed or unknown tool plans, validation rejection, repeated tools, and overall budget exhaustion.

In real captured runs, tool execution has succeeded while the final explanation was rejected or timed out. This is handled intentionally by keeping the tool results and showing fallback text. It should not be reported as a fully successful Ollama explanation.

## Testing and evaluation

The automated suite is pytest-based. It covers:

- all difficulty ranges and attempt limits;
- higher/lower and win outcomes;
- remaining-range calculations;
- repeated guesses and midpoints;
- RAG retrieval and prompt construction;
- valid, malformed, unknown, and unsafe tool plans;
- Ollama success, rejection, connection failure, and timeout mocks;
- deterministic fallbacks; and
- Streamlit state behavior.

The latest executed full run collected 67 tests, and all 67 passed. Automated tests mock Ollama; they are not evidence that the local model is available in every environment.

`evaluate.py` provides 10 lightweight deterministic PASS/FAIL checks for range and guess calculations, tool parsing, unknown and unsafe tools, fallback behavior, difficulty handling, and RAG retrieval. It exits nonzero if a required check fails. `python evaluate.py --ollama` optionally attempts a real local request but does not turn that smoke test into a required pass/fail check.

## Limitations and possible errors

- Ollama latency can vary, and model loading or CPU-only inference can cause final-explanation timeouts.
- A model can select a valid tool for the wrong reason or omit useful tools. Guardrails record this and keep Python analysis authoritative.
- Retrieval is simple keyword matching, not semantic vector retrieval.
- The knowledge base is English and limited to the small number-guessing strategy domain.
- Ollama text can be rejected even when the tool results are correct.
- The app is not checked for every device size or browser. Streamlit UI tests cover selected behavior, not a complete manual browser review.

## Human oversight

A player must still interpret the clues and make the guess. The app displays fallback text whenever model explanations are unavailable or unsafe. Numeric recommendations are reviewed by Python against the range and history. Agent trace output lets a student or evaluator see exactly what the model selected and whether the tool result passed validation.

## AI collaboration and development reflection

I used Copilot and ChatGPT as coding helpers during development. Git history and the current files support these examples:

- **Accepted:** Copilot assistance was used to move secret-range and higher/lower comparison logic into `logic_utils.py`, keep secret numbers inside selected difficulty ranges, and add docstrings and PEP 8 cleanup. I verified these changes with pytest and regression tests rather than assuming the generated code was correct.
- **Accepted:** Copilot-generated test ideas were used for exact matches, far-apart guesses, and near-boundary guesses. The tests remain in `tests/test_game_logic.py`, and I reviewed their behavior against `check_guess`.
- **Revised:** Copilot contributed to the RAG and agent implementation, including public-state tool definitions and response validation. I kept the numeric calculations in Python and adjusted the tool workflow so the model selects from fixed tools rather than supplying calculations.
- **Revised after test failures:** My original reflection says an AI suggestion was to change tests before I ran them. Three of the four tests failed; I then used AI to revise the old tests to match the `check_guess` function, and all four passed. Looking back, passing tests alone did not prove the behavior was right. I learned to check that each test still verifies the intended behavior instead of only changing tests until they pass.
- **Reviewed, not blindly trusted:** Copilot can produce extra changes or implementation details that a prompt did not request. For this project, I kept changes tied to working game behavior and used the existing test suite as evidence.

I learned that AI assistance can help create an implementation or test idea quickly, but it does not prove that the result is correct. The main way I handled that was to run targeted tests, check public-state behavior, and use local fallbacks where model output could be wrong.

## Future improvements

If I continue this project, I would work on improving Ollama response speed and
reducing timeouts, expand the strategy knowledge base and possibly try semantic
retrieval, test more real gameplay situations instead of relying mostly on
mocked model responses, and make the final model-generated explanations more
consistent.
