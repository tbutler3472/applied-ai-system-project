# AI Strategy Agent Execution Traces

The following traces were captured from live requests to the locally installed
`llama3.2:3b` model. The game state shown is public; no secret number was used.
The model selected different combinations and orders of tools for the different
game states; some runs used four tools and others used five. The final
explanations were sometimes rejected by response checks or timed out, so the
app displayed its deterministic local fallback in those cases.

## Scenario 1: New game

- Public game state: Easy, allowed range 1–20, no guesses yet.
- Tool choices, in order:
  1. `evaluate_attempts` — “Ensure a win is still guaranteed before moving on to the next step.”
     Result: 20 possibilities, 6 attempts remaining, guarantee possible.
  2. `recommend_next_guess` — “Analysis shows the remaining range is narrow enough to suggest a single next guess to maximize chances of winning.”
     Result: remaining range 1–20; recommendation 10.
  3. `review_guess_history` — “Need to summarize previous clues for better analysis.”
     Result: no guesses or repeats.
  4. `analyze_range` — “Previous results provide sufficient information for analysis.”
     Result: range 1–20; 20 possibilities; not completed.
  5. `retrieve_strategy` — “No relevant strategy advice available.”
     Result: Easy difficulty, midpoint, and binary-search strategies were retrieved.
- Final deterministic recommendation: guess 10, with range 1–20 and 6 attempts
  remaining.
- Final response: the model's completion text was rejected by the safety and
  usefulness check. The local fallback shown to the player was: “Start with 10
  to split the range into smaller choices. Use the higher or lower clue to
  narrow things down with each turn. You still have enough tries to guarantee
  a win.”

## Scenario 2: Several higher/lower clues

- Public game state: Hard, allowed range 1–100; 60 Too High, 30 Too Low, and
  45 Too Low.
- Tool choices, in order:
  1. `evaluate_attempts` — “Assess current game state and available attempts to predict win feasibility.”
     Result: 14 possibilities, 2 attempts remaining, a guaranteed win is not possible.
  2. `review_guess_history` — “Understanding previous guesses is essential to refine the range.”
     Result: no repeated guesses; the second and third clues each narrowed the
     possibilities by at least half.
  3. `analyze_range` — “Range is too large to analyze with current clues.”
     Result: range 46–59; 14 possibilities; not completed.
  4. `recommend_next_guess` — “Next-guess tool selection based on remaining range and previous results.”
     Result: remaining range 46–59; recommendation 52.
  5. `retrieve_strategy` — “Next step would be to review the history and adjust our strategy based on previous guesses.”
     Result: Hard-difficulty and Too High/Too Low range-narrowing strategies were retrieved.
- Final deterministic recommendation: guess 52, with range 46–59 and 2
  attempts remaining. A guaranteed win is no longer possible.
- Final response: the model's completion text was rejected by the safety and
  usefulness check. The local fallback shown to the player was: “The clues have
  narrowed down the choices, but there are more possibilities than your
  remaining tries can guarantee. Choose near the middle and use the next clue
  to cut away as many options as possible.”

## How the sequences differ

The new-game run checked attempt feasibility and selected a midpoint before
reviewing empty history. The run with several clues first assessed the limited
attempts and reviewed the actual clue efficiency, then calculated the narrowed
range and next guess. Both runs retrieved strategies, but selected different
relevant documents for their difficulty and situation.

## Scenario 3: Easy opening (live run after agent refinements)

- Public game state: Easy, allowed range 1–20, no guesses yet.
- Tool choices, in order:
  1. `analyze_range` — “Analyzing the range helps to determine a more focused next guess.”
     Result: range 1–20, 20 possible numbers.
  2. `recommend_next_guess` — no reason was provided.
     Result: recommended guess 10 in range 1–20.
  3. `evaluate_attempts` — “Evaluate remaining attempts and win guarantee.”
     Result: 6 attempts, 20 possible numbers, win can still be guaranteed.
  4. `retrieve_strategy` — no reason was provided.
     Result: Easy difficulty, midpoint, and binary-search strategies.
- The model finished after four tool executions; history review was not
  offered because there were no guesses to review. Its six-sentence final
  response named an implementation tool and was rejected by the player-facing
  response checks.
- Final recommendation: 10. The deterministic fallback shown was: “On Easy,
  there are 20 possible numbers, from 1 to 20. Start with 10 to split the
  choices into smaller groups. Use each higher-or-lower clue to narrow the
  range; you have 6 attempts and should use them carefully. You still have
  enough tries to guarantee a win with balanced guesses.”

## Scenario 4: Hard endgame after uneven guesses (live run after agent refinements)

- Public game state: Hard, allowed range 1–100; 20 Too Low, 72 Too Low,
  98 Too High, and 95 Too High.
- Tool choices, in order:
  1. `evaluate_attempts` — checked whether a win was still guaranteed with one
     attempt. Result: 22 possibilities, 1 attempt, no guarantee.
  2. `review_guess_history` — “Identifying repeated clues helps refine the range and improve the next guess.”
     Result: no repeated guesses; 3 of 4 clues narrowed the possibilities by
     less than half.
  3. `analyze_range` — returned range 73–94 with 22 possible numbers.
  4. `recommend_next_guess` — returned 83, nearest unused integer to the
     midpoint.
  5. `retrieve_strategy` — retrieved Hard-difficulty and higher/lower
     narrowing strategies.
- After all five tool executions, the completion request timed out. The
  personalized local fallback was shown; it explained the narrowed range, the
  inefficient clues, the final informed guess, the lack of a guaranteed win,
  and that midpoint selection does not improve the odds on a single final
  guess.

The later Easy opening and Hard endgame runs also used different sequences. In
all scenarios, deterministic Python results remained authoritative for
displayed bounds and guesses.

## What these traces prove

- **Successful Ollama tool selection:** The live runs above show the model
  returning tool plans and the Python workflow executing named read-only tools.
- **Successful tool execution followed by fallback:** The Easy opening run
  completed tool execution, but the final text named implementation details and
  failed validation. It was replaced with deterministic text. The same pattern
  appears in the Hard live run where the completion timed out.
- **Complete model request failure:** The Hard endgame completion request timed
  out after tools ran. Selection-time timeout behavior is also implemented and
  covered by mocked tests, but it is not claimed as a separate live run here.
- **Automated or mocked behavior:** The pytest suite uses mocked Ollama replies
  and `requests.Timeout` objects for malformed plans, unknown tools, unsafe
  arguments, rejected summaries, and fallback paths. Those examples verify the
  code but are not additional live-model transcripts.

## Validation failures in live and mocked runs

A live Easy opening summary was rejected because it mentioned an implementation
tool instead of staying within player-facing strategy language. A live Hard
endgame request timed out during the final explanation. In automated tests,
additional summaries are rejected for false guarantee claims, invented numbers,
wrong remaining-attempt counts, repeated-guess claims, and hidden-answer
speculation.

## Copilot-assisted decisions in the traces

Copilot helped draft the tool schema and trace display. I revised the design so
Ollama could supply only tool names, reasons, and empty arguments. Python owns
all calculations and the trace preserves validation warnings rather than hiding
model mistakes.
