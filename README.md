# Game Glitch Investigator

A Streamlit number-guessing game with difficulty settings, higher/lower hints,
Hot/Warm/Cold feedback, score, guess history, and an optional local AI Strategy
Coach.

## Run the game

1. Install Python dependencies with `pip install -r requirements.txt`.
2. To use the coach, start Ollama and make sure `llama3.2:3b` is available
   locally (`ollama pull llama3.2:3b`).
3. Start the app with `python -m streamlit run app.py`.

The game itself works without Ollama. If the local model is unavailable or
returns an unusable answer, the coach shows a friendly local explanation instead.

## AI Strategy Coach

Choose **Ask AI for Strategy** to get a recommendation based on the selected
difficulty, the original range, and public guess outcomes. The coach narrows
the remaining range from Too High/Too Low feedback, counts possible numbers,
skips guesses already made, and selects an unused number nearest the midpoint.
It also checks remaining attempts against how many possibilities can be
resolved in that many higher/lower guesses, then shows the ranges that either
possible result would leave.

For its explanation, the app retrieves relevant strategy entries from
`data/guessing_strategies.json` using lightweight keyword matching. The local
Ollama model applies those tips to the player's situation. The deterministic
recommendation and bounds remain separate from the model-written explanation;
responses with contradictory claims or invented numeric details are replaced
with a natural local fallback. **View Retrieved Strategy Sources** keeps the
source titles, IDs, and text available for demonstration.

Only public information is used by retrieval and coaching: difficulty, allowed
bounds, guesses, outcomes, and remaining attempts. The secret number is never
sent to the model or strategy helpers.

## Tests

Run the full test suite with:

```shell
python -m pytest -v
```

Ollama is mocked in automated tests, so a running model is not required.
