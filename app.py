import json
import random
import streamlit as st
from logic_utils import (
    check_guess,
    get_range_for_difficulty,
    parse_guess,
    update_score,
)
from rag_utils import (
    ATTEMPT_LIMITS,
    StrategyError,
    analyze_strategy,
    fallback_explanation,
    generate_explanation,
    retrieve_strategies,
)
from agent_utils import AgentError, run_strategy_agent

st.set_page_config(page_title="Glitchy Guesser", page_icon="🎮")

st.title("🎮 Game Glitch Investigator")
st.caption("An AI-generated guessing game. Something is off.")

st.sidebar.header("Settings")

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["Easy", "Normal", "Hard"],
    index=1,
)

attempt_limit = ATTEMPT_LIMITS[difficulty]

low, high = get_range_for_difficulty(difficulty)

st.sidebar.caption(f"Range: {low} to {high}")
st.sidebar.caption(f"Attempts allowed: {attempt_limit}")

if (
    "secret" not in st.session_state
    or st.session_state.get("secret_difficulty") != difficulty
):
    st.session_state.secret = random.randint(low, high)
    st.session_state.secret_difficulty = difficulty
    st.session_state.attempts = 0
    st.session_state.score = 0
    st.session_state.status = "playing"
    st.session_state.history = []
    st.session_state.guess_outcomes = []

if "attempts" not in st.session_state:
    st.session_state.attempts = 0
if "score" not in st.session_state:
    st.session_state.score = 0

if "status" not in st.session_state:
    st.session_state.status = "playing"

if "history" not in st.session_state:
    st.session_state.history = []
if "guess_outcomes" not in st.session_state:
    st.session_state.guess_outcomes = []

st.sidebar.subheader("Guess History")
if st.session_state.history:
    st.sidebar.write(st.session_state.history)
else:
    st.sidebar.caption("No valid guesses yet.")

st.subheader("Make a guess")

st.info(
    f"Guess a number between {low} and {high}. "
    f"Attempts left: {attempt_limit - st.session_state.attempts}"
)

raw_guess = st.text_input(
    "Enter your guess:",
    key=f"guess_input_{difficulty}"
)

col1, col2, col3 = st.columns(3)
with col1:
    submit = st.button("Submit Guess 🚀")
with col2:
    new_game = st.button("New Game 🔁")
with col3:
    show_hint = st.checkbox("Show hint", value=True)
# FIX: Used AI assistance so New Game generates a secret within the selected range.
if new_game:
    st.session_state.attempts = 0
    st.session_state.score = 0
    st.session_state.status = "playing"
    st.session_state.secret = random.randint(low, high)
    st.session_state.secret_difficulty = difficulty
    st.session_state.history = []
    st.session_state.guess_outcomes = []
    st.success("New game started.")
    st.rerun()

if st.button("Ask AI for Strategy"):
    public_history = [
        {"guess": guess, "outcome": outcome}
        for guess, outcome in zip(
            st.session_state.history, st.session_state.guess_outcomes
        )
    ]
    try:
        analysis = analyze_strategy(
            difficulty, (low, high), public_history
        )
    except ValueError as error:
        st.subheader("AI Strategy Coach")
        st.info("I can’t make a reliable suggestion from those clues. Start a new game to try again.")
    else:
        st.subheader("AI Strategy Coach")
        if analysis.completed:
            st.success("You found it! Enjoy the win, and start a new game when you’re ready.")
        elif analysis.attempts_remaining == 0:
            st.info("That round is over, so there are no guesses left. Start a new game to try another strategy.")
        elif analysis.recommended_guess is None or analysis.remaining_range is None:
            st.info("There isn’t an unused number left in the possible range. Start a new game for a fresh round.")
        else:
            remaining_low, remaining_high = analysis.remaining_range
            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("Recommended Next Guess", analysis.recommended_guess)
            metric2.metric(
                "Remaining Possible Range",
                f"{remaining_low}–{remaining_high}",
            )
            metric3.metric("Possible Numbers Remaining", analysis.possible_count)
            metric4.metric("Attempts Remaining", analysis.attempts_remaining)

            if analysis.attempts_remaining == 1:
                st.warning("You have one guess left. Make it count.")
            if analysis.can_guarantee_win:
                st.success(
                    "Good news—you still have enough tries to guarantee a win "
                    "if you keep narrowing the range with the clues."
                )
            else:
                st.info(
                    "There are more possible numbers than your remaining guesses "
                    "can guarantee. Keep choosing near the middle for your best chance."
                )

            st.markdown("### Your Strategy")
            try:
                sources = retrieve_strategies(
                    difficulty, (low, high), public_history
                )
            except StrategyError as error:
                sources = []
                st.caption("I’m using a quick game tip while the coach is unavailable.")

            if sources:
                explanation = generate_explanation(
                    difficulty,
                    (low, high),
                    public_history,
                    sources,
                )
            else:
                explanation = fallback_explanation(difficulty, analysis)
            st.write(explanation)

            st.markdown("### What To Do Next")
            high_text = (
                f"{analysis.higher_range[0]}–{analysis.higher_range[1]}"
                if analysis.higher_range
                else "no numbers would remain"
            )
            low_text = (
                f"{analysis.lower_range[0]}–{analysis.lower_range[1]}"
                if analysis.lower_range
                else "no numbers would remain"
            )
            st.write(
                f"If {analysis.recommended_guess} is **Too High**, the range "
                f"would become {high_text}. If it’s **Too Low**, it would "
                f"become {low_text}."
            )
            with st.expander("View Retrieved Strategy Sources"):
                if sources:
                    for source in sources:
                        st.markdown(
                            f"**{source['title']}** (`{source['id']}`)"
                        )
                        st.write(source["text"])
                else:
                    st.caption("Strategy sources are unavailable right now.")

st.divider()
st.subheader("AI Strategy Agent")
if st.button("Ask AI Agent"):
    public_history = [
        {"guess": guess, "outcome": outcome}
        for guess, outcome in zip(
            st.session_state.history, st.session_state.guess_outcomes
        )
    ]
    try:
        with st.spinner("The agent is choosing and using strategy tools..."):
            agent_result = run_strategy_agent(
                difficulty=difficulty,
                allowed_range=(low, high),
                history=public_history,
            )
    except AgentError as error:
        st.warning(f"The strategy agent could not analyze this game: {error}")
    else:
        agent_analysis = agent_result.analysis
        if agent_analysis.completed:
            st.success(
                "You already found the number. Start a new game whenever you’re ready."
            )
        elif agent_analysis.attempts_remaining == 0:
            st.info("This round is complete; there are no guesses remaining.")
        elif (
            agent_analysis.recommended_guess is None
            or agent_analysis.remaining_range is None
        ):
            st.info(
                "There are no unused possible guesses left. Start a new game for another round."
            )
        else:
            range_low, range_high = agent_analysis.remaining_range
            agent_cols = st.columns(4)
            agent_cols[0].metric(
                "Recommended Guess", agent_analysis.recommended_guess
            )
            agent_cols[1].metric(
                "Possible Range", f"{range_low}–{range_high}"
            )
            agent_cols[2].metric(
                "Attempts Left", agent_analysis.attempts_remaining
            )
            agent_cols[3].metric(
                "Possible Answers", agent_analysis.possible_count
            )
            st.markdown("### Your Game Analysis")
            st.write(agent_result.explanation)
            if agent_result.explanation_source == "fallback":
                fallback_notice = "The explanation above is a deterministic fallback."
                if agent_result.stopped_reason:
                    fallback_notice += f" {agent_result.stopped_reason}"
                st.caption(fallback_notice)

        with st.expander("Agent Decision Trace"):
            if agent_result.trace:
                for step_number, entry in enumerate(agent_result.trace, start=1):
                    st.markdown(f"**Step {step_number}: {entry.tool}**")
                    st.caption(f"Why: {entry.reason or 'Not provided'}")
                    if entry.warning:
                        st.warning(entry.warning)
                    st.markdown("Tool result")
                    try:
                        result_data = json.loads(entry.result)
                    except json.JSONDecodeError:
                        st.write(entry.result)
                    else:
                        st.json(result_data, expanded=False)
            else:
                st.caption("No tool calls were needed for this game state.")

if st.session_state.status != "playing":
    if st.session_state.status == "won":
        st.success("You already won. Start a new game to play again.")
    else:
        st.error("Game over. Start a new game to try again.")
    st.stop()

if submit:
    ok, guess_int, err = parse_guess(raw_guess)

    if not ok:
        st.error(err)
    elif not low <= guess_int <= high:
        st.error(f"Enter a number between {low} and {high}.")
    else:
        st.session_state.attempts += 1
        st.session_state.history.append(guess_int)

        if st.session_state.attempts % 2 == 0:
            secret = str(st.session_state.secret)
        else:
            secret = st.session_state.secret

        outcome, message = check_guess(guess_int, secret)
        st.session_state.guess_outcomes.append(outcome)

        distance = abs(guess_int - st.session_state.secret)
        range_size = high - low
        if distance <= range_size * 0.1:
            temperature = "🔥 Hot"
        elif distance <= range_size * 0.25:
            temperature = "🌡️ Warm"
        else:
            temperature = "❄️ Cold"
        st.info(f"Guess temperature: {temperature}")

        if show_hint:
            st.warning(message)

        st.session_state.score = update_score(
            current_score=st.session_state.score,
            outcome=outcome,
            attempt_number=st.session_state.attempts,
        )

        if outcome == "Win":
            st.balloons()
            st.session_state.status = "won"
            st.success(
                f"You won! The secret was {st.session_state.secret}. "
                f"Final score: {st.session_state.score}"
            )
        else:
            if st.session_state.attempts >= attempt_limit:
                st.session_state.status = "lost"
                st.error(
                    f"Out of attempts! "
                    f"The secret was {st.session_state.secret}. "
                    f"Score: {st.session_state.score}"
                )

st.divider()
