import re


# FIX: Used AI assistance to make the secret number follow the selected
# difficulty range.
def get_range_for_difficulty(difficulty: str):
    """Return the inclusive secret-number range for a difficulty level.

    Args:
        difficulty: The selected difficulty name.

    Returns:
        A ``(low, high)`` tuple containing the inclusive range. Unknown
        difficulty names use the Hard range.
    """
    if difficulty == "Easy":
        return 1, 20
    if difficulty == "Normal":
        return 1, 50
    if difficulty == "Hard":
        return 1, 100
    return 1, 100


def parse_guess(raw: str | None):
    """Parse raw user input into an integer guess.

    Args:
        raw: The text entered by the player.

    Returns:
        A tuple of ``(ok, guess, error_message)``. ``ok`` indicates whether
        parsing succeeded, ``guess`` contains the integer when valid, and
        ``error_message`` contains feedback when invalid.
    """
    if raw is None or raw.strip() == "":
        return False, None, "Enter a guess."

    value_text = raw.strip()
    if re.fullmatch(r"[+-]?\d+", value_text) is None:
        return False, None, "That is not a number."

    return True, int(value_text), None

# FIX: Used AI assistance to refactor check_guess and correct the reversed
# higher/lower hints.


def check_guess(guess: int, secret: int | str):
    """Compare a guess with the secret and return the outcome and message.

    Args:
        guess: The player's guess.
        secret: The secret value to compare against.

    Returns:
        A tuple containing an outcome (``"Win"``, ``"Too High"``, or
        ``"Too Low"``) and the corresponding user-facing message.
    """
    guess = int(guess)
    secret = int(secret)

    if guess == secret:
        return "Win", "🎉 Correct!"

    if guess > secret:
        return "Too High", "📉 Go LOWER!"
    return "Too Low", "📈 Go HIGHER!"


def update_score(current_score: int, outcome: str, attempt_number: int):
    """Update the player's score based on an outcome and attempt number.

    Args:
        current_score: The player's score before the update.
        outcome: The result of the current guess.
        attempt_number: The one-based number of the current attempt.

    Returns:
        The updated score.
    """
    if outcome == "Win":
        return current_score + max(10, 100 - 10 * (attempt_number + 1))

    if outcome == "Too High":
        return current_score + (5 if attempt_number % 2 == 0 else -5)

    if outcome == "Too Low":
        return current_score - 5

    return current_score
