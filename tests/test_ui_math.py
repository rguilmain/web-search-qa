from __future__ import annotations

from web_search_qa import ui


def test_latex_delimiters_cover_display_and_inline_math() -> None:
    pairs = {
        (delimiter["left"], delimiter["right"], delimiter["display"])
        for delimiter in ui._LATEX_DELIMITERS
    }
    assert ("$$", "$$", True) in pairs
    assert ("\\[", "\\]", True) in pairs
    assert ("\\(", "\\)", False) in pairs
    # Bare $ is deliberately not a Gradio delimiter; it is normalized first.
    assert ("$", "$", False) not in pairs


def test_normalize_math_converts_bare_dollars() -> None:
    assert ui._normalize_math(r"Here is $E=mc^2$ today.") == r"Here is \(E=mc^2\) today."
    assert ui._normalize_math(r"$a$ and $b$") == r"\(a\) and \(b\)"


def test_normalize_math_preserves_existing_delimiters() -> None:
    display = r"$$\int_0^1 x^2\,dx = \frac13$$"
    inline = r"\(a^2 + b^2 = c^2\)"
    bracket = r"\[x_{n+1} = x_n\]"
    text = f"{display} {inline} {bracket}"
    assert ui._normalize_math(text) == text


def test_normalize_math_ignores_currency_and_escaped_dollars() -> None:
    assert ui._normalize_math("costs $5 and $10") == "costs $5 and $10"
    assert ui._normalize_math(r"price \$5") == r"price \$5"
    assert ui._normalize_math("costs $ 5") == "costs $ 5"
    assert ui._normalize_math("costs 5 $") == "costs 5 $"


def test_normalize_math_requires_a_closed_pair() -> None:
    assert ui._normalize_math("only $x") == "only $x"
