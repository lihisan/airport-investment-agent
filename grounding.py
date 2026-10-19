"""Hallucination detector: is every number in the agent's answer backed by a tool result?

The rule "every number comes from a tool" lives in the prompt; this checks it in code. We pull the
numbers out of the reply (4.9%, 19,779, 26.6M, 1.2k ...) and look for each one among the numbers in
the tool outputs, allowing for the rounding the reply shows (82.6% matches 0.8255) and for fractions
written as percentages. Numbers from the system prompt or the user's question are allowed too.

A number the model computed itself (e.g. "4.2 points higher") is reported as unverified as well:
it may be right, but it was not produced by our code, so a human should check it.
"""

import json
import re
from dataclasses import dataclass

SCALES = {"k": 1e3, "K": 1e3, "M": 1e6, "million": 1e6, "B": 1e9, "billion": 1e9}

NUMBER_RE = re.compile(
    r"(?<![\w.\-])"                                   # not part of a word, code or date ("gpt-oss-120b", "2025-08")
    r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"             # 19,779 / 4.9 / 2025
    r"(?:\s?(%|k|K|M|B)(?![A-Za-z])|\s(million|billion)\b)?"
    r"(?![\w\-])"
)


THIN_SPACES = "\u202f\u2009\u00a0"  # narrow, thin and no-break spaces models use inside numbers


def normalize_text(text):
    """'6\u202f219' -> '6,219' (space as thousands separator) and other odd spaces -> ' '."""
    text = re.sub(f"(?<=\\d)[{THIN_SPACES}](?=\\d{{3}}(?!\\d))", ",", text)
    return re.sub(f"[{THIN_SPACES}]", " ", text)


@dataclass
class Number:
    text: str
    value: float       # in plain units: "4.9%" -> 4.9, "26.6M" -> 26600000
    tolerance: float   # half a unit of the precision shown
    percent: bool


def extract_numbers(text):
    numbers = []
    for m in NUMBER_RE.finditer(normalize_text(text)):
        digits, unit = m.group(1), m.group(2) or m.group(3)
        mantissa = float(digits.replace(",", ""))
        scale = SCALES.get(unit, 1.0)
        if "." in digits:
            step = 10 ** -len(digits.split(".")[1])
        else:  # "6,200" is a rounded number: allow rounding to its trailing zeros
            plain = digits.replace(",", "")
            step = 10 ** (len(plain) - len(plain.rstrip("0"))) if plain != "0" else 1
        numbers.append(Number(m.group(0).strip(), mantissa * scale, step * scale / 2 + 1e-9, unit == "%"))
    return numbers


def _is_trivial(n):
    """Small counts ("top 3", "rank 1") and calendar years are not claims worth checking."""
    if n.percent or n.text[-1].isalpha():
        return False
    if n.value.is_integer() and "." not in n.text:
        return n.value <= 10 or 1900 <= n.value <= 2100
    return False


def _tool_values(outputs):
    values = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            values.extend([float(x), float(x) * 100])  # fractions may be shown as percentages
        elif isinstance(x, str):
            values.extend(n.value for n in extract_numbers(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for out in outputs:
        try:
            walk(json.loads(out))
        except (TypeError, ValueError):
            walk(str(out))
    return values


def ungrounded_numbers(reply, tool_outputs, allowed_text=""):
    """Numbers in the reply that appear neither in the tool outputs nor in allowed_text."""
    known = _tool_values(tool_outputs) + [n.value for n in extract_numbers(allowed_text)]
    flagged = []
    for n in extract_numbers(reply):
        if _is_trivial(n) or n.text in flagged:
            continue
        if not any(abs(v - n.value) <= n.tolerance for v in known):
            flagged.append(n.text)
    return flagged
