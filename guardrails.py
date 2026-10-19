"""Guardrails: checks that run *around* the model, in plain code.

The prompt asks the model to behave; guardrails make sure it does even when it doesn't.
Here: catch obvious prompt-injection attempts before they ever reach the model.
The other safety rules live in code too: every number comes from a tool (tools.py),
and the agent loop caps the number of tool calls per question (agent.py).
"""

import re

INJECTION_PATTERNS = [
    r"ignore (all|any|the|your)? ?(previous|prior|above) instructions",
    r"system prompt",
    r"you are now",
    r"developer mode",
]


def looks_like_injection(text):
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)
