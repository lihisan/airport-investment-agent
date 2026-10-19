"""The agent loop: the heart of the project.

An "agent" is just a loop around the model:
    1. send the conversation + the list of tools to the model
    2. if the model answers with text -> show it to the analyst, done
    3. if the model asks to use a tool -> run the tool, send back the result, go to 1
The model decides *what* to do; our code decides *whether it is allowed* and does it.
Which model (Groq / Claude) is decided in llm.py.
"""

import json
import logging

from guardrails import looks_like_injection
from llm import make_llm
from prompts import SYSTEM_PROMPT
from tools import Backend

MAX_STEPS = 10  # safety net: never let one question trigger an endless tool loop

log = logging.getLogger("agent")


class Agent:
    def __init__(self, backend=None, llm=None):
        self.backend = backend or Backend()
        self.llm = llm or make_llm()
        self.messages = []  # the full conversation, the model's only "memory"
        self.trace = []     # every tool call, for debugging and evals

    def reply(self, user_text):
        """Handle one user message and return the agent's answer."""
        log.info("USER: %s", user_text)

        if looks_like_injection(user_text):
            # Input guardrail: don't even send this to the model.
            return "I can only help with US airport and aviation investment analysis. What would you like to know?"

        self.messages.append({"role": "user", "content": user_text})

        for _ in range(MAX_STEPS):
            turn = self.llm.call(SYSTEM_PROMPT, self.messages)
            # Always keep the model's full answer (text + tool calls) in the history.
            self.messages.append(turn.assistant_message)

            if turn.stop == "tool_use":
                results = self._run_tools(turn.tool_calls)
                self.messages.extend(self.llm.tool_result_messages(results))
                continue  # back to the model with the tool results

            if turn.stop == "refusal":
                return "Sorry, I can't help with that. I can analyze US airports, traffic and congestion."

            text = turn.text.strip()
            if turn.stop == "max_tokens":
                text += "\n[answer was cut off]"
            log.info("AGENT: %s", text)
            return text

        return ("I couldn't complete this analysis within the step limit. "
                "Try asking a narrower question (e.g. one airport or one region).")

    def _run_tools(self, tool_calls):
        """Run every tool call the model asked for and collect the results."""
        results = []
        for call in tool_calls:
            if call.parse_error:
                output, is_error = json.dumps({"error": call.parse_error}), True
            else:
                output, is_error = self.backend.run(call.name, call.input)
            self.trace.append({"tool": call.name, "input": call.input, "error": is_error, "output": output})
            log.info("TOOL %s(%s) -> %s", call.name, json.dumps(call.input, ensure_ascii=False), output[:500])
            # "id" links the result back to the model's request.
            results.append({"id": call.id, "output": output, "is_error": is_error})
        return results
