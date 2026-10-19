"""Model providers. The agent loop doesn't care which model it talks to.

Each provider translates between its own API format and one small neutral shape:
    Turn(text, tool_calls, stop, assistant_message)
so switching from Groq to Claude (or anything else) is one line, not a rewrite.

Choose with the LLM_PROVIDER env var: "groq" (default) or "claude".
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from tools import TOOLS


def load_env(path=Path(__file__).parent / ".env"):
    """Read KEY=value lines from .env into the environment (.env wins over the shell)."""
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                os.environ[key.strip()] = value.strip()


load_env()


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    parse_error: str = None  # set when the model produced invalid JSON arguments


@dataclass
class Turn:
    text: str
    tool_calls: list = field(default_factory=list)
    stop: str = "end"               # "end" | "tool_use" | "refusal" | "max_tokens"
    assistant_message: dict = None  # the model's answer, in the provider's own format


class GroqLLM:
    """Groq: very fast open models, OpenAI-compatible "chat completions" format."""

    # Tried in order. Each model has its own free-tier quota, so when the first one hits its
    # daily or per-minute limit the agent keeps working on the next one (and the UI says which).
    MODELS = [os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"), "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

    def __init__(self, client=None):
        self.model = self.MODELS[0]
        if client is None:
            import groq
            client = groq.Groq()  # reads GROQ_API_KEY from env
        self.client = client
        # Same tools, OpenAI format: {"type": "function", "function": {...}}
        self.tools = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in TOOLS
        ]

    def reset(self):
        """Start each new question on the primary model again (a rate limit may have been temporary)."""
        self.model = self.MODELS[0]

    def call(self, system, messages):
        response = self._create([{"role": "system", "content": system}] + messages)
        choice = response.choices[0]
        msg = choice.message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args, err = json.loads(tc.function.arguments or "{}"), None
            except json.JSONDecodeError as e:
                args, err = {}, f"Invalid JSON arguments: {e}"
            calls.append(ToolCall(tc.id, tc.function.name, args, err))

        assistant = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        stop = "tool_use" if calls else ("max_tokens" if choice.finish_reason == "length" else "end")
        return Turn(msg.content or "", calls, stop, assistant)

    def _create(self, messages, retries=2):
        import groq
        for attempt in range(retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=self.tools,
                    tool_choice="auto", temperature=0.2,
                    # Groq's free tier counts max_tokens toward a per-request token cap, so keep it modest.
                    max_tokens=2500,
                )
            except groq.RateLimitError:
                # Quota used up for this model: switch to the next one and retry right away.
                index = self.MODELS.index(self.model)
                if index + 1 >= len(self.MODELS):
                    raise
                self.model = self.MODELS[index + 1]
                return self._create(messages, retries)
            except groq.BadRequestError as e:
                # Groq returns 400 "tool_use_failed" when the model writes a broken
                # tool call. It's usually a one-off, so simply try again.
                if "tool_use_failed" not in str(e) or attempt == retries:
                    raise

    def tool_result_messages(self, results):
        # OpenAI format: one "tool" message per result.
        return [{"role": "tool", "tool_call_id": r["id"], "content": r["output"]} for r in results]


class ClaudeLLM:
    """Anthropic Claude via the official SDK."""

    MODEL = "claude-opus-5-5"

    def __init__(self, client=None):
        self.model = self.MODEL
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.client = client

    def call(self, system, messages):
        response = self.client.beta.messages.create(
            model=self.MODEL,
            max_tokens=16000,
            system=system,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "medium"},
            # If the model declines for safety reasons, the API retries on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        stop = {"tool_use": "tool_use", "refusal": "refusal", "max_tokens": "max_tokens"}.get(response.stop_reason, "end")
        return Turn(text, calls, stop, {"role": "assistant", "content": response.content})

    def tool_result_messages(self, results):
        # Claude format: all results together in one "user" message.
        return [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r["id"], "content": r["output"], "is_error": r["is_error"]}
            for r in results
        ]}]


def make_llm(provider=None):
    provider = (provider or os.environ.get("LLM_PROVIDER", "groq")).lower()
    if provider == "claude":
        return ClaudeLLM()
    if provider == "groq":
        return GroqLLM()
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; use 'groq' or 'claude'")
