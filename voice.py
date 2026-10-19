"""Voice input: turn a recorded question into text with Whisper on Groq.

The transcribed text then goes through the normal chat path, so the agent, tools and scoring
don't know (or care) whether the question was typed or spoken.
"""

import llm  # noqa: F401  (loads .env so GROQ_API_KEY is available)

WHISPER_MODEL = "whisper-large-v3-turbo"


def transcribe(audio_bytes, client=None, filename="question.wav"):
    """Return the text spoken in the audio ("" if nothing was said)."""
    if not audio_bytes:
        raise ValueError("No audio recorded")
    if client is None:
        import groq
        client = groq.Groq()
    result = client.audio.transcriptions.create(file=(filename, audio_bytes), model=WHISPER_MODEL)
    return (result.text or "").strip()
