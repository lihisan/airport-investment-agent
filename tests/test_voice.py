"""Offline tests for voice input (fake Whisper client, no network)."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice import WHISPER_MODEL, transcribe  # noqa: E402


class FakeAudioClient:
    def __init__(self, text=None, error=None):
        self.calls = []
        self.text, self.error = text, error
        self.audio = NS(transcriptions=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return NS(text=self.text)


class VoiceTests(unittest.TestCase):
    def test_transcribe_sends_audio_to_whisper_and_strips_text(self):
        client = FakeAudioClient(text="  Compare LAX and SNA congestion.  ")
        self.assertEqual(transcribe(b"RIFF....", client=client), "Compare LAX and SNA congestion.")
        call = client.calls[0]
        self.assertEqual(call["model"], WHISPER_MODEL)
        self.assertEqual(call["file"][1], b"RIFF....")

    def test_empty_audio_is_rejected_without_calling_the_api(self):
        client = FakeAudioClient(text="x")
        with self.assertRaises(ValueError):
            transcribe(b"", client=client)
        self.assertEqual(client.calls, [])

    def test_silence_gives_empty_string(self):
        self.assertEqual(transcribe(b"RIFF", client=FakeAudioClient(text="   ")), "")


if __name__ == "__main__":
    unittest.main()
