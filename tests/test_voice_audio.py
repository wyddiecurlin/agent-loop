# AI_OWNED
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "websockets>=13", "certifi", "numpy>=1.26", "sounddevice>=0.5", "onnxruntime>=1.17"]
# ///
"""Host voice adapter regressions, with fake sockets and speakers; no agent, mic or API.

    uv run tests/test_voice_audio.py
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from websockets.exceptions import ConnectionClosedError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice import client
from voice.speech import Clips, DEFAULT_INSTRUCTIONS, STT, TTS, TTS_RATE, stt_language, tts_language


def event(kind, **fields):
    return json.dumps({"type": kind, **fields})


def audio_events(text="All done. Another sentence.", pcm=b"\0\0" * 100):
    return [event("audio.start", sentence_text=text, format="pcm", sample_rate=TTS_RATE),
            pcm, event("audio.done", total_bytes=len(pcm)), event("session.done")]


class Socket:
    def __init__(self, events):
        self.events = iter(events)
        self.sent = []
        self.closed = False

    def send(self, msg):
        self.sent.append(json.loads(msg))

    def recv(self, timeout):
        msg = next(self.events)
        if isinstance(msg, Exception):
            raise msg
        return msg

    def close(self):
        self.closed = True


class Sink:
    def __init__(self, offset=0):
        self.enqueued = self.consumed = offset
        self.chunks = []
        self.stopped = False

    def play(self, pcm):
        self.chunks.append(pcm)
        self.enqueued += len(pcm)

    def stop(self):
        self.stopped = True

    def busy(self):
        return False


class VoiceAudioTests(unittest.TestCase):
    def tts(self, events):
        tts = TTS("http://test/v1/", "ryan")
        ws = Socket(events)
        tts._connect = Mock(return_value=ws)
        return tts, ws

    def test_recognition_language_is_a_hint_that_can_be_left_off(self):
        for source, expected in [("en", "en"), ("zh-CN", "zh"), ("auto", ""), ("", ""), (" Auto ", "")]:
            self.assertEqual(stt_language(source), expected)
        for language, expected in [("zh", "zh"), ("auto", None)]:
            stt = STT("http://test/v1", language, "primer")
            response = Mock(headers={})
            response.json.return_value = {"text": "帮我看一下这个文件。"}
            with patch.object(stt.client, "post", return_value=response) as post:
                self.assertEqual(stt.transcribe(b"\0\0" * 100), "帮我看一下这个文件。")
            data = post.call_args.kwargs["data"]
            # A pinned language makes Whisper render every other one into it; with none,
            # it detects, which is the only way a bilingual speaker is transcribed at all.
            self.assertEqual(data.get("language"), expected)
            self.assertEqual(data["prompt"], "primer")
            stt.client.close()

    def test_a_long_turn_is_waited_for_in_proportion(self):
        # A fixed 30 s timeout gives up on a minute of speech just as Whisper is finishing it.
        stt = STT("http://test/v1", "en")
        response = Mock(headers={})
        response.json.return_value = {"text": "a minute of it"}
        for seconds, expected in [(3, 30.0), (60, 90.0)]:
            with patch.object(stt.client, "post", return_value=response) as post:
                stt.transcribe(b"\0\0" * (16_000 * seconds))
            self.assertEqual(post.call_args.kwargs["timeout"], expected)
        stt.client.close()

    def test_language_is_explicit_and_overridable(self):
        for source, expected in [("en", "English"), ("en-US", "English"),
                                 ("zh", "Chinese"), ("french", "French"), ("Auto", "Auto")]:
            self.assertEqual(tts_language(source), expected)
        with self.assertRaises(ValueError):
            tts_language("unsupported")

    def test_profile_and_cache_invalidation(self):
        response = Mock(content=b"\0\0" * 100, headers={"content-type": "audio/pcm"})
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            for style in [DEFAULT_INSTRUCTIONS, "Quiet and even."]:
                clips = Clips("http://test/v1", "ryan", "en", cache, style)
                tts = TTS("http://test/v1", "ryan", "en", style)
                with patch.object(clips.client, "post", return_value=response) as post:
                    self.assertEqual(clips.get("Let me check."), response.content)
                    clips.get("Let me check.")
                    post.assert_called_once()
                    payload = post.call_args.kwargs["json"]
                    for key in ["model", "voice", "language", "instructions", "response_format"]:
                        self.assertEqual(payload[key], tts.config[key])
                clips.client.close()
            self.assertEqual(len(list(cache.glob("*.pcm"))), 2)
            for path in cache.glob("*.pcm"):
                path.write_bytes(b"broken!")
            clips = Clips("http://test/v1", "ryan", "en", cache)
            with patch.object(clips.client, "post", return_value=response) as post:
                self.assertEqual(clips.get("Let me check."), response.content)
                post.assert_called_once()
            clips.client.close()
            # A different gateway must not reuse clips from the first one.
            clips = Clips("http://other/v1", "ryan", "en", cache)
            with patch.object(clips.client, "post", return_value=response) as post:
                clips.get("Let me check.")
                post.assert_called_once()
            clips.client.close()

    def test_invalid_clip_is_never_cached(self):
        cases = [(b"", "audio/pcm"), (b"x", "audio/pcm"), (b"RIFF" + b"\0" * 100, "audio/pcm"),
                 (b"\0\0" * (TTS_RATE * 9), "audio/pcm"),
                 # An error page is not audio, whatever its bytes happen to look like.
                 (b"\0\0" * 100, "text/html"), (b"\0\0" * 100, "application/json"),
                 (b"\0\0" * 100, "audio/wav")]
        for pcm, kind in cases:
            with self.subTest(size=len(pcm), kind=kind), tempfile.TemporaryDirectory() as directory:
                clips = Clips("http://test/v1", "ryan", "English", Path(directory))
                headers = {"content-type": f"{kind}; charset=utf-8"}
                with patch.object(clips.client, "post", return_value=Mock(content=pcm, headers=headers)):
                    with self.assertRaises(RuntimeError):
                        clips.get("Let me check.")
                self.assertEqual(list(Path(directory).glob("*.pcm")), [])
                clips.client.close()

    def test_whole_reply_and_session_reuse(self):
        text = "All done. Another sentence."
        tts, ws = self.tts(audio_events(text) + audio_events("Next reply."))
        sink = Sink(1000)
        segments = tts.speak(text, sink, threading.Event())
        self.assertEqual(segments, [[text, 1000, 1200]])
        self.assertEqual(ws.sent, [{"type": "input.text", "text": text}, {"type": "input.done"}])
        tts.speak("Next reply.", sink, threading.Event())
        tts._connect.assert_called_once()
        self.assertFalse(ws.closed)

    def test_multiple_server_segments_and_odd_network_chunks(self):
        events = [event("audio.start", sentence_text="First."), b"\1", b"\2\3\4",
                  event("audio.done", total_bytes=4), event("audio.start", sentence_text="Second."),
                  b"\5\6", event("audio.done", total_bytes=2), event("session.done")]
        tts, _ = self.tts(events)
        sink = Sink()
        segments = tts.speak("First. Second.", sink, threading.Event())
        self.assertEqual(b"".join(sink.chunks), b"\1\2\3\4\5\6")
        self.assertTrue(all(len(chunk) % 2 == 0 for chunk in sink.chunks))
        self.assertEqual(segments, [["First.", 0, 4], ["Second.", 4, 6]])

    def test_bad_streams_stop_playback_and_discard_socket(self):
        cases = [
            [event("error", message="bad configuration")],
            [event("audio.start", format="wav", sample_rate=TTS_RATE)],
            [event("audio.start", format="pcm", sample_rate=48000)],
            [b"\0\0"],
            [event("session.done")],
            [event("audio.start"), b"\0", event("audio.done")],
            [event("audio.start"), b"\0\0", event("audio.done", error=True)],
            [event("audio.start"), b"\0\0", event("audio.done", total_bytes=4)],
            [event("audio.start"), b"\0\0", event("session.done")],
            [event("audio.start"), b"\0\0" * (TTS_RATE * 14)],
            [event("audio.start"), OSError("connection lost")],
            [event("audio.start"), ConnectionClosedError(None, None)],
        ]
        for events in cases:
            with self.subTest(events=[type(e).__name__ for e in events]):
                tts, ws = self.tts(events)
                sink = Sink()
                with self.assertRaises((RuntimeError, OSError)):
                    tts.speak("Hello.", sink, threading.Event())
                self.assertTrue(ws.closed)
                self.assertTrue(sink.stopped)
                self.assertIsNone(tts.ws)

    def test_duration_guard_is_sized_in_speech_not_characters(self):
        # 48 Chinese characters are about 12 seconds of audio. Budgeted as characters they
        # were allowed 6, and every Chinese answer died as "exceeded the audio duration limit".
        text = "健康是一个广泛的话题。您具体想了解哪方面的健康呢？比如饮食、运动、睡眠、心理健康，还是其他方面？"
        pcm = b"\0\0" * (TTS_RATE * 12)
        tts, ws = self.tts([event("audio.start", sentence_text=text), pcm,
                            event("audio.done", total_bytes=len(pcm)), event("session.done")])
        sink = Sink()
        self.assertEqual(tts.speak(text, sink, threading.Event()), [[text, 0, len(pcm)]])
        self.assertEqual(ws.sent[0], {"type": "input.text", "text": text})
        self.assertFalse(ws.closed)

    def test_stall_is_bounded(self):
        tts, ws = self.tts([TimeoutError()] * 100)
        sink = Sink()
        with patch("voice.speech.time.monotonic", side_effect=[0, 16]):
            with self.assertRaisesRegex(TimeoutError, "stalled"):
                tts.speak("Hello.", sink, threading.Event())
        self.assertTrue(ws.closed)
        self.assertTrue(sink.stopped)

    def test_cancel_during_receive_drops_late_audio(self):
        tts, ws = self.tts(audio_events())
        cancel, sink = threading.Event(), Sink()
        original_recv = ws.recv

        def recv(timeout):
            msg = original_recv(timeout)
            if isinstance(msg, bytes):
                cancel.set()
            return msg

        ws.recv = recv
        tts.speak("All done.", sink, cancel)
        self.assertEqual(sink.chunks, [])
        self.assertTrue(ws.closed)

    def test_empty_or_cancelled_text_does_not_connect(self):
        tts, _ = self.tts([])
        tts.speak("[laughs]", Sink(), threading.Event())
        cancel = threading.Event()
        cancel.set()
        tts.speak("Do not play.", Sink(), cancel)
        tts._connect.assert_not_called()

    def test_cancel_during_playback_also_closes_socket(self):
        tts, ws = self.tts(audio_events())
        cancel, sink = threading.Event(), Sink()

        def busy():
            cancel.set()
            return True

        sink.busy = busy
        tts.speak("All done.", sink, cancel)
        self.assertTrue(ws.closed)
        self.assertTrue(sink.stopped)

    def test_interrupt_uses_absolute_playback_offsets(self):
        speech = client.Speech.__new__(client.Speech)
        speech.speaker = Sink(1200)
        speech.start = 1000
        speech.sentences = [["Already heard.", 1000, 1100], ["Not finished.", 1100, 1500]]
        speech.cancel = threading.Event()
        self.assertEqual(speech.interrupt(), "Already heard.")
        self.assertTrue(speech.cancel.is_set())
        self.assertTrue(speech.speaker.stopped)

    def test_muted_inflight_narration_never_plays_later(self):
        entered, release = threading.Event(), threading.Event()

        def get(text):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release synthesis")
            return b"\0\0"

        sink = Sink()
        narration = client.Narration(Mock(get=get), sink, lambda: False)
        try:
            narration.offer("Let me check.")
            self.assertTrue(entered.wait(2))
            narration.mute()
            release.set()
        finally:
            narration.close()
            release.set()
            narration.thread.join(2)
        self.assertFalse(narration.thread.is_alive())
        self.assertEqual(sink.chunks, [])


if __name__ == "__main__":
    unittest.main()
