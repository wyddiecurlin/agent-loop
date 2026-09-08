# AI_OWNED
"""Voice HTTP and playback regressions, with no gateway or sound card required.

	uv run --with httpx --with numpy --with onnxruntime -m tests.test_voice_audio
"""

import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

with patch.dict(sys.modules, {"sounddevice": types.ModuleType("sounddevice")}):
	from voice.client import Narration, Speech
from voice.speech import Clips, TTS, audio_limit


class Sink:
	def __init__(self, offset=0):
		self.enqueued = self.consumed = offset
		self.chunks = []
		self.stopped = False

	def play(self, pcm):
		self.chunks.append(pcm)
		self.enqueued += len(pcm)

	def busy(self):
		return False

	def stop(self):
		self.stopped = True


class Stream(httpx.SyncByteStream):
	def __init__(self, chunks):
		self.chunks = chunks
		self.closed = False

	def __iter__(self):
		for chunk in self.chunks:
			if isinstance(chunk, Exception):
				raise chunk
			yield chunk

	def close(self):
		self.closed = True


class VoiceAudioTests(unittest.TestCase):
	def tts(self, chunks, content_type="audio/pcm", status=200):
		stream = Stream(chunks)
		requests = []

		def handle(request):
			requests.append(json.loads(request.content))
			return httpx.Response(status, headers={"content-type": content_type}, stream=stream)

		tts = TTS("http://tts/v1", "ryan")
		tts.client.close()
		tts.client = httpx.Client(base_url="http://tts/v1", transport=httpx.MockTransport(handle))
		self.addCleanup(tts.client.close)
		return tts, stream, requests

	def test_stream_reassembles_samples_and_keeps_absolute_offsets(self):
		tts, stream, requests = self.tts([b"\x01", b"\x02\x03", b"\x04\x05", b"\x06"])
		sink = Sink(offset=100)
		text = "The tests passed. The change is ready."
		marks = tts.speak(text, sink, threading.Event())
		self.assertEqual(b"".join(sink.chunks), bytes(range(1, 7)))
		self.assertTrue(all(len(chunk) % 2 == 0 for chunk in sink.chunks))
		self.assertEqual(marks, [[text, 100, 106]])
		self.assertEqual(len(requests), 1)
		self.assertEqual(requests[0]["stream_format"], "audio")
		self.assertTrue(stream.closed)

	def test_cancel_does_not_enqueue_remaining_chunks_or_next_utterance(self):
		tts, stream, requests = self.tts([b"\0" * 8, b"\0" * 8])
		cancel, sink = threading.Event(), Sink()
		play = sink.play

		def interrupt(pcm):
			play(pcm)
			cancel.set()

		sink.play = interrupt
		tts.speak("A longer answer. " * 40, sink, cancel)
		self.assertEqual(len(sink.chunks), 1)
		self.assertEqual(len(requests), 1)
		self.assertTrue(sink.stopped and stream.closed)

	def test_already_cancelled_speech_makes_no_request(self):
		tts, _, requests = self.tts([b"\0" * 8])
		cancel = threading.Event()
		cancel.set()
		tts.speak("Hello there.", Sink(), cancel)
		self.assertEqual(requests, [])

	def test_invalid_responses_stop_playback(self):
		cases = [([], "audio/pcm", 200), ([b"12345"], "audio/pcm", 200),
		         ([b"RIFF1234"], "audio/pcm", 200), ([b'{}'], "application/json", 200),
		         ([b'error'], "application/json", 500),
		         ([b"\0" * 8, httpx.ReadError("broken stream")], "audio/pcm", 200)]
		for chunks, content_type, status in cases:
			with self.subTest(chunks=chunks, content_type=content_type, status=status):
				tts, stream, requests = self.tts(chunks, content_type, status)
				sink = Sink()
				with self.assertRaises((ValueError, httpx.HTTPError)):
					tts.speak("Hello there.", sink, threading.Event())
				self.assertTrue(sink.stopped and stream.closed)
				self.assertEqual(len(requests), 1)

	def test_runaway_generation_is_cut_off(self):
		tts, stream, _ = self.tts([b"\0" * 8, b"\0" * audio_limit("Hello.")])
		sink = Sink()
		with self.assertRaisesRegex(ValueError, "audio budget"):
			tts.speak("Hello.", sink, threading.Event())
		self.assertEqual(sink.enqueued, 8)
		self.assertTrue(sink.stopped and stream.closed)

	def test_stream_deadline(self):
		tts, stream, _ = self.tts([b"\0" * 8])
		with patch("voice.speech.time.monotonic", side_effect=[0, 31]):
			with self.assertRaisesRegex(TimeoutError, "timed out"):
				tts.speak("Hello.", Sink(), threading.Event())
		self.assertTrue(stream.closed)

	def test_clip_conditioning_and_cache_identity(self):
		requests = []

		def handle(request):
			requests.append(json.loads(request.content))
			return httpx.Response(200, content=b"\0" * 8, headers={"content-type": "audio/pcm"})

		with tempfile.TemporaryDirectory() as directory:
			for instructions, seed, api in [(None, 42, "http://tts/v1"),
			                                ("A soft, even voice.", 42, "http://tts/v1"),
			                                (None, 7, "http://tts/v1"),
			                                (None, 42, "http://other/v1")]:
				clips = Clips(api, "ryan", "English", Path(directory), instructions, seed)
				clips.client.close()
				clips.client = httpx.Client(base_url=api, transport=httpx.MockTransport(handle))
				self.addCleanup(clips.client.close)
				tts = TTS(api, "ryan", "English", instructions, seed)
				self.addCleanup(tts.client.close)
				self.assertEqual(clips.config, tts.config)
				self.assertEqual(clips.get("Let me check that."), clips.get("Let me check that."))
			self.assertEqual(len(requests), 4)
			self.assertEqual(len(list(Path(directory).glob("*.pcm"))), 4)

	def test_muting_invalidates_a_clip_already_being_fetched(self):
		fetching, release = threading.Event(), threading.Event()

		class DelayedClips:
			def get(self, text):
				fetching.set()
				if not release.wait(2):
					raise TimeoutError("test did not release clip")
				return b"\0" * 8

		sink = Sink()
		narration = Narration(DelayedClips(), sink, lambda: False)
		try:
			narration.offer("Let me check that.")
			self.assertTrue(fetching.wait(2))
			narration.mute()
			release.set()
		finally:
			release.set()
			narration.close()
			narration.thread.join(2)
		self.assertFalse(narration.thread.is_alive())
		self.assertEqual(sink.chunks, [])

	def test_interrupt_uses_absolute_playback_position(self):
		tts, _, _ = self.tts([])
		speech = Speech.__new__(Speech)
		speech.tts, speech.speaker = tts, Sink(offset=1000)
		speech.cancel, speech.start = threading.Event(), 1000
		speech.sentences = [["First sentence.", 1000, 1100], ["Second sentence.", 1100, 1300]]
		speech.speaker.consumed = 1100
		self.assertEqual(speech.interrupt(), "First sentence.")


if __name__ == "__main__":
	unittest.main()
