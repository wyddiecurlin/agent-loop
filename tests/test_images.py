# AI_OWNED
"""Image conversion and provider contracts; --live probes every catalog entry."""

import base64
from dataclasses import replace
from io import BytesIO
import json
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from PIL import Image

from agent_loop.images import convert, MAX_IMAGE_BYTES
from agent_loop.providers import CATALOG, ChatProvider, OpenAIProvider, image_parts, validate_images


def fixture(color="red"):
	image = Image.new("RGB", (160, 120), "white")
	from PIL import ImageDraw
	ImageDraw.Draw(image).rectangle((30, 20, 130, 100), fill=color)
	out = BytesIO()
	image.save(out, "PNG")
	return out.getvalue()


def live():
	parts = convert(fixture())
	messages = [{"role": "user", "content": [
		{"type": "input_text", "text": "What color is the rectangle? Reply with only the color."}, *parts]}]

	def probe(pair):
		name, alias = pair
		spec = CATALOG[name][alias]
		result = {"provider": name, "alias": alias, "model": spec.id}
		try:
			# Discovery probes must reach endpoints whose vision support is unknown.
			CATALOG[name][alias] = replace(spec, vision=True, max_images=1)
			provider = OpenAIProvider() if name == "openai" else ChatProvider(backend=name)
			turn = provider.generate(messages, spec.id, None, thinking=False, max_output_tokens=1024, timeout=45)
			result.update(status="pass" if "red" in (turn.text or "").lower() else "fail", answer=turn.text)
			if result["status"] == "pass":
				changed = provider.generate([{"role": "user", "content": [*messages[0]["content"][:1], *convert(fixture("blue"))]}],
				                            spec.id, None, thinking=False, max_output_tokens=1024, timeout=45)
				result["changed_image_answer"] = changed.text
				if "blue" not in (changed.text or "").lower():
					result["status"] = "inconclusive"
				blind = provider.generate([{"role": "user", "content": messages[0]["content"][:1]}], spec.id, None,
				                          thinking=False, max_output_tokens=1024, timeout=45)
				result["blind_answer"] = blind.text
		except Exception as exc:
			unsupported = any(text in str(exc) for text in ("does not support image", "Multimodal not supported"))
			result.update(status="unsupported" if unsupported else "unverified", error=str(exc)[:500])
		finally:
			CATALOG[name][alias] = spec
		return result

	pairs = [(name, alias) for name, models in CATALOG.items() for alias in models]
	with ThreadPoolExecutor(max_workers=3) as pool:
		for result in pool.map(probe, pairs):
			print(json.dumps(result), flush=True)


def live_tool():
	from agent_loop.loop import agent_loop
	from agent_loop.runtime import DockerRuntime
	runtime = DockerRuntime()
	runtime.setup()
	runtime.write("image-test.png", fixture())
	for name, models in CATALOG.items():
		for alias, spec in models.items():
			result = {"provider": name, "alias": alias, "model": spec.id}
			if not spec.vision:
				result["status"] = "unsupported" if spec.max_images == 0 else "unverified"
			else:
				with patch.dict(os.environ, {"PROVIDER": name, "MODEL": alias, "OPENAI_MODEL": spec.id,
				                            "QWEN_MODEL": spec.id, "FALLBACK": "none", "AGENT_TIMEOUT_S": "45", "AGENT_MAX_RETRIES": "0"}):
					run = agent_loop("Use image_view on image-test.png. What color is the rectangle? Submit only the color with done.",
					                 runtime, tools=["image_view"], max_steps=4, max_output_tokens=2048, thinking=False, verbose=False)
					calls = [item.get("name") for item in run.messages if item.get("type") == "function_call"]
					answer = run.messages[-1].get("content", "")
					result.update(status="pass" if run.stop_reason == "done" and "image_view" in calls and "red" in str(answer).lower() else "fail",
					              answer=answer, error=run.error)
			print(json.dumps(result), flush=True)


class ImagesTest(unittest.TestCase):
	def test_together_qwen_request_requirements(self):
		from tests.test_providers import FakeClient, StreamingClient
		client = StreamingClient('{"path":"a.txt","content":"hello"}')
		callback = Mock()
		turn = ChatProvider("together", client=client).generate("hello", CATALOG["together"]["qwen-3.7-plus"].id, None, on_tool_call=callback)
		self.assertTrue(client.seen["stream"])
		callback.assert_not_called()
		self.assertEqual(json.loads(turn.tool_calls[0].arguments), {"path": "a.txt", "content": "hello"})
		client = FakeClient()
		provider = ChatProvider("together", client=client)
		for thinking, effort in ((False, "low"), (True, "xhigh")):
			provider.generate("hello", CATALOG["together"]["qwen-3.8-max"].id, None, thinking=thinking)
			self.assertEqual(client.seen["reasoning_effort"], effort)
		ChatProvider("fireworks", client=client).generate("hello", CATALOG["fireworks"]["qwen-3.7-plus"].id, None)
		self.assertNotIn("stream", client.seen)

	def test_wire_formats(self):
		from tests.test_providers import FakeClient
		part, = convert(fixture())
		messages = [{"role": "user", "content": [{"type": "input_text", "text": "Inspect"}, part]}]
		client = FakeClient()
		ChatProvider("qwen", client=client).generate(messages, "qwen3.5-9b", None)
		self.assertEqual(client.seen["messages"][0]["content"][1],
		                 {"type": "image_url", "image_url": {"url": part["image_url"], "detail": "low"}})
		client = Mock()
		provider = OpenAIProvider(client=client)
		with patch.object(provider, "_to_turn"):
			provider.generate(messages, "gpt-5-nano", None)
		self.assertEqual(client.responses.create.call_args.kwargs["input"], messages)

	def test_fallback_checks_lower_limit_before_sending(self):
		from tests.test_providers import pair, status_error
		fallback, client = pair(status_error(503))
		part, = convert(fixture())
		with patch.dict(CATALOG["together"], {"kimi-k3": replace(CATALOG["together"]["kimi-k3"], max_images=1)}):
			with self.assertRaisesRegex(ValueError, "at most 1"):
				fallback.generate([{"role": "user", "content": [part, part]}], CATALOG["fireworks"]["kimi-k3"].id, None)
		self.assertFalse(client.seen)

	def test_image_estimate_ignores_base64_length(self):
		from agent_loop.context import ContextBudget
		def estimate(length):
			return ContextBudget.size([{"role": "user", "content": [{"type": "input_image", "image_url": "x" * length}]}], [])
		self.assertEqual(estimate(100), estimate(50_000))
		self.assertGreater(estimate(100), 4096)

	def test_download_bounds_and_redirects(self):
		from agent_loop.web import WebClient, httpx
		client = WebClient(http=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=fixture()))))
		self.assertEqual(client.image_bytes("https://example.com/photo", 100_000), fixture())
		with self.assertRaisesRegex(ValueError, "exceeds"):
			client.image_bytes("https://example.com/photo", 1)
		client = WebClient(http=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(302, headers={"location": "https://other.example/photo"}))))
		with self.assertRaisesRegex(ValueError, "another host"):
			client.image_bytes("https://example.com/photo", 100_000)

	def test_tool_pairs_precede_image_and_done_waits(self):
		from agent_loop import loop
		from agent_loop.tools import ToolCall, ToolResult
		part, = convert(fixture())
		registry = Mock()
		registry.execute.side_effect = [ToolResult(True, "prepared", {"attachments": [part]}),
		                               ToolResult(True, "premature"), ToolResult(True, "red")]
		turns = [Mock(text=None, tool_calls=[ToolCall("a", "image_view", "{}"), ToolCall("b", "done", "{}")], stop_reason="completed"),
		         Mock(text=None, tool_calls=[ToolCall("c", "done", "{}")], stop_reason="completed")]
		def attach(messages, parts, prompt):
			self.assertEqual([item["call_id"] for item in messages[-2:]], ["a", "b"])
			messages.append({"role": "user", "content": parts})
		with patch.object(loop, "build_registry", return_value=registry), patch.object(loop, "generate", side_effect=turns), \
		     patch.object(loop, "attach_images", side_effect=attach), patch.object(loop, "asdict", return_value={}):
			run = loop.agent_loop("Inspect", None, verbose=False)
		self.assertEqual(run.steps, 2)
		self.assertEqual(run.messages[-1]["content"], "red")

	def test_pdf_splits_pages_in_order(self):
		out = BytesIO()
		Image.new("RGB", (300, 200), "red").save(out, "PDF", save_all=True,
			append_images=[Image.new("RGB", (200, 300), "blue")])
		parts = convert(out.getvalue())
		self.assertEqual([p["text"] for p in parts if p["type"] == "input_text"], ["Page 1", "Page 2"])
		self.assertEqual(len(image_parts([{"content": parts}])), 2)

	def test_source_equivalence(self):
		from agent_loop.tools import image_view
		runtime, web = Mock(), Mock()
		runtime.read_bytes.return_value = web.image_bytes.return_value = fixture()
		sources = ("photo.png", "https://example.com/photo", fixture(),
		           "data:image/png;base64," + base64.b64encode(fixture()).decode())
		outputs = [image_parts([{"content": image_view(runtime, web, source).metadata["attachments"]}]) for source in sources]
		self.assertTrue(all(output == outputs[0] for output in outputs))

	def test_count_is_across_history_for_every_model(self):
		part, = convert(fixture())
		for name, models in CATALOG.items():
			for alias, spec in models.items():
				with self.subTest(provider=name, model=alias):
					if not spec.vision:
						with self.assertRaises(ValueError):
							validate_images([{"content": [part]}], spec.id, name)
						continue
					limit = spec.max_images or spec.image_batch
					messages = [{"content": [part]}] * limit
					validate_images(messages, spec.id, name)
					with self.assertRaises(ValueError):
						validate_images([*messages, {"content": [part]}], spec.id, name)

	def test_pdf_batches_do_not_accumulate_images(self):
		from agent_loop.loop import attach_images
		part, = convert(fixture())
		parts = [p for page in range(1, 10) for p in ({"type": "input_text", "text": f"Page {page}"}, part)]
		provider = ChatProvider(backend="qwen", client=Mock())
		seen = []
		def generate(**kwargs):
			seen.append(kwargs["messages"])
			return Mock(text="Page observations")
		messages = [{"role": "system", "content": "Help"}]
		with patch("agent_loop.loop.make_provider", return_value=provider), \
		     patch("agent_loop.loop.default_model", return_value="qwen3.5-9b"), \
		     patch("agent_loop.loop.generate", side_effect=generate):
			attach_images(messages, parts, "Summarize")
		self.assertEqual([len(image_parts(batch)) for batch in seen], [4, 4, 1])
		labels = [p["text"] for batch in seen for p in batch[0]["content"] if p.get("text", "").startswith("Page ")]
		self.assertEqual(labels, [f"Page {page}" for page in range(1, 10)])
		self.assertFalse(image_parts(messages))

	def test_all_small_raster_formats(self):
		for format in ("JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF", "AVIF", "HEIF"):
			with self.subTest(format=format):
				out = BytesIO()
				Image.new("RGB", (320, 160), "red").save(out, format)
				part, = convert(out.getvalue())
				with Image.open(BytesIO(base64.b64decode(part["image_url"].split(",")[1]))) as result:
					self.assertEqual(result.format, "JPEG")
					self.assertEqual(result.size, (320, 160))

	def test_large_noisy_image_is_small(self):
		out = BytesIO()
		Image.effect_noise((2000, 1500), 100).save(out, "PNG")
		self.assertGreater(len(out.getvalue()), 500_000)
		part, = convert(out.getvalue())
		data = base64.b64decode(part["image_url"].split(",")[1])
		self.assertLessEqual(len(data), MAX_IMAGE_BYTES)
		with Image.open(BytesIO(data)) as result:
			self.assertLessEqual(max(result.size), 512)

	def test_orientation_and_metadata(self):
		out = BytesIO()
		exif = Image.Exif()
		exif[274] = 6
		exif[315] = "Private author"
		Image.new("RGB", (300, 200), "red").save(out, "JPEG", exif=exif)
		part, = convert(out.getvalue())
		with Image.open(BytesIO(base64.b64decode(part["image_url"].split(",")[1]))) as result:
			self.assertEqual(result.size, (200, 300))
			self.assertFalse(result.getexif())

	def test_bad_input(self):
		for data in (b"", b"not a photo"):
			with self.subTest(data=data), self.assertRaises(ValueError):
				convert(data)


if __name__ == "__main__":
	if "--live-tool" in sys.argv:
		live_tool()
	elif "--live" in sys.argv:
		live()
	else:
		unittest.main()
