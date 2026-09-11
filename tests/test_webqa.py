# AI_OWNED
"""Offline checks for evidence grading and hard evaluation budgets."""

import io
import json
import time
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from agent_loop.providers import ModelTurn, Usage
from agent_loop.tools import ToolCall
from evals import webqa


TASK = {"id": "fixture", "question": "Which element is used?", "query": "element",
        "aliases": ["Zinc"], "source_domains": ["example.com"], "evidence_terms": ["element"]}
URL = "https://example.com/paper"
QUOTE = "The element used is zinc."
TRACE = [{"tool": "web_search", "ok": True,
          "output": f"1. Paper\n   {URL}\n   {QUOTE}", "metadata": {"hits": 1}}]


def answer(value="Zinc", url=URL, quote=QUOTE):
	return json.dumps({"answer": value, "citations": [{"url": url, "quote": quote}]})


class WebQATests(unittest.TestCase):
	def test_numbered_lists_in_excerpts_are_not_search_hit_headers(self):
		trace = [{**TRACE[0], "output": TRACE[0]["output"] + "\n1. A subsection\n\n2. Another subsection"}]
		self.assertEqual(len(list(webqa.evidence(trace))), 1)
		self.assertTrue(webqa.grade(TASK, answer(), trace)["passed"])
		trace = [{**TRACE[0], "output": TRACE[0]["output"].replace("zinc", "[zinc](https://example.com/zinc)")}]
		self.assertTrue(webqa.grade(TASK, answer(), trace)["passed"])

	def test_canonical_answer_needs_observed_support(self):
		self.assertTrue(webqa.grade(TASK, answer(), TRACE)["passed"])
		self.assertTrue(webqa.grade(TASK, answer(url=URL + "#section"), TRACE)["passed"])
		for text, trace in [
			(answer(value="not zinc"), TRACE), (answer(), []),
			(answer(url="https://example.com/other"), TRACE),
			("Zinc", TRACE),
			('{"answer": "Zinc", "citations": null}', TRACE),
		]:
			with self.subTest(answer=text):
				self.assertFalse(webqa.grade(TASK, text, trace)["passed"])
		for quote in ("The element used is zinc, according to the authors.", "zinc", "Zinc is wrong."):
			result = webqa.grade(TASK, answer(quote=quote), TRACE)
			self.assertTrue(result["passed"])  # The cited source still supports the answer.
			self.assertFalse(result["quote_verified"])  # Quote fidelity is a separate metric.

	def test_unknown_domains_errors_and_redirects_are_not_support(self):
		for source in ("https://example.com.evil.test/paper", "https://spam.test/paper"):
			trace = [{**TRACE[0], "output": TRACE[0]["output"].replace(URL, source)}]
			self.assertFalse(webqa.grade(TASK, answer(url=source), trace)["passed"])
		self.assertFalse(webqa.grade(TASK, answer(), [{**TRACE[0], "ok": False}])["passed"])
		trace = [{"tool": "web_fetch", "ok": True, "output": QUOTE,
		          "metadata": {"url": URL, "redirect": "https://elsewhere.org/"}}]
		self.assertFalse(webqa.grade(TASK, answer(), trace)["passed"])
		self.assertFalse(webqa.grade(TASK, answer(quote="Zinc is wrong."), TRACE)["quote_verified"])

	def test_sqlite_build_setting_does_not_establish_the_default(self):
		task = next(json.loads(line) for line in webqa.DATA.read_text().splitlines()
		            if json.loads(line)["id"] == "docs/sqlite-compound-select")
		url = "https://www.sqlite.org/sfa/version?verbose"
		text = "SQLITE_MAX_COMPOUND_SELECT=500\nSQLITE_MAX_DEFAULT_PAGE_SIZE=8192"
		self.assertFalse(webqa.supports(task, url, text))
		self.assertTrue(webqa.supports(task, task["source_urls"][0], task["reference_quote"]))

	def run_fake(self, calls, respond=None, seconds=60, variant="brave"):
		requests, prompts = [], []
		def handler(req):
			requests.append(req)
			if respond:
				return respond(req)
			return webqa.httpx.Response(200, json={"web": {"results": [
				{"url": URL, "title": "Paper", "description": QUOTE}]}})
		calls = iter(calls)
		class Provider:
			def generate(self, messages, model, tools, **kw):
				prompts.append((messages[1]["content"], tools, kw))
				return ModelTurn(None, next(calls), Usage(input_tokens=10, output_tokens=2), "completed")
		http = webqa.httpx.Client(transport=webqa.httpx.MockTransport(handler))
		with patch.object(webqa.httpx, "Client", return_value=http), \
		     patch.dict(webqa.os.environ, {"BRAVE_SEARCH_API_KEY": "fixture"}), \
		     redirect_stderr(io.StringIO()):
			result = webqa.run_task(TASK, variant, provider=Provider(), model="fixture", seconds=seconds)
		return result, requests, prompts

	def test_real_loop_can_search_and_finish_without_receiving_the_key(self):
		result, requests, prompts = self.run_fake([
			[ToolCall("1", "web_search", '{"query":"element","limit":100}')],
			[ToolCall("2", "done", answer())],
		])
		self.assertTrue(result["passed"], result)
		self.assertEqual(requests[0].url.params["count"], "5")
		self.assertEqual(result["turns"], 2)
		self.assertEqual(result["usage"]["input_tokens"], 20)
		self.assertTrue(all(p[0] == TASK["question"] + webqa.ANSWER_FORMAT for p in prompts))
		self.assertEqual({t["name"] for t in prompts[0][1]}, {"web_search", "web_fetch", "done"})
		self.assertLessEqual(prompts[0][2]["timeout"], 60)

	def test_batch_cannot_bypass_search_budget(self):
		result, requests, _ = self.run_fake([
			[ToolCall(str(i), "web_search", json.dumps({"query": str(i)})) for i in range(4)],
		])
		self.assertEqual(result["stop_reason"], "web_search_budget")
		self.assertEqual(len(requests), 3)
		self.assertFalse(result["passed"])

	def test_fetch_budget_and_no_web_control(self):
		with patch("agent_loop.web.validate", side_effect=lambda url: url):
			result, requests, _ = self.run_fake([
				[ToolCall(str(i), "web_fetch", json.dumps({"url": URL})) for i in range(3)]],
				respond=lambda req: webqa.httpx.Response(200, text=QUOTE, headers={"content-type": "text/plain"}))
		self.assertEqual(result["stop_reason"], "web_fetch_budget")
		self.assertEqual(len(requests), 2)
		result, requests, prompts = self.run_fake([
			[ToolCall("1", "done", answer())]], variant="no-web")
		self.assertTrue(result["answer_correct"])
		self.assertFalse(result["passed"])
		self.assertEqual(requests, [])
		self.assertEqual([t["name"] for t in prompts[0][1]], ["done"])

	def test_deadline_interrupts_in_flight_tool(self):
		def slow(req):
			time.sleep(1)
			raise AssertionError("deadline failed")
		result, requests, _ = self.run_fake([
			[ToolCall("1", "web_search", '{"query":"element"}')]], respond=slow, seconds=.02)
		self.assertEqual(result["stop_reason"], "deadline")
		self.assertLess(result["duration_s"], .5)
		self.assertEqual(len(requests), 1)
		self.assertFalse(result["passed"])

	def test_provider_comparison_disables_search_fallback(self):
		with patch.dict(webqa.os.environ, {"PARALLEL_SEARCH_API_KEY": "fixture"}):
			result, requests, _ = self.run_fake([
				[ToolCall("1", "web_search", '{"query":"element"}')],
				[ToolCall("2", "done", answer())]],
				respond=lambda req: webqa.httpx.Response(401))
		self.assertEqual(len(requests), 1)
		self.assertEqual(requests[0].method, "GET")
		self.assertFalse(result["passed"])


if __name__ == "__main__":
	unittest.main()
