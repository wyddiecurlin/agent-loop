# AI_OWNED
"""Offline voice summaries, self-hosted language policy, and room-noise timing."""
import io
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_loop.loop import AgentRun
from agent_loop.providers import ChatProvider, QWEN_ENGLISH_ONLY
from tests.test_providers import FakeClient
from voice import bridge
from voice.turns import BargeIn, Endpointer, frames, spoken_answer, TOO_LONG


class VoiceSummaryTests(unittest.TestCase):
    def test_summary_is_bounded_toolless_and_has_no_thinking(self):
        provider = Mock()
        provider.generate.return_value = SimpleNamespace(text="The tests passed. Two warnings remain.")
        answer = "start " + "detail " * 4000 + "final caveat"
        result = bridge.ask_summary(provider, "model", answer, 20)
        self.assertEqual(result, "The tests passed. Two warnings remain.")
        args, kw = provider.generate.call_args
        self.assertIsNone(args[2])
        self.assertFalse(kw['thinking'])
        self.assertEqual(kw['max_output_tokens'], bridge.SUMMARY_MAX_TOKENS)
        self.assertLess(len(args[0][1]['content']), 16100)
        self.assertTrue(args[0][1]['content'].endswith('final caveat'))
        self.assertIn('at most 28 words', args[0][0]['content'])

    def test_empty_and_overlong_summaries_are_rejected(self):
        for text in ('', 'word ' * 1000):
            provider = Mock()
            provider.generate.return_value = SimpleNamespace(text=text)
            with self.assertRaises(ValueError):
                bridge.ask_summary(provider, 'model', 'answer', 10)

    def serve(self, answer, summarize, seconds=10):
        messages = [{'role': 'user', 'content': 'question'},
                    {'type': 'function_call', 'name': 'done', 'call_id': 'done1',
                     'arguments': json.dumps({'answer': answer})},
                    {'type': 'function_call_output', 'call_id': 'done1', 'output': answer},
                    {'role': 'assistant', 'content': answer}]
        output = io.StringIO()
        with patch.object(bridge, 'agent_loop', return_value=AgentRun(messages, 'done', 1)), \
                patch.object(bridge.sys, 'stdout', output), patch.object(bridge.sys, 'stderr', io.StringIO()):
            bridge.serve(io.StringIO(json.dumps({'type': 'clock', 'max_answer_s': seconds}) + '\n' +
                                    json.dumps({'type': 'prompt', 'text': 'question'}) + '\n'),
                         None, summarize=summarize)
        return [json.loads(line) for line in output.getvalue().splitlines()], messages

    def test_long_answer_speaks_summary_and_preserves_original(self):
        original = 'The test passed. ' * 100
        summarize = Mock(return_value='The test passed.')
        events, messages = self.serve(original, summarize)
        summarize.assert_called_once_with(original, 10)
        self.assertEqual(events[-1]['text'], original)
        self.assertEqual(events[-1]['spoken'], 'The test passed.')
        self.assertEqual(spoken_answer(events[-1]['spoken'], max_seconds=10), 'The test passed.')
        self.assertEqual(events[-2]['answer'], original)
        self.assertEqual(events[-2]['spoken'], 'The test passed.')
        self.assertEqual(messages[-1]['content'], original)

    def test_short_answer_needs_no_extra_call(self):
        summarize = Mock()
        events, _ = self.serve('Done.', summarize)
        summarize.assert_not_called()
        self.assertEqual(events[-1]['spoken'], 'Done.')

    def test_summary_failure_keeps_answer_and_local_fallback(self):
        original = 'The test passed. ' * 100
        events, _ = self.serve(original, Mock(side_effect=TimeoutError))
        self.assertTrue(events[-1]['ok'])
        self.assertEqual(events[-1]['text'], original)
        self.assertEqual(spoken_answer(events[-1]['spoken'], max_seconds=10), TOO_LONG['en'])

    def test_invalid_speech_budgets_are_rejected(self):
        for value in (0, -1, float('inf'), float('nan'), True, '10', None):
            with self.assertRaises(ValueError):
                bridge.turn_options({'max_answer_s': value})


class QwenLanguageTests(unittest.TestCase):
    def test_all_self_hosted_requests_get_english_without_mutating_history(self):
        cases = ['你好', [{'role': 'system', 'content': 'Reply in the user language.'},
                         {'role': 'user', 'content': '你好'}],
                 [{'role': 'developer', 'content': [{'type': 'text', 'text': 'Rules'}]}]]
        for messages in cases:
            original = deepcopy(messages)
            client = FakeClient()
            ChatProvider('qwen', client=client).generate(messages, 'qwen3.5-9b', None)
            self.assertIn(QWEN_ENGLISH_ONLY, str(client.seen['messages']))
            self.assertEqual(messages, original)
        client = FakeClient()
        ChatProvider('fireworks', client=client).generate('你好', 'glm-5.3', None)
        self.assertNotIn(QWEN_ENGLISH_ONLY, str(client.seen['messages']))


class RoomNoiseTimingTests(unittest.TestCase):
    def test_background_bursts_do_not_start_a_turn(self):
        endpoint = Endpointer()
        for probability in [.55] * 100 + [.95] * 4 + [.1] * 60:
            self.assertIsNone(endpoint.feed(probability))

    def test_one_second_thinking_pause_does_not_dispatch(self):
        endpoint = Endpointer()
        events = []
        for p in [.95] * 20 + [.1] * frames(1000) + [.95] * 20:
            event = endpoint.feed(p)
            if event:
                events.append(event)
        self.assertEqual(events, ['start', 'speculate'])
        for _ in range(frames(1500) - 1):
            self.assertNotIn(endpoint.feed(.1), ('end', 'abort'))
        self.assertEqual(endpoint.feed(.1), 'end')
        self.assertEqual(endpoint.speech_frames, 40)

    def test_background_and_short_bursts_do_not_interrupt(self):
        barge = BargeIn()
        for probability in [.65] * 100 + [.99] * frames(250) + [.1]:
            self.assertFalse(barge.feed(probability, True))
        for _ in range(frames(350) - 1):
            self.assertFalse(barge.feed(.99, True))
        self.assertTrue(barge.feed(.99, True))


if __name__ == '__main__':
    unittest.main()
