# AI_OWNED
"""Offline response budgets and conversation rollover: run inside ./run.sh."""
import importlib
import io
import json
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_loop.context import ContextBudget, model_limits
from agent_loop.loop import AgentRun
from agent_loop.providers import ChatProvider, OpenAIProvider, ModelTurn, Usage, ToolCall
from tests.test_providers import FakeClient, pair, status_error
from voice import bridge

loop = importlib.import_module('agent_loop.loop')


class ResponseBudgetTests(unittest.TestCase):
    def test_chat_override_is_per_call_and_survives_fallback(self):
        client = FakeClient()
        provider = ChatProvider('qwen', client=client, max_output_tokens=8192, thinking=True)
        provider.generate([], 'qwen3.5-9b', None, max_output_tokens=128, thinking=False)
        self.assertEqual(client.seen['max_tokens'], 128)
        self.assertFalse(client.seen['extra_body']['chat_template_kwargs']['enable_thinking'])
        provider.generate([], 'qwen3.5-9b', None)
        self.assertEqual(client.seen['max_tokens'], 8192)
        self.assertTrue(client.seen['extra_body']['chat_template_kwargs']['enable_thinking'])
        fallback, client = pair(status_error(503))
        fallback.generate([], 'accounts/fireworks/models/glm-5p3-flash', None, max_output_tokens=128)
        self.assertEqual(client.seen['max_tokens'], 128)

    def test_responses_override_and_validation(self):
        client = Mock()
        provider = OpenAIProvider(client=client)
        with patch.object(provider, '_to_turn', return_value=None):
            provider.generate([], 'gpt-5-nano', None, max_output_tokens=128)
            self.assertEqual(client.responses.create.call_args.kwargs['max_output_tokens'], 128)
            provider.generate([], 'gpt-5-nano', None)
            self.assertNotIn('max_output_tokens', client.responses.create.call_args.kwargs)
            for invalid in (True, 0, -1, 1.5, '128', 65537):
                with self.assertRaises(ValueError):
                    provider.generate([], 'gpt-5-nano', None, max_output_tokens=invalid)

    def test_limits_follow_model_provider_fallback_and_custom_deployment(self):
        with patch.dict(os.environ, {'PROVIDER': 'qwen', 'MODEL': '', 'QWEN_MODEL': 'qwen3.5-9b'}, clear=True):
            self.assertEqual(model_limits()[0], 32768)
        with patch.dict(os.environ, {'PROVIDER': 'together', 'MODEL': 'qwen-3.7-plus', 'FALLBACK': 'fireworks'}, clear=True):
            self.assertEqual(model_limits()[0], 262144)
        with patch.dict(os.environ, {'PROVIDER': 'qwen', 'MODEL': 'custom'}, clear=True):
            self.assertEqual(model_limits()[0], 32768)
            os.environ['CONTEXT_WINDOW_TOKENS'] = '16384'
            self.assertEqual(model_limits()[0], 16384)

    def test_threshold_uses_input_usage_and_reserves_output(self):
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'old'},
                    {'role': 'assistant', 'content': 'old reply'}, {'role': 'user', 'content': 'new'}]
        budget = ContextBudget('auto-clear', window=2000)
        budget.observe(messages, [], Usage(input_tokens=1660, cached_input_tokens=1500))
        self.assertFalse(budget.prepare(messages, [], 3, 128))
        self.assertTrue(budget.prepare(messages, [], 3, 140))  # exactly 90%
        self.assertEqual([m['content'] for m in messages], ['rules', 'new'])
        self.assertIsNone(budget.anchor)

    def test_clear_preserves_current_tools_and_compaction_is_noop(self):
        current = [{'role': 'user', 'content': 'new'},
                   {'type': 'function_call', 'call_id': 'x', 'name': 'fs_read', 'arguments': '{}'},
                   {'type': 'function_call_output', 'call_id': 'x', 'output': 'result'}]
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': '中' * 800}, *current]
        before = deepcopy(messages)
        self.assertFalse(ContextBudget('compaction', window=2000).prepare(messages, [], 2, 128))
        self.assertEqual(messages, before)
        self.assertTrue(ContextBudget('auto-clear', window=2000).prepare(messages, [], 2, 128))
        self.assertEqual(messages[1:], current)

    def test_oversized_current_turn_or_tool_schema_is_rejected(self):
        budget = ContextBudget('auto-clear', window=2000)
        for messages, schema in (([{'role': 'user', 'content': 'x' * 2000}], []),
                                 ([{'role': 'user', 'content': 'hello'}], [{'description': 'x' * 2000}])):
            with self.assertRaisesRegex(ValueError, 'too large'):
                budget.prepare(messages, schema, 1, 128)

    def test_actual_loop_limits_every_step_and_drops_previous_turns(self):
        requests = []
        def generate(**kwargs):
            requests.append(deepcopy(kwargs))
            call = ToolCall('x', 'fs_read' if len(requests) == 1 else 'done', '{}')
            return ModelTurn(None, [call], Usage(input_tokens=100), 'tool_calls')
        registry = Mock()
        registry.schema.return_value = []
        registry.execute.return_value = SimpleNamespace(ok=True, output='short answer', to_model_output=lambda: 'result')
        history = [{'role': 'system', 'content': 'old rules'}, {'role': 'user', 'content': 'x' * 4000}]
        with patch.object(loop, 'generate', side_effect=generate), patch.object(loop, 'build_registry', return_value=registry):
            run = loop.agent_loop('new request', None, history=history, system_prompt='rules', verbose=False,
                                  max_output_tokens=128, context_budget=ContextBudget('auto-clear', window=3000))
        self.assertTrue(run.ok)
        self.assertTrue(run.history_cleared)
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r['max_output_tokens'] == 128 for r in requests))
        self.assertEqual(requests[0]['messages'][1]['content'], 'new request')
        self.assertEqual(requests[1]['messages'][-1]['type'], 'function_call_output')

    def test_large_tool_result_clears_old_turns_without_replaying_current_work(self):
        requests = []
        def generate(**kwargs):
            requests.append(deepcopy(kwargs['messages']))
            call = ToolCall('x', 'fs_read' if len(requests) == 1 else 'done', '{}')
            return ModelTurn(None, [call], Usage(input_tokens=1800), 'tool_calls')
        registry = Mock()
        registry.schema.return_value = []
        registry.execute.return_value = SimpleNamespace(ok=True, output='answer', to_model_output=lambda: 'x' * 1000)
        history = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'old'}]
        with patch.object(loop, 'generate', side_effect=generate), patch.object(loop, 'build_registry', return_value=registry):
            run = loop.agent_loop('new', None, history=history, system_prompt='rules', verbose=False,
                                  max_output_tokens=128, context_budget=ContextBudget('auto-clear', window=3000))
        self.assertTrue(run.ok)
        self.assertTrue(run.history_cleared)
        self.assertEqual(registry.execute.call_count, 2)
        self.assertEqual(requests[0][1]['content'], 'old')
        self.assertEqual(requests[1][1]['content'], 'new')
        self.assertEqual(requests[1][-2]['call_id'], requests[1][-1]['call_id'])

    def test_bridge_turn_overrides_do_not_leak_and_interrupt_survives_clear(self):
        calls = []
        def fake_loop(prompt, runtime, **kwargs):
            calls.append((prompt, kwargs))
            return AgentRun([{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': 'x' * 20000}], 'done', 1)
        events = [ {'type': 'clock', 'max_output_tokens': 128, 'mode': 'auto-clear'},
                   {'type': 'prompt', 'text': 'first'},
                   {'type': 'heard', 'text': 'heard fragment'},
                   {'type': 'prompt', 'text': 'second', 'max_output_tokens': 256},
                   {'type': 'prompt', 'text': 'third'},
                   {'type': 'prompt', 'text': 'invalid', 'max_output_tokens': True}]
        registry = Mock()
        registry.schema.return_value = []
        with patch.dict(os.environ, {'PROVIDER': 'qwen', 'MODEL': '', 'CONTEXT_WINDOW_TOKENS': '16000'}), \
             patch.object(bridge, 'agent_loop', side_effect=fake_loop), patch.object(bridge, 'build_registry', return_value=registry), \
             patch.object(bridge, 'emit') as emit:
            bridge.serve(io.StringIO('\n'.join(map(json.dumps, events))), None)
        self.assertEqual([kw['max_output_tokens'] for _, kw in calls], [128, 256, 128])
        self.assertIsNone(calls[1][1]['history'])
        self.assertIn('heard fragment', calls[1][0])
        self.assertIn('second', calls[1][0])
        self.assertTrue(any(c.args[0]['type'] == 'context_cleared' for c in emit.call_args_list))
        self.assertFalse(emit.call_args.args[0]['ok'])


if __name__ == '__main__':
    unittest.main()
