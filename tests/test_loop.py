# AI_OWNED
"""Offline regression tests for repeated tool calls and bounded recovery."""
import importlib
import io
import json
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from unittest.mock import patch

from agent_loop.providers import ModelTurn, ToolCall
from agent_loop.runtime import DockerRuntime
from agent_loop.tools import Tool, ToolRegistry, ToolResult, done

loop = importlib.import_module('agent_loop.loop')


class RepeatedCallTests(unittest.TestCase):
    def run_calls(self, turns, outputs=None, max_steps=20):
        requests, executed = [], []

        def execute(query, limit=15):
            executed.append(query)
            return ToolResult(ok=True, output=next(outputs) if outputs else 'same result')

        registry = ToolRegistry([
            Tool('web_search', 'search', {}, execute),
            Tool('done', 'finish', {}, done),
        ])
        turns = iter(turns)

        def generate(**kwargs):
            requests.append(deepcopy(kwargs))
            calls = next(turns)
            return ModelTurn(None, [ToolCall(f'{len(requests)}-{i}', name, args)
                                    for i, (name, args) in enumerate(calls)], None, 'completed')

        with patch.object(loop, 'generate', side_effect=generate), \
             patch.object(loop, 'build_registry', return_value=registry), redirect_stderr(io.StringIO()):
            run = loop.agent_loop('find benchmarks', DockerRuntime(), verbose=False, max_steps=max_steps)
        # Even a provider ignoring the restricted schema must keep calls paired.
        calls = [m['call_id'] for m in run.messages if m.get('type') == 'function_call']
        results = [m['call_id'] for m in run.messages if m.get('type') == 'function_call_output']
        self.assertEqual(calls, results)
        return run, requests, executed

    def search(self, query='LOM'):
        return [('web_search', json.dumps({'query': query, 'limit': 15}))]

    def finish(self):
        return [('done', '{"answer":"Here are the benchmarks I found."}')]

    def test_identical_search_recovers_with_answer_and_normalizes_json(self):
        turns = [self.search(), [('web_search', '{"limit":15,"query":"LOM"}')],
                 self.search(), self.search(), self.finish()]
        run, requests, executed = self.run_calls(turns)
        self.assertTrue(run.ok)
        self.assertEqual(run.steps, 5)
        self.assertEqual(len(executed), 4)
        self.assertEqual([t['name'] for t in requests[-1]['tools']], ['done'])
        self.assertIn('identical results', requests[-1]['messages'][-1]['content'])
        self.assertIn('benchmarks', run.answer)

    def test_alternating_calls_are_also_detected(self):
        run, requests, executed = self.run_calls(
            [self.search(q) for q in ['A', 'B', 'A', 'B', 'A']] + [self.finish()])
        self.assertTrue(run.ok)
        self.assertEqual(len(executed), 5)
        self.assertEqual([t['name'] for t in requests[-1]['tools']], ['done'])

    def test_changed_results_allow_polling_to_continue(self):
        run, requests, _ = self.run_calls([self.search()] * 7 + [self.finish()],
                                         outputs=iter(str(i) for i in range(7)))
        self.assertTrue(run.ok)
        self.assertTrue(all(len(r['tools']) == 2 for r in requests))

    def test_new_work_resets_repeat_counter(self):
        run, requests, _ = self.run_calls(
            [self.search()] * 3 + [self.search('new')] + [self.search()] * 2 + [self.finish()])
        self.assertTrue(run.ok)
        self.assertTrue(all(len(r['tools']) == 2 for r in requests))

    def test_ignored_recovery_is_bounded_and_does_not_execute_searches(self):
        run, requests, executed = self.run_calls([self.search()] * 6)
        self.assertEqual(run.stop_reason, 'stalled')
        self.assertEqual(run.steps, 6)
        self.assertEqual(len(requests), 6)
        self.assertEqual(len(executed), 4)
        self.assertIn('Repeated tool calls', run.error)

    def test_malformed_done_can_be_repaired_during_recovery(self):
        run, _, executed = self.run_calls(
            [self.search()] * 4 + [[('done', '{"answer":')], self.finish()])
        self.assertTrue(run.ok)
        self.assertEqual(run.steps, 6)
        self.assertEqual(len(executed), 4)

    def test_recovery_exhaustion_at_step_limit_is_still_stalled(self):
        run, _, _ = self.run_calls([self.search()] * 6, max_steps=6)
        self.assertEqual(run.stop_reason, 'stalled')

    def test_recovery_cannot_exceed_overall_step_limit(self):
        run, requests, _ = self.run_calls([self.search()] * 4, max_steps=4)
        self.assertEqual(run.stop_reason, 'max_steps')
        self.assertEqual(len(requests), 4)

    def test_empty_recovery_turns_are_bounded(self):
        run, requests, _ = self.run_calls([self.search()] * 4 + [[], []])
        self.assertEqual(run.stop_reason, 'stalled')
        self.assertEqual(len(requests), 6)

    def test_multiple_calls_in_one_turn_do_not_exhaust_turn_threshold(self):
        run, requests, _ = self.run_calls([self.search() * 5, self.finish()])
        self.assertTrue(run.ok)
        self.assertEqual(len(requests[-1]['tools']), 2)


if __name__ == '__main__':
    unittest.main()
