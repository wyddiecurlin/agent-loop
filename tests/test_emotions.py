# AI_OWNED
"""AGENT_TARGET=test AGENT_ENTRYPOINT=python ./run.sh -m unittest tests.test_emotions -v"""
import json
import io
import queue
import threading
import unittest
from unittest.mock import patch

from voice.emotions import LiveEmotions, QwenClassifier, SCHEMA, observe_stream


class EmotionTests(unittest.TestCase):
    def test_qwen_schema_and_validation(self):
        classifier = QwenClassifier()
        result = {'emotion': 'cute', 'intensity': .7, 'delivery': 'tender'}

        def respond(request, timeout):
            self.assertEqual(timeout, 2.5)
            body = json.loads(request.data)
            self.assertEqual(body['response_format']['json_schema']['schema'], SCHEMA)
            self.assertTrue(body['response_format']['json_schema']['strict'])
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': json.dumps(result)}}]}).encode())

        with patch('voice.emotions.urlopen', respond):
            self.assertEqual(classifier('抱抱我嘛'), {**result, 'emotion': 'jealous'})
            for bad in ({**result, 'emotion': 'unknown'}, {**result, 'intensity': float('nan')},
                        {**result, 'intensity': True}, {**result, 'delivery': 'unknown'}, []):
                result = bad
                with self.assertRaises(ValueError):
                    classifier('hello')

    def test_nonblocking_coalescing_and_old_turn_result_dropped(self):
        entered, release = threading.Event(), threading.Event()
        contexts, events = [], queue.Queue()

        def classify(context):
            contexts.append(json.loads(context))
            if len(contexts) == 1:
                entered.set()
                release.wait(2)
            return {'emotion': 'happy', 'delivery': 'delighted', 'intensity': .7}

        worker = LiveEmotions(events.put, classify, interval=.01)
        try:
            worker.begin(1, 'old')
            self.assertTrue(entered.wait(1))
            worker.begin(2, 'new')
            for _ in range(1000):
                worker.feed('abcde')
            release.set()
            result = events.get(timeout=2)
            self.assertEqual(result['turn'], 2)
            self.assertEqual(len(contexts), 2)
            self.assertEqual(len(contexts[-1]['assistant']), 4000)
            self.assertEqual(contexts[-1]['user'], 'new')
            self.assertTrue(events.empty())
        finally:
            release.set()
            worker.close()

    def test_failure_reports_status_and_next_update_recovers(self):
        events = queue.Queue()
        calls = 0

        def classify(context):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError('secret transcript must never be logged')
            return {'emotion': 'sad', 'delivery': 'reassuring', 'intensity': .6}

        worker = LiveEmotions(events.put, classify, interval=.01)
        try:
            worker.begin(1, 'I lost my pet')
            failure = events.get(timeout=1)
            self.assertEqual(failure['status'], 'failed')
            self.assertNotIn('secret', json.dumps(failure))
            worker.feed("I'm so sorry.")
            self.assertEqual(events.get(timeout=1)['emotion'], 'sad')
        finally:
            worker.close()

    def test_live_callback_tap_preserves_logs_and_only_reads_answer_arguments(self):
        from agent_loop import loop
        fed, logged = [], []

        class Observer:
            def feed(self, text): fed.append(text)

        def generate(**kwargs):
            kwargs['on_text']('That is wonderful!')
            kwargs['on_tool_call']('fs_read', 'function_call')
            kwargs['on_tool_call']('{"path":"sad.txt"}', 'function_args')
            kwargs['on_tool_call']('done', 'function_call')
            kwargs['on_tool_call']('{"answer":"Congratulations', 'function_args')
            kwargs['on_tool_call']('!"}', 'function_args')
            return 'completed'

        with patch.object(loop, 'generate', generate):
            with observe_stream(Observer()):
                self.assertEqual(loop.generate(on_text=lambda x: logged.append(x),
                                               on_tool_call=lambda *x: logged.append(x)), 'completed')
            self.assertIs(loop.generate, generate)
        self.assertEqual(fed, ['That is wonderful!', '{"answer":"Congratulations', '!"}'])
        self.assertEqual(len(logged), 6)

    def test_final_snapshot_retries_once_and_close_discards_inflight_result(self):
        events = queue.Queue()
        calls = []

        def fail(context):
            calls.append(context)
            raise TimeoutError()

        worker = LiveEmotions(events.put, fail, interval=.01)
        try:
            worker.begin(1, 'hello')
            self.assertEqual(events.get(timeout=1)['status'], 'failed')
            self.assertEqual(events.get(timeout=2)['status'], 'failed')
            with self.assertRaises(queue.Empty):
                events.get(timeout=.9)
            self.assertEqual(len(calls), 2)
        finally:
            worker.close()

        entered, release = threading.Event(), threading.Event()

        def slow(context):
            entered.set()
            release.wait(2)
            return {'emotion': 'love', 'delivery': 'tender', 'intensity': .7}

        worker = LiveEmotions(events.put, slow, interval=.01)
        worker.begin(1, 'hello')
        self.assertTrue(entered.wait(1))
        threading.Timer(.05, release.set).start()
        worker.close()
        self.assertFalse(worker.thread.is_alive())
        self.assertTrue(events.empty())


if __name__ == '__main__':
    unittest.main()
