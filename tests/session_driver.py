# AI_OWNED
"""Trusted fixture entry point used by the real Docker lifecycle integration test."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent_loop.runtime import DockerRuntime


def loop_mount(source):
    from agent_loop.__main__ import main as agent_main
    class ProviderServer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            messages = body['messages']
            assert len([m for m in messages if m.get('role') == 'user']) == 1, 'mount restart duplicated user prompt'
            outputs = [m for m in messages if m.get('role') == 'tool']
            if not outputs:
                name, arguments = 'mount_volume', {'host_path': source}
            elif len(outputs) == 1:
                name, arguments = 'fs_read', {'path': outputs[0]['content'] + '/input.txt'}
            else:
                assert 'host input' in outputs[-1]['content']
                name, arguments = 'done', {'answer': 'Mounted folder read after restart'}
            chunk = {'id': 'chatcmpl-test', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'qwen3.5-9b',
                     'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call-' + name, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, 'finish_reason': None}]}
            finish = {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}
            data = ('data: ' + json.dumps(chunk) + '\n\ndata: ' + json.dumps(finish) + '\n\ndata: [DONE]\n\n').encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = HTTPServer(('127.0.0.1', 0), ProviderServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    os.environ.update(PROVIDER='qwen', QWEN_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1', MODEL='qwen3.5-9b', FALLBACK='none')
    try:
        return agent_main(['Mount the folder and read input.txt'])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    if sys.argv[1] == 'loop_mount':
        return loop_mount(sys.argv[2])
    runtime = DockerRuntime()
    runtime.setup()
    try:
        mode = sys.argv[1]
        if mode == 'create':
            runtime.write('saved.txt', 'workspace survives')
            Path('/opt/session-package').write_text('outside-work survives')
            # A workspace module must not replace trusted code when Python starts again.
            runtime.write('agent_loop/__init__.py', 'raise RuntimeError("workspace code executed as root")')
            runtime.checkpoint([{'role': 'user', 'content': 'remember this conversation'}])
            assert not runtime.run('cat /run/agent-loop/private/live/credentials/environment').ok
            assert not runtime.run('cat /run/agent-loop/private/live/control.sock').ok
            print('PASS create')
        elif mode == 'resume':
            assert runtime.read_text('saved.txt') == 'workspace survives'
            assert Path('/opt/session-package').read_text() == 'outside-work survives'
            assert runtime.conversation['messages'][0]['content'] == 'remember this conversation'
            assert not runtime.run('cat /run/agent-loop/private/live/credentials/environment').ok
            print('PASS resume')
        elif mode == 'mount':
            if not runtime.conversation.get('continue'):
                readonly = runtime.mount_volume(sys.argv[2])
                writable = runtime.mount_volume(sys.argv[3], read_only=False)
                runtime.checkpoint([{'role': 'user', 'content': 'mount once'}, {'role': 'assistant', 'content': json.dumps([readonly, writable])}], continuing=True)
                print('PASS mount requested')
                return 75
            readonly, writable = json.loads(runtime.conversation['messages'][-1]['content'])
            assert runtime.read_text(readonly + '/input.txt') == 'host input'
            try:
                runtime.write(readonly + '/bad.txt', 'forbidden')
            except PermissionError:
                pass
            else:
                raise AssertionError('read-only mount writable')
            assert not runtime.run('touch ' + readonly + '/bad.txt').ok
            runtime.write(writable + '/output.txt', 'host output')
            snapshot = runtime.snapshot()
            runtime.reset(snapshot)
            assert runtime.read_text(writable + '/output.txt') == 'host output'
            assert readonly not in runtime.run('git ls-files').stdout
            runtime.checkpoint([{'role': 'assistant', 'content': 'mounts survived restart'}])
            print('PASS mounted folders and reset isolation')
        elif mode == 'signal':
            runtime.install_signal_handlers()
            runtime.in_turn = True
            print('tool running', flush=True)
            result = runtime.run('sleep 1; echo completed > signal.txt')
            assert result.ok and runtime.stop_requested
            runtime.checkpoint([{'role': 'assistant', 'content': 'tool completed before shutdown'}])
            runtime.in_turn = False
            print('PASS graceful tool completion')
        else:
            raise ValueError('unknown fixture mode')
        return 0
    finally:
        runtime.teardown()


if __name__ == '__main__':
    raise SystemExit(main())
