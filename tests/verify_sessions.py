# AI_OWNED
"""Host-side orchestration test; all agent execution goes through run.sh into Docker."""
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from launcher.main import Session

ROOT = Path(__file__).resolve().parent.parent


def main():
    directory = Path(tempfile.mkdtemp(prefix='als-', dir='/tmp'))
    source = Path(tempfile.mkdtemp(prefix='alv-', dir='/tmp'))
    writable = source / 'writable'
    readonly = source / 'readonly'
    writable.mkdir()
    readonly.mkdir()
    (readonly / 'input.txt').write_text('host input')
    readonly.chmod(0o700)
    (readonly / 'input.txt').chmod(0o600)
    secret = secrets.token_hex(32)
    env_file = directory / 'test.env'
    env_file.write_text('QWEN_API_KEY=' + secret + '\nMODEL=test-model\n')
    env = {**os.environ, 'XDG_STATE_HOME': str(directory), 'AGENT_TARGET': 'test', 'AGENT_ENTRYPOINT': 'python', 'AGENT_ENV_FILE': str(env_file)}
    images = set()
    try:
        def run(*args, success=True):
            result = subprocess.run([str(ROOT / 'agent-loop'), '--local', *args], env=env, text=True, input='', capture_output=True)
            print(result.stdout)
            if success and result.returncode:
                raise AssertionError(result.stderr)
            return result
        run('--', '-P', '-m', 'tests.session_driver', 'create')
        metadata_path = next((directory / 'agent-loop/sessions').glob('*/metadata.json'))
        metadata = json.loads(metadata_path.read_text())
        images.add(metadata['image_id'])
        identifier = metadata['session_id']
        run('--resume', identifier, '--', '-P', '-m', 'tests.session_driver', 'resume')
        metadata = json.loads(metadata_path.read_text())
        images.add(metadata['image_id'])
        assert secret not in metadata_path.read_text()
        config = subprocess.check_output(['docker', 'image', 'inspect', metadata['image_id']], text=True)
        assert secret not in config and 'QWEN_API_KEY' not in config
        with (metadata_path.parent / metadata['archive']).open('rb') as archive:
            previous = b''
            while chunk := archive.read(1024 * 1024):
                assert secret.encode() not in previous + chunk, 'credential persisted in saved image'
                previous = chunk[-len(secret):]
        run('--', '-P', '-m', 'tests.session_driver', 'mount', str(readonly), str(writable))
        assert (writable / 'output.txt').read_text() == 'host output'
        assert not (readonly / 'bad.txt').exists()
        paths = list((directory / 'agent-loop/sessions').glob('*/metadata.json'))
        mount_metadata = next(json.loads(p.read_text()) for p in paths if p != metadata_path)
        images.add(mount_metadata['image_id'])
        completed = run('--', '-P', '-m', 'tests.session_driver', 'loop_mount', str(readonly))
        assert 'Mounted folder read after restart' in completed.stdout
        process = subprocess.Popen([str(ROOT / 'agent-loop'), '--local', '--', '-P', '-m', 'tests.session_driver', 'signal'], env=env, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout.readline().strip() == 'tool running'
        process.terminate()
        output, error = process.communicate(timeout=60)
        assert process.returncode == 0, error
        assert 'PASS graceful tool completion' in output
        for saved in (directory / 'agent-loop/sessions').glob('*/metadata.json'):
            images.add(json.loads(saved.read_text())['image_id'])
        shutil.rmtree(readonly)
        result = run('--resume', mount_metadata['session_id'], '--', '-P', '-m', 'tests.session_driver', 'resume', success=False)
        assert result.returncode != 0 and 'No such file' in result.stderr

        # Fail the external Docker save operation: retain the container and previous checkpoint.
        metadata = json.loads(metadata_path.read_text())
        session = Session(metadata_path.parent, metadata, None, None)
        container = subprocess.check_output(['docker', 'create', metadata['image_id']], text=True).strip()
        original = metadata_path.read_bytes()
        real_run = subprocess.run
        def fail_save(command, **kwargs):
            if command[:3] == ['docker', 'image', 'save']:
                raise subprocess.CalledProcessError(1, command)
            return real_run(command, **kwargs)
        try:
            with patch('subprocess.run', side_effect=fail_save):
                try:
                    session.save(container, {})
                except subprocess.CalledProcessError:
                    pass
                else:
                    raise AssertionError('save failure was ignored')
            assert metadata_path.read_bytes() == original
            subprocess.run(['docker', 'inspect', container], check=True, stdout=subprocess.DEVNULL)
        finally:
            subprocess.run(['docker', 'rm', container], check=True, stdout=subprocess.DEVNULL)
        print('PASS: real Docker restore, conversation, filesystem outside /work, dynamic mounts, readonly/writable isolation, secret-free archives, missing mounts, and save failure recovery')
    finally:
        for image in images:
            subprocess.run(['docker', 'image', 'rm', image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(directory)
        shutil.rmtree(source)


if __name__ == '__main__':
    main()
