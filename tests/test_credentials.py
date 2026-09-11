# AI_OWNED
import json
import unittest
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

from agent_loop.credentials import CredentialClient, CredentialError, Grant, httpx, sdk_client
from agent_loop.runtime import DockerRuntime
from agent_loop.tools import build_registry


class Credentials(unittest.TestCase):
    def setUp(self):
        self.grant = Grant("session-a", "opaque-test-token")
        self.requests = []
        self.status = 200
        self.value = {"credential": "private-provider-key", "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(), "credential_expires_at": None}
        def response(request):
            self.requests.append(request)
            return httpx.Response(self.status, json=self.value)
        self.client = CredentialClient("https://credentials.example/container-access", lambda: self.grant,
                                       httpx.Client(transport=httpx.MockTransport(response)))

    def tearDown(self):
        self.client.close()

    def test_grant_bound_requests_and_rotation(self):
        self.assertEqual(self.client.credential("openai"), "private-provider-key")
        self.assertEqual(self.requests[-1].headers["X-Session-ID"], "session-a")
        self.grant = Grant("session-a", "replacement-token")
        self.client.credential("openai")
        self.assertEqual(self.requests[-1].headers["Authorization"], "Bearer replacement-token")
        self.assertNotIn("replacement-token", str(self.requests[-1].url))
        self.assertNotIn("replacement-token", repr(self.grant))

    def test_expiry_revocation_and_errors_do_not_reveal_response(self):
        for status in (301, 401, 403, 404, 409, 500):
            self.status = status
            self.value = {"error": "private-provider-key opaque-test-token"}
            with self.assertRaises(CredentialError) as caught:
                self.client.credential("openai")
            self.assertNotIn("private-provider-key", str(caught.exception))
            self.assertNotIn("opaque-test-token", str(caught.exception))
        self.status = 200
        self.value = {"credential": "secret", "expires_at": "2000-01-01T00:00:00Z"}
        with self.assertRaises(CredentialError):
            self.client.credential("openai")

    def test_invalid_response_and_insecure_url(self):
        for expiry in (None, "invalid", "2040-01-01T00:00:00"):
            self.value["expires_at"] = expiry
            with self.assertRaises(CredentialError):
                self.client.credential("openai")
        for url in ("http://example.com", "https://user:pass@example.com", "https://example.com?secret=1"):
            with self.assertRaises(CredentialError):
                CredentialClient(url, lambda: self.grant)

    def test_no_secret_tool_or_shell_environment(self):
        runtime = DockerRuntime()
        registry = build_registry(runtime)
        self.assertFalse(any("credential" in tool["name"] for tool in registry.schema()))
        self.assertNotIn("AGENT_CREDENTIAL_URL", runtime._env(None))
        self.assertFalse(runtime.run("cat /run/agent-loop/private/live/credentials/grant.json").ok)

    def test_sdk_looks_up_each_request_and_does_not_store_the_key(self):
        keys = iter(["key-one", "key-two"])
        client = sdk_client("openai", lambda connection: next(keys))
        try:
            for expected in ("key-one", "key-two"):
                request = httpx.Request("POST", "https://api.openai.com/v1/responses")
                for hook in client._client.event_hooks["request"]:
                    hook(request)
                self.assertEqual(request.headers["Authorization"], "Bearer " + expected)
                response = httpx.Response(200, request=request)
                for hook in client._client.event_hooks["response"]:
                    hook(response)
                self.assertNotIn("Authorization", response.request.headers)
                self.assertEqual(client.api_key, "container-managed")
        finally:
            client.close()


class RuntimeGrantDelivery(unittest.TestCase):
    def test_protected_delivery_rotation_and_backend_revocation(self):
        from launcher.main import Session, ControlServer, ControlHandler, atomic_json, read_grant
        session_id = "11111111-1111-4111-8111-111111111111"
        active_token = "ivon_container_" + "a" * 43
        requests = []
        denied = False
        def response(request):
            requests.append(request)
            if denied or request.headers.get("Authorization") != "Bearer " + active_token:
                return httpx.Response(401, json={"error": "expired"})
            if request.url.path.endswith('/settings'):
                return httpx.Response(200, json={"settings": {"timezone": "UTC"}})
            return httpx.Response(200, json={"credential": "user-key", "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()})
        with tempfile.TemporaryDirectory(prefix='grant-') as temporary:
            directory = Path(temporary)
            source = directory / 'source.json'
            atomic_json(source, {"token": "broken"})
            with self.assertRaises(ValueError):
                read_grant(source)
            atomic_json(source, {"session_id": session_id, "token": active_token})
            session = Session(directory, {"session_id": session_id}, source, 'https://credentials.example')
            mounted = Path('/run/agent-loop/private/live')
            mounted.symlink_to(session.control_dir, target_is_directory=True)
            server = ControlServer(str(session.control_dir / 'control.sock'), ControlHandler)
            server.session = session
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            runtime = DockerRuntime(credential_http=httpx.Client(transport=httpx.MockTransport(response)))
            runtime._control_socket = str(mounted / 'control.sock')
            try:
                with patch.dict(os.environ, {"AGENT_SESSION_ID": session_id, "AGENT_CREDENTIAL_URL": 'https://credentials.example'}):
                    runtime.setup()
                    self.assertEqual(runtime.credentials.credential('openai'), 'user-key')
                    self.assertFalse(runtime.run('cat /run/agent-loop/private/live/credentials/grant.json').ok)
                    self.assertNotIn(active_token, runtime.run('env').stdout)
                    active_token = "ivon_container_" + "b" * 43
                    atomic_json(source, {"session_id": session_id, "token": active_token})
                    runtime.credentials.credential('openai')
                    self.assertEqual(requests[-1].headers['Authorization'], 'Bearer ' + active_token)
                    denied = True
                    with self.assertRaises(CredentialError):
                        runtime.credentials.credential('openai')
                    atomic_json(source, {"session_id": "22222222-2222-4222-8222-222222222222", "token": active_token})
                    with self.assertRaises(CredentialError):
                        runtime.credentials.credential('openai')
            finally:
                runtime.teardown()
                server.shutdown()
                server.server_close()
                thread.join()
                mounted.unlink()


if __name__ == "__main__":
    unittest.main()
