# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Tests for the OpenAI-compatible HTTPS client wrapper."""

import json

import pytest

import evotoolkit.tools.llm as llm_module
from evotoolkit.tools import HttpsApi


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {"content-length": str(len(json.dumps(payload)))}

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RecordingConnection:
    instances = []
    payload = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def __init__(self, host, timeout=None):
        self.host = host
        self.timeout = timeout
        self.requests = []
        RecordingConnection.instances.append(self)

    def request(self, method, url, payload, headers):
        self.requests.append((method, url, json.loads(payload), headers))

    def getresponse(self):
        return FakeHTTPResponse(self.payload)


class FlakyConnection(RecordingConnection):
    attempts = 0

    def getresponse(self):
        FlakyConnection.attempts += 1
        if FlakyConnection.attempts == 1:
            raise RuntimeError("temporary error")
        return super().getresponse()


class AlwaysFailingConnection(RecordingConnection):
    def getresponse(self):
        raise RuntimeError("permanent error")


class TestHttpsApiInit:
    def test_accepts_full_url(self):
        api = HttpsApi(
            api_url="https://api.openai.com/v1/chat/completions",
            key="k",
            model="gpt-4o",
            embed_url="https://api.openai.com/v1/embeddings",
        )

        assert api._host == "api.openai.com"
        assert api._url == "/v1/chat/completions"
        assert api._embed_url == "/v1/embeddings"

    def test_accepts_hostname_only(self):
        api = HttpsApi(api_url="api.openai.com", key="k", model="gpt-4o")

        assert api._host == "api.openai.com"
        assert api._url == "/v1/chat/completions"
        assert api._embed_url == "/v1/embeddings"

    def test_rejects_url_without_protocol(self):
        with pytest.raises(ValueError, match="Did you forget the protocol"):
            HttpsApi(api_url="api.openai.com/v1/chat/completions", key="k", model="gpt-4o")

    def test_rejects_invalid_embedding_url_without_protocol(self):
        with pytest.raises(ValueError, match="Did you forget the protocol"):
            HttpsApi(
                api_url="api.openai.com",
                key="k",
                model="gpt-4o",
                embed_url="api.openai.com/v1/embeddings",
            )


class TestHttpsApiRequests:
    def test_get_response_posts_to_expected_endpoint(self, monkeypatch):
        RecordingConnection.instances.clear()
        RecordingConnection.payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        monkeypatch.setattr(llm_module.http.client, "HTTPSConnection", RecordingConnection)

        api = HttpsApi(api_url="api.openai.com", key="secret", model="gpt-4o", temperature=0.3)
        response, usage = api.get_response("hello world")

        assert response == "ok"
        assert usage["total_tokens"] == 2

        conn = RecordingConnection.instances[-1]
        method, url, payload, headers = conn.requests[-1]
        assert method == "POST"
        assert url == "/v1/chat/completions"
        assert payload["messages"] == [{"role": "user", "content": "hello world"}]
        assert payload["temperature"] == 0.3
        assert headers["Authorization"] == "Bearer secret"

    def test_get_response_rewrites_system_role_for_o1_preview(self, monkeypatch):
        RecordingConnection.instances.clear()
        RecordingConnection.payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        monkeypatch.setattr(llm_module.http.client, "HTTPSConnection", RecordingConnection)

        api = HttpsApi(api_url="api.openai.com", key="secret", model="o1-preview-test")
        prompt = [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]

        api.get_response(prompt)

        conn = RecordingConnection.instances[-1]
        _, _, payload, _ = conn.requests[-1]
        assert payload["messages"][0]["role"] == "user"
        assert prompt[0]["role"] == "user"

    def test_get_response_retries_after_temporary_failure(self, monkeypatch):
        FlakyConnection.instances.clear()
        FlakyConnection.attempts = 0
        FlakyConnection.payload = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        monkeypatch.setattr(llm_module.http.client, "HTTPSConnection", FlakyConnection)

        api = HttpsApi(api_url="api.openai.com", key="secret", model="gpt-4o")
        response, usage = api.get_response("retry please")

        assert response == "ok"
        assert usage["total_tokens"] == 2
        assert FlakyConnection.attempts == 2

    def test_get_response_raises_after_max_retries(self, monkeypatch):
        monkeypatch.setattr(llm_module.http.client, "HTTPSConnection", AlwaysFailingConnection)

        api = HttpsApi(api_url="api.openai.com", key="secret", model="gpt-4o")
        api._max_retry = 2

        with pytest.raises(RuntimeError, match="Model Response Error"):
            api.get_response("still failing")

    def test_get_embedding_uses_embedding_endpoint(self, monkeypatch):
        RecordingConnection.instances.clear()
        RecordingConnection.payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        monkeypatch.setattr(llm_module.http.client, "HTTPSConnection", RecordingConnection)

        api = HttpsApi(
            api_url="api.openai.com",
            key="secret",
            model="text-embedding-3-small",
            embed_url="/v1/embeddings",
        )
        embedding = api.get_embedding("hello")

        assert embedding == [0.1, 0.2, 0.3]
        conn = RecordingConnection.instances[-1]
        method, url, payload, _ = conn.requests[-1]
        assert method == "POST"
        assert url == "/v1/embeddings"
        assert payload["input"] == "hello"
