# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


import json
import traceback
import http.client
from typing import Any, Tuple
from urllib.parse import urlparse


class HttpsApi:
    def __init__(self, api_url, key, model, embed_url=None, timeout=60, **kwargs):
        """
        Initialize the HttpsApi class.

        Args:
            api_url (str): API endpoint, supports multiple formats:
                - Full URL: "https://api.openai.com/v1/chat/completions"
                - Hostname only: "api.openai.com" (defaults to /v1/chat/completions)
            key (str): API key for authentication.
            model (str): Model name to use (e.g., "gpt-4o").
            embed_url (str | None): Embedding API URL (optional), supports:
                - Full URL: "https://api.openai.com/v1/embeddings"
                - Path only: "/v1/embeddings"
                - Auto-inferred if not provided
            timeout (int): Request timeout in seconds (default: 60).
            **kwargs (Any): Additional keyword arguments (e.g., temperature).

        Example:
            >>> # Using full URL
            >>> api = HttpsApi(
            ...     api_url="https://api.openai.com/v1/chat/completions",
            ...     key="sk-xxx",
            ...     model="gpt-4o"
            ... )
            >>> # Using hostname only
            >>> api = HttpsApi(
            ...     api_url="api.openai.com",
            ...     key="sk-xxx",
            ...     model="gpt-4o"
            ... )
        """
        # Parse the main API URL
        if api_url.startswith(("http://", "https://")):
            # Full URL with protocol
            parsed = urlparse(api_url)
            self._host = parsed.netloc
            self._url = parsed.path or "/v1/chat/completions"

            # Validate host
            if not self._host:
                raise ValueError(f"Invalid API URL: missing hostname in '{api_url}'")
        else:
            # Check if it looks like a URL without protocol (e.g., "api.openai.com/v1/chat/completions")
            if "/" in api_url:
                raise ValueError(
                    f"Invalid API URL: '{api_url}'\n"
                    f"Did you forget the protocol? Try: 'https://{api_url}'"
                )

            # Plain hostname (e.g., "api.openai.com" or "ai.api.xn--fiqs8s")
            self._host = api_url.strip()
            self._url = "/v1/chat/completions"

            # Basic hostname validation
            if not self._host:
                raise ValueError("API URL cannot be empty")
            if " " in self._host:
                raise ValueError(f"Invalid hostname: '{api_url}' contains spaces")

        # Handle embedding URL
        if embed_url:
            if embed_url.startswith(("http://", "https://")):
                # Full URL with protocol
                embed_parsed = urlparse(embed_url)
                self._embed_url = embed_parsed.path or "/v1/embeddings"

                # Validate path is not empty
                if not self._embed_url or self._embed_url == "/":
                    self._embed_url = "/v1/embeddings"
            else:
                # Plain path or potential mistake
                embed_url = embed_url.strip()

                # Validate not empty
                if not embed_url:
                    raise ValueError("Embedding URL cannot be empty")

                # Check if it looks like a hostname without protocol (e.g., "api.openai.com/v1/embeddings")
                if not embed_url.startswith("/") and "." in embed_url.split("/")[0]:
                    raise ValueError(
                        f"Invalid embedding URL: '{embed_url}'\n"
                        f"Did you forget the protocol? Try: 'https://{embed_url}'\n"
                        f"Or use path format: '/{embed_url}'"
                    )

                # Plain path
                self._embed_url = (
                    embed_url if embed_url.startswith("/") else f"/{embed_url}"
                )
        else:
            # Auto-infer embedding URL
            self._embed_url = "/v1/embeddings"

        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._max_retry = 10

    def get_response(self, prompt: str | Any, *args, **kwargs) -> Tuple[str, dict]:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt.strip()}]

        retry = 0
        while True:
            try:
                if self._model.startswith("o1-preview"):
                    for p in prompt:
                        if p["role"] == "system":
                            p["role"] = "user"

                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps(
                    {
                        # 'max_tokens': self._kwargs.get('max_tokens', 4096),
                        # 'top_p': self._kwargs.get('top_p', None),
                        "temperature": self._kwargs.get("temperature", 1.0),
                        "model": self._model,
                        "messages": prompt,
                    }
                )
                headers = {
                    "Authorization": f"Bearer {self._key}",
                    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                    "Content-Type": "application/json",
                }
                conn.request("POST", self._url, payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data["choices"][0]["message"]["content"]
                usage = data["usage"]
                # if self._model.startswith('claude'):
                #     response = data['content'][0]['text']
                # else:
                #     response = data['choices'][0]['message']['content']
                return response, usage
            except Exception:
                retry += 1
                if retry >= self._max_retry:
                    raise RuntimeError(
                        # f'{self.__class__.__name__} error: {traceback.format_exc()}.\n'
                        "Model Response Error! You may check your API host and API key."
                    )
                else:
                    print("Model Response Error! Retrying...")
                    # print(f'{self.__class__.__name__} error: {traceback.format_exc()}. Retrying...\n')

    def get_embedding(self, text: str | Any, *args, **kwargs) -> str:
        content_embedding = {"input": text, "model": self._model}

        retry = 0
        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps(content_embedding)
                headers = {
                    "Authorization": f"Bearer {self._key}",
                    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
                    "Content-Type": "application/json",
                }
                conn.request("POST", self._embed_url, payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data["data"][0]["embedding"]
                # if self._model.startswith('claude'):
                #     response = data['content'][0]['text']
                # else:
                #     response = data['choices'][0]['message']['content']
                return response
            except Exception:
                retry += 1
                if retry >= self._max_retry:
                    raise RuntimeError(
                        f"{self.__class__.__name__} error: {traceback.format_exc()}.\n"
                        f"You may check your API host and API key."
                    )
                else:
                    print(
                        f"{self.__class__.__name__} error: {traceback.format_exc()}. Retrying...\n"
                    )
