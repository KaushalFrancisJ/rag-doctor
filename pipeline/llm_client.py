"""Centralized LLM client abstraction for completions and JSON responses."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"


class LLMClient:
    """Centralized client for LLM completions with automated retries and JSON support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self._client = None

    @property
    def client(self):
        """Lazy-load Groq API client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set.")
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        max_retries: int = 5,
    ) -> str:
        """Execute a chat completion request with exponential backoff on rate limits."""
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str or "retry" in err_str) and attempt < max_retries - 1:
                    wait_sec = 4.0 * (attempt + 1)
                    m_min = re.search(r"try again in (\d+)m([\d\.]+)s", str(e), re.I)
                    m_sec = re.search(r"try again in ([\d\.]+)s", str(e), re.I)
                    m_ms = re.search(r"try again in ([\d\.]+)ms", str(e), re.I)
                    if m_min:
                        wait_sec = int(m_min.group(1)) * 60 + float(m_min.group(2)) + 1.0
                    elif m_sec:
                        wait_sec = float(m_sec.group(1)) + 1.0
                    elif m_ms:
                        wait_sec = float(m_ms.group(1)) / 1000.0 + 1.0
                    time.sleep(wait_sec)
                else:
                    raise

        if response and response.choices:
            return response.choices[0].message.content or ""
        return ""

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call LLM and parse output as structured JSON dictionary."""
        content = self.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        if not content:
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if m:
                return json.loads(m.group(1))
            raise
