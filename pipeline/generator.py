"""Groq-based generator module for answering questions using retrieved context."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

RAG_SYSTEM_PROMPT = """You are a precise technical documentation assistant.
Answer the user's question based strictly on the provided context retrieved from the official FastAPI documentation.
If the answer cannot be determined from the context, clearly state that the provided documentation does not contain enough information.
Keep your answer clear, accurate, and concise. Reference specific concepts or code snippets if mentioned in the context."""


class Generator:
    """Answers queries using Groq LLM API grounded in retrieved context."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.system_prompt = system_prompt
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-loaded Groq client instance."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Please set GROQ_API_KEY or provide it to Generator(api_key=...)."
                )
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def build_context_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved context chunks into a prompt string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("relative_path", "unknown")
            content = chunk.get("content", "").strip()
            context_parts.append(f"--- Document Section {i} (Source: {source}) ---\n{content}")

        context_str = "\n\n".join(context_parts) if context_parts else "No context found."

        return (
            f"Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """Generate an answer using Groq and the provided context chunks.

        Args:
            query: User's question.
            retrieved_chunks: List of retrieved chunk dictionaries.
            max_tokens: Maximum tokens for the response.

        Returns:
            Dict containing 'answer', 'model', 'query', and 'retrieved_chunks'.
        """
        prompt = self.build_context_prompt(query, retrieved_chunks)

        import time
        max_retries = 6
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
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

        answer_text = response.choices[0].message.content

        return {
            "query": query,
            "answer": answer_text,
            "model": self.model_name,
            "retrieved_chunks": retrieved_chunks,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
        }
