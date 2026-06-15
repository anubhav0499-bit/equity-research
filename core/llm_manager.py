"""
LLM Manager — provider-agnostic interface supporting OpenAI, Anthropic, Groq,
Google Gemini, Together, OpenRouter, Ollama, and LM Studio.
Auto-detects available provider via API key presence.
"""

from __future__ import annotations
import time
import json
from typing import Optional
from loguru import logger
from .config import (
    LLM_CONFIG, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY,
    GOOGLE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY, PROVIDER_MODELS
)


class LLMManager:
    def __init__(self):
        self.backend, self.primary_model, self.fast_model = self._detect_backend()
        logger.info(f"LLM backend: {self.backend} | primary={self.primary_model} | fast={self.fast_model}")

    def _detect_backend(self) -> tuple[str, str, str]:
        forced = LLM_CONFIG.provider
        if forced != "auto":
            models = PROVIDER_MODELS.get(forced, {})
            return forced, models.get("primary", ""), models.get("fast", "")

        checks = [
            ("groq",       GROQ_API_KEY),
            ("openai",     OPENAI_API_KEY),
            ("anthropic",  ANTHROPIC_API_KEY),
            ("together",   TOGETHER_API_KEY),
            ("openrouter", OPENROUTER_API_KEY),
        ]
        for name, key in checks:
            if key:
                models = PROVIDER_MODELS[name]
                return name, models["primary"], models["fast"]

        if GOOGLE_API_KEY:
            models = PROVIDER_MODELS["gemini"]
            return "gemini", models["primary"], models["fast"]

        # Try Ollama
        try:
            import httpx
            r = httpx.get(f"{LLM_CONFIG.ollama_base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                models = PROVIDER_MODELS["ollama"]
                return "ollama", models["primary"], models["fast"]
        except Exception:
            pass

        return "template", "none", "none"

    def get_backend_info(self) -> str:
        return f"{self.backend} ({self.primary_model})"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = None,
        use_fast_model: bool = False,
        json_mode: bool = False,
    ) -> str:
        model = self.fast_model if use_fast_model else self.primary_model
        temp = temperature if temperature is not None else LLM_CONFIG.temperature

        for attempt in range(LLM_CONFIG.max_retries):
            try:
                if self.backend == "openai":
                    return self._openai(prompt, system_prompt, model, max_tokens, temp, json_mode)
                elif self.backend == "anthropic":
                    return self._anthropic(prompt, system_prompt, model, max_tokens, temp)
                elif self.backend == "groq":
                    return self._groq(prompt, system_prompt, model, max_tokens, temp)
                elif self.backend == "gemini":
                    return self._gemini(prompt, system_prompt, model, max_tokens, temp)
                elif self.backend in ("together", "openrouter"):
                    return self._openai_compatible(prompt, system_prompt, model, max_tokens, temp)
                elif self.backend == "ollama":
                    return self._ollama(prompt, system_prompt, model, max_tokens, temp)
                else:
                    return self._template_mode(prompt, system_prompt)
            except Exception as e:
                if attempt < LLM_CONFIG.max_retries - 1:
                    wait = LLM_CONFIG.retry_delay * (2 ** attempt)
                    logger.warning(f"LLM attempt {attempt+1} failed ({e}). Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"LLM failed after {LLM_CONFIG.max_retries} attempts: {e}")
                    return f"[LLM ERROR: {e}]"

    def generate_json(self, prompt: str, system_prompt: str = "", max_tokens: int = 2048) -> dict:
        raw = self.generate(prompt, system_prompt, max_tokens, json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
        return {"error": "json_parse_failed", "raw": raw[:500]}

    # ── Provider implementations ──────────────────────────────────

    def _openai(self, prompt, system, model, max_tokens, temp, json_mode) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_CONFIG.timeout)
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are an expert financial analyst."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temp,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""

    def _anthropic(self, prompt, system, model, max_tokens, temp) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=LLM_CONFIG.timeout)
        r = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temp,
            system=system or "You are an expert financial analyst.",
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text if r.content else ""

    def _groq(self, prompt, system, model, max_tokens, temp) -> str:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY, timeout=LLM_CONFIG.timeout)
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are an expert financial analyst."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content or ""

    def _gemini(self, prompt, system, model, max_tokens, temp) -> str:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        m = genai.GenerativeModel(
            model_name=model,
            system_instruction=system or "You are an expert financial analyst.",
        )
        r = m.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=temp),
        )
        return r.text or ""

    def _openai_compatible(self, prompt, system, model, max_tokens, temp) -> str:
        from openai import OpenAI
        base_map = {
            "together":    ("https://api.together.xyz/v1",    TOGETHER_API_KEY),
            "openrouter":  ("https://openrouter.ai/api/v1",   OPENROUTER_API_KEY),
        }
        base_url, api_key = base_map.get(self.backend, ("", ""))
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_CONFIG.timeout)
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system or "You are an expert financial analyst."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content or ""

    def _ollama(self, prompt, system, model, max_tokens, temp) -> str:
        import httpx
        payload = {
            "model": model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temp},
        }
        r = httpx.post(
            f"{LLM_CONFIG.ollama_base_url}/api/generate",
            json=payload,
            timeout=LLM_CONFIG.timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")

    def _template_mode(self, prompt, system) -> str:
        logger.warning("No LLM backend available — returning template response")
        return (
            "[TEMPLATE MODE — No LLM configured]\n"
            "Install an LLM provider: pip install openai / anthropic / groq\n"
            f"Prompt length: {len(prompt)} chars"
        )
