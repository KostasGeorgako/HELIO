"""
LLM client wrapper — GitHub Models (via azure-ai-inference SDK).

GitHub Models exposes OpenAI/Anthropic/Mistral/etc models behind a single
endpoint, authenticated with a GitHub personal access token. Free tier
covers anything a hackathon needs.

Public API is identical to the previous Azure version, so nothing else
in the project needs to change.

Environment variables:
  GITHUB_TOKEN          (required for real calls — a GitHub PAT)
  GITHUB_MODELS_ENDPOINT defaults to "https://models.inference.ai.azure.com"
  GITHUB_MODEL          defaults to "gpt-4o-mini"

If GITHUB_TOKEN is missing, deterministic mocks fire so the rest of the
app stays debuggable.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
from typing import Any

from prompts import PROMPT_SYSTEM

# ── Config ────────────────────────────────────────────────────────────────

TOKEN    = os.environ.get("GITHUB_TOKEN")
ENDPOINT = os.environ.get("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com")
MODEL    = os.environ.get("GITHUB_MODEL", "gpt-4o-mini")
MAX_TOKENS = 1024

_client = None
_SystemMessage = None
_UserMessage = None
_have_token = bool(TOKEN)

if _have_token:
    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential
        _client = ChatCompletionsClient(
            endpoint=ENDPOINT,
            credential=AzureKeyCredential(TOKEN),
        )
        _SystemMessage = SystemMessage
        _UserMessage = UserMessage
        print(f"[llm] GitHub Models ready · {MODEL} @ {ENDPOINT}")
    except Exception as e:
        print(f"[llm] azure-ai-inference unavailable: {e}")
        _have_token = False
else:
    print("[llm] no GITHUB_TOKEN set — using deterministic mocks")


# ── Timeout guard ─────────────────────────────────────────────────────────
# Network calls to GitHub Models have no built-in timeout — a slow or
# rate-limited endpoint would hang the whole request forever. Run every
# completion in a worker thread with a hard deadline; on timeout we raise and
# the caller falls back to a deterministic mock.

LLM_TIMEOUT_S = 30
_LLM_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=6)


def _complete(**kwargs):
    """`_client.complete(**kwargs)` with a hard timeout."""
    return _LLM_POOL.submit(_client.complete, **kwargs).result(timeout=LLM_TIMEOUT_S)


# ── Public API ────────────────────────────────────────────────────────────

def chat_text(user_prompt: str) -> str:
    """One-shot text completion. Used for HELIO's prose replies."""
    if not _have_token:
        return _mock_text(user_prompt)
    try:
        resp = _complete(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            messages=[
                _SystemMessage(content=PROMPT_SYSTEM),
                _UserMessage(content=user_prompt),
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[llm] chat_text failed, falling back to mock: {e}")
        return _mock_text(user_prompt)


def chat_json(user_prompt: str) -> dict[str, Any]:
    """
    One-shot JSON extraction. The endpoint rejects response_format pinning, so
    we go straight to a plain completion — the prompt demands JSON and
    _extract_json recovers it.
    """
    if not _have_token:
        return _mock_json(user_prompt)

    try:
        resp = _complete(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                _SystemMessage(content=PROMPT_SYSTEM),
                _UserMessage(content=user_prompt),
            ],
        )
        raw = resp.choices[0].message.content or ""
        return _extract_json(raw)
    except Exception as e:
        print(f"[llm] chat_json failed, falling back to mock: {e}")
        return _mock_json(user_prompt)


# ── JSON recovery ─────────────────────────────────────────────────────────

def _extract_json(s: str) -> dict[str, Any]:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = s.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
    raise ValueError(f"could not recover JSON from response: {s[:200]}…")


# ── Mocks ─────────────────────────────────────────────────────────────────

def _mock_text(prompt: str) -> str:
    p = prompt.lower()
    if "welcome message" in p or "warm welcome" in p:
        return (
            "Welcome — I've pulled in the hyperspectral data and can see four "
            "candidate sites with multiple acquisitions across the past year. "
            "Before we dive in, two things: which of these sites would you like "
            "to compare for this investment, and what's the asking price on each? "
            "Once I have those, I'll walk you through what the spectral data is "
            "telling us about the land."
        )
    if "confirmation" in p and "use case" in p:
        return (
            "Got it — I'll analyse the selected sites with that use case in mind. "
            "Running the full hyperspectral pipeline now: spectral exploration, "
            "indices, anomaly detection, and cohort scoring. This typically takes "
            "around 60 to 90 seconds."
        )
    if "confirmation" in p:
        return (
            "Perfect — I'll focus the analysis on those sites at those prices. "
            "Now: what are you planning to do with the land? Olive groves, a solar "
            "farm, agritourism, mineral extraction — any direction will do. Just "
            "tell me in plain language."
        )
    if "investment recommendation" in p or "ranking:" in p.lower():
        return (
            "Based on the hyperspectral signature across the cohort, the leading "
            "site offers the strongest combination of soil quality, vegetation "
            "vigour, and spatial consistency. The anomaly burden is low, meaning "
            "we see few subsurface red flags. For your stated use case, this is "
            "the parcel I'd buy. Risk is moderate and tied principally to "
            "seasonal moisture variability, which a short site visit could verify."
        )
    return "Understood. Continuing the analysis."


def _mock_json(prompt: str) -> dict[str, Any]:
    p = prompt.lower()
    if "extract the site selection" in p or "sites_selected" in p:
        return {
            "sites_selected": ["arkadia", "arkadia2", "magnisia", "veroia"],
            "prices": {"arkadia": 1_000_000, "arkadia2": 1_000_000,
                       "magnisia": 1_000_000, "veroia": 1_000_000},
            "missing_prices": [],
            "needs_clarification": False,
            "clarification_needed": "",
        }
    if "assign scoring weights" in p or "w_soil" in p:
        return {
            "use_case": "agricultural cultivation",
            "reasoning": ("Agricultural use prioritises soil quality and "
                          "vegetation health, with moisture as a supporting "
                          "factor and anomaly burden as a risk guard."),
            "weights": {
                "W_SOIL": 0.30, "W_CLAY": 0.05, "W_MINERAL": 0.00,
                "W_CONSIST": 0.25, "W_VEG": 0.25, "W_MOISTURE": 0.10,
                "W_ANOMALY": 0.05,
            },
            "anomaly_sign": -1,
            "date_discounts": {
                "arkadia": 1.0, "arkadia2": 0.70,
                "magnisia": 1.0, "veroia": 1.0,
            },
        }
    return {}
