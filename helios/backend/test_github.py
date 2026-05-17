"""
GitHub Models connectivity test.

Run this BEFORE launching uvicorn. If this works, the app will work.

Usage:
    cd backend
    export GITHUB_TOKEN="ghp_..."
    python test_github.py
"""

import os
import sys

TOKEN    = os.environ.get("GITHUB_TOKEN")
ENDPOINT = os.environ.get("GITHUB_MODELS_ENDPOINT", "https://models.inference.ai.azure.com")
MODEL    = os.environ.get("GITHUB_MODEL", "gpt-4o-mini")

print("─" * 60)
print(f"endpoint    : {ENDPOINT}")
print(f"model       : {MODEL}")
print(f"token set   : {bool(TOKEN)} {('(' + TOKEN[:7] + '…' + TOKEN[-4:] + ')') if TOKEN else ''}")
print("─" * 60)

if not TOKEN:
    print("✗ GITHUB_TOKEN not set.")
    print()
    print("To get a token:")
    print("  1. Go to https://github.com/settings/tokens")
    print("  2. Click 'Generate new token (classic)'")
    print("  3. No scopes needed for GitHub Models — leave them all unchecked")
    print("  4. Copy the token starting with 'ghp_'")
    print("  5. export GITHUB_TOKEN=\"ghp_...\"")
    sys.exit(1)

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential
except ImportError as e:
    print(f"✗ azure-ai-inference not installed: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

print()
print("→ initialising client…")
try:
    client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
except Exception as e:
    print(f"✗ client init failed: {e}")
    sys.exit(1)

print("→ calling GitHub Models (1 round-trip)…")
try:
    resp = client.complete(
        model=MODEL,
        max_tokens=80,
        messages=[
            SystemMessage(content="You are a terse assistant. Answer in one short sentence."),
            UserMessage(content="If you can read this, reply with: HELIO is online."),
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    print()
    print(f"✓ response: {text}")
    if resp.usage:
        print(f"✓ tokens used: prompt={resp.usage.prompt_tokens}, "
              f"completion={resp.usage.completion_tokens}")
    print()
    print("✓ GitHub Models is reachable. You are good to launch uvicorn.")
except Exception as e:
    err = str(e)
    print()
    print(f"✗ call failed: {type(e).__name__}")
    print(f"  {err[:300]}")
    print()
    if "401" in err or "Unauthorized" in err:
        print("→ Token rejected. Common causes:")
        print("  · Token expired (PATs have an expiration date)")
        print("  · Token copied with trailing whitespace")
        print("  · Wrong token type (use a 'classic' PAT, not fine-grained)")
    elif "404" in err or "not found" in err.lower():
        print(f"→ Model '{MODEL}' not available on GitHub Models.")
        print("  Try: gpt-4o-mini, gpt-4o, Phi-3-mini-128k-instruct,")
        print("       Mistral-large-2407, Meta-Llama-3.1-70B-Instruct")
    elif "429" in err or "rate" in err.lower():
        print("→ Rate limit hit. GitHub Models free tier has per-minute caps.")
        print("  Wait 60 seconds and retry.")
    sys.exit(1)
