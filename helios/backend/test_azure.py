import os
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

# 1. Correct Endpoint for GitHub Models
endpoint = "https://models.inference.ai.azure.com"

# 2. Use a valid, available model (e.g., gpt-4o-mini)
model_name = "gpt-4o-mini"

# Ensure token is present
token = os.environ.get("GITHUB_TOKEN")
if not token:
    print("✗ GITHUB_TOKEN environment variable not set.")
    sys.exit(1)

print("→ Initializing GitHub Models client via Azure SDK...")
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

print(f"→ Sending request using model: {model_name}...")
try:
    response = client.complete(
        # FIX: Explicitly pass content= to the Message objects
        messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the capital of France?"),
        ],
        model=model_name
    )

    print("\n✓ Success!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"\n✗ Call failed: {e}")