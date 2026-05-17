# pipeline_utils.py
import json, os
from datetime import datetime

MANIFEST_PATH = "../data/pipeline_results.json"

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"pipeline_version": "1.0", "run_timestamp": datetime.now().isoformat(), "steps": {}}

def save_step(step_key, data):
    m = load_manifest()
    m["steps"][step_key] = {"status": "complete", "timestamp": datetime.now().isoformat(), **data}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  Manifest updated → {MANIFEST_PATH}")