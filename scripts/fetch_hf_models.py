"""Download the Hugging Face models Rag_Lite uses, at their pinned revisions.

HF_HUB_OFFLINE=0 python scripts/fetch_hf_models.py
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"  # must be set before huggingface_hub is imported

from gqa.hf_models import PINNED, fetch_pinned, verify_pins  # noqa: E402

fetch_pinned()
problems = verify_pins()
for repo, sha in PINNED.items():
    print(f"{repo} @ {sha}")
print("\n".join(problems) if problems else "all pinned revisions cached")
sys.exit(1 if problems else 0)
