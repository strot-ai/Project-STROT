"""End-to-end STROT walkthrough.

Compile time: introspect a dataset sample, synthesize execution paths with a
large generative LLM, and bake the default Julia-1 local runtime into a
standalone asset. Runtime: load the asset and feed it live records.

Run it:

    # With a real synthesizer LLM (recommended):
    export OPENAI_API_KEY=sk-...
    python examples/end_to_end.py

    # Offline demo (deterministic stub synthesizer, no API key needed):
    python examples/end_to_end.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from strot import FallbackException, StrotCompiler
from strot.compiler import openai_llm_client

# ---------------------------------------------------------------------- #
# 1. A representative sample of the live data stream                      #
# ---------------------------------------------------------------------- #

SUPPORT_TICKETS = [
    {"ticket_id": 101, "created_at": "2026-10-01", "channel": "email", "body": "I was double charged on my invoice", "account_tier": "pro"},
    {"ticket_id": 102, "created_at": "2026-10-02", "channel": "chat", "body": "App crashes when exporting a report", "account_tier": "free"},
    {"ticket_id": 103, "created_at": "2026-10-03", "channel": "email", "body": "Please delete my account and all data", "account_tier": "pro"},
]

GOAL = "Triage incoming support tickets into the correct resolution workflow."

ACTION_REGISTRY = {
    "billing_dispute": "Refund checks, invoice corrections, double-charge investigations",
    "technical_defect": "Crashes, bugs, and broken product behavior",
    "account_lifecycle": "Account deletion, data export, GDPR requests",
}


def stub_synthesizer(prompt: str) -> str:
    """Offline stand-in for GPT-4o/Claude: returns contract-compliant paths."""
    return '''```python
from typing import Any, Dict


def execute_billing_dispute(record: dict) -> dict:
    try:
        return {"queue": "billing", "priority": "high" if record.get("account_tier") == "pro" else "normal", "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "billing", "error": str(exc)}


def execute_technical_defect(record: dict) -> dict:
    try:
        return {"queue": "engineering", "needs_repro": True, "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "engineering", "error": str(exc)}


def execute_account_lifecycle(record: dict) -> dict:
    try:
        return {"queue": "trust_and_safety", "legal_hold_check": True, "ticket_id": record.get("ticket_id")}
    except Exception as exc:
        return {"queue": "trust_and_safety", "error": str(exc)}
```'''


def main() -> None:
    # ------------------------------------------------------------------ #
    # 2. COMPILE TIME: synthesize the self-routing asset                  #
    # ------------------------------------------------------------------ #
    synthesizer = (
        openai_llm_client(model="gpt-4o")
        if os.environ.get("OPENAI_API_KEY")
        else stub_synthesizer
    )

    compiler = StrotCompiler(
        nlu_model="SupersonicLabs/Julia-1",  # the default local CPU runtime
        llm_client=synthesizer,
        max_retries=3,
    )
    asset = compiler.compile(
        sample=SUPPORT_TICKETS,
        goal=GOAL,
        action_registry=ACTION_REGISTRY,
    )
    asset_path = asset.save(Path(__file__).parent / "compiled_pipeline.py")
    print(f"Compiled {asset.runtime_kind} asset in {asset.attempts} attempt(s): {asset_path}")

    # ------------------------------------------------------------------ #
    # 3. RUNTIME: load the standalone asset and feed it live data         #
    # ------------------------------------------------------------------ #
    # Requires the local extra (torch + transformers) for Julia-1:
    #   pip install strot-ai[local]
    try:
        spec = importlib.util.spec_from_file_location("compiled_pipeline", asset_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        pipeline = module.CompiledPipeline()
    except Exception as exc:
        print(f"Runtime not available here ({exc}); asset is still valid for production.")
        sys.exit(0)

    for ticket in SUPPORT_TICKETS:
        try:
            outcome = pipeline.process(ticket)
            print(json.dumps(outcome, default=str))
        except (FallbackException, module.FallbackException) as exc:
            print(f"Ticket {ticket['ticket_id']} routed to fallback: {exc}")


if __name__ == "__main__":
    main()
