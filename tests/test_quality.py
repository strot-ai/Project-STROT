"""Optional quality check against a local Julia checkpoint. No downloads."""

import json
import os
import runpy
from pathlib import Path

import pytest

import strot


def test_support_ticket_quality(monkeypatch, tmp_path):
    checkpoint = os.environ.get("STROT_TEST_CHECKPOINT")
    if not checkpoint:
        pytest.skip("Set STROT_TEST_CHECKPOINT to run the real Julia quality check")
    assert (Path(checkpoint) / "model.safetensors").is_file()
    monkeypatch.setenv("STROT_NLU_CHECKPOINT", checkpoint)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    fixtures = Path(__file__).parent / "fixtures"
    cases = json.loads((fixtures / "support_tickets.json").read_text(encoding="utf-8"))
    response = (fixtures / "support_paths.md").read_text(encoding="utf-8")
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/end_to_end.py"))
    result = strot.run([case["record"] for case in cases], example["GOAL"],
                       example["ACTION_REGISTRY"], llm=lambda _: response, state=tmp_path)
    queues = {"billing_dispute": "billing", "technical_defect": "engineering", "account_lifecycle": "privacy"}
    actual = {item["record"]["ticket_id"]: (item["action"], item["result"]["queue"])
              for item in result.outcomes}
    expected = {case["record"]["ticket_id"]: (case["expect"], queues[case["expect"]]) for case in cases}
    assert not result.fallbacks
    assert actual == expected
    assert all(0 <= item["confidence"] <= 1 for item in result.outcomes)
