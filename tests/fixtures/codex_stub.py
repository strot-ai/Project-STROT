"""Fake CLI for process tests. It makes no model or network calls."""

import json
import os
import sys
from pathlib import Path

prompt = sys.stdin.read()
with Path(os.environ["STROT_TEST_CLI_LOG"]).open("a", encoding="utf-8") as log:
    log.write(json.dumps({"prompt": prompt, "args": sys.argv[1:], "cwd": os.getcwd()}) + "\n")
response = (
    Path(os.environ["STROT_TEST_CLI_RESPONSE"]).read_text(encoding="utf-8")
    if "## PREVIOUS ATTEMPT (REJECTED)" in prompt
    else "```python\ndef route_context(record):\n    return (\n```"
)
output = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
output.write_text(response, encoding="utf-8")
print("Progress: café. This is not the model response.")
print("Startup notice", file=sys.stderr)
