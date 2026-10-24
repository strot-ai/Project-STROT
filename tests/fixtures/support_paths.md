Here is the ticket router:

```python
ROUTING_LABELS = {
    "billing_dispute": "billing invoice double charge refund",
    "technical_defect": "app crash bug broken export",
    "account_lifecycle": "account deletion personal data privacy",
}


def route_context(record: dict) -> str:
    return str(record.get("body") or "")


def execute_billing_dispute(record: dict) -> dict:
    return {"queue": "billing", "ticket_id": record.get("ticket_id")}


def execute_technical_defect(record: dict) -> dict:
    return {"queue": "engineering", "ticket_id": record.get("ticket_id")}


def execute_account_lifecycle(record: dict) -> dict:
    return {"queue": "privacy", "ticket_id": record.get("ticket_id")}
```
