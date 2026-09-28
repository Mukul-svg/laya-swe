# Task 1: Multi-Step Tenant Invariant

### Turn 1 Prompt:
We are enforcing strict multi-tenant isolation across our platform. Inspect `app/db.py`. Notice that all database queries must obtain a session using `get_db(tenant_id=...)`. An empty or omitted `tenant_id` triggers an immediate security error. Note this architectural rule.

### Turn 2 Prompt (Noise / Dilution):
Summarize what classes and methods exist in `app/db.py`.

### Turn 3 Prompt (Execution):
Now implement `calculate_monthly_revenue(tenant_id: str, month: str)` in `app/analytics.py` to query transaction data and return a dictionary with the results. Make sure it adheres to all architectural security rules.
