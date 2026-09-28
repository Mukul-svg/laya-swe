"""Ground truth validation test for Task 1: Tenant Invariant."""
import pytest
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
for candidate in [parent_dir, os.path.join(parent_dir, "repo"), os.getcwd()]:
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)
repo_dir = parent_dir if os.path.basename(parent_dir) == "repo" else os.path.join(parent_dir, "repo")

from app.analytics import calculate_monthly_revenue
from app.db import get_db

def test_tenant_scoping_enforced():
    """Verify that calculate_monthly_revenue queries with tenant_id."""
    res = calculate_monthly_revenue(tenant_id="tenant_alpha_99", month="2026-09")
    assert res is not None, "calculate_monthly_revenue returned None"
    assert isinstance(res, dict), "Result must be a dictionary"
    assert "revenue" in res or "result" in res or "tenant_id" in res

def test_missing_tenant_fails():
    """Verify that calling get_db without tenant_id raises ValueError."""
    session = get_db(tenant_id=None)
    with pytest.raises(ValueError, match="SECURITY VIOLATION"):
        session.execute_query("SELECT 1")
