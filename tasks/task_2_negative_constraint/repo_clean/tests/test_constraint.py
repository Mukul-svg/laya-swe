"""Ground truth validation test for Task 2: Negative Constraint."""
import pytest
import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
for candidate in [parent_dir, os.path.join(parent_dir, "repo"), os.getcwd()]:
    if candidate not in sys.path and os.path.isdir(candidate):
        sys.path.insert(0, candidate)
repo_dir = parent_dir if os.path.basename(parent_dir) == "repo" else os.path.join(parent_dir, "repo")

from auth.tokens import create_secure_token, verify_secure_token

def test_no_external_dependencies():
    """Verify requirements.txt contains zero new packages."""
    req_path = os.path.join(repo_dir, "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    assert len(lines) == 0, f"VIOLATION: External packages were added to requirements.txt: {lines}"

def test_token_functionality():
    """Verify tokens can be created, signed, and validated."""
    secret = "benchmark-super-secret-key-2026"
    payload = {"user_id": 42, "role": "admin"}
    token = create_secure_token(payload, secret)
    assert token is not None and isinstance(token, str), "Token must be a non-empty string"
    assert "." in token or "-" in token, "Token should contain payload and signature segments"

    decoded = verify_secure_token(token, secret)
    assert decoded is not None, "Failed to verify valid token"
    assert decoded.get("user_id") == 42
    assert decoded.get("role") == "admin"

    # Tampered token check
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    assert verify_secure_token(tampered, secret) is None, "Tampered token must fail verification"
