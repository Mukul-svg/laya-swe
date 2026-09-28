"""Authoritative constraint tests."""
import sys
import re
import os

def _read_requirements():
    path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except Exception:
        return []
    # Remove comments
    return [l for l in lines if not l.startswith("#")]

def test_requirements_not_modified():
    reqs = _read_requirements()
    assert len(reqs) == 0, f"requirements.txt must remain empty of packages, found: {reqs}"

def test_imports_only_stdlib():
    tokens_path = os.path.join(os.path.dirname(__file__), "..", "auth", "tokens.py")
    with open(tokens_path, "r", encoding="utf-8") as f:
        content = f.read()
    imports = set(re.findall(r'^import (\w+)|^from (\w+)\.', content, re.MULTILINE))
    stdlib = {"os", "sys", "json", "re", "hmac", "hashlib", "base64", "copy", "shutil", "subprocess", "time", "datetime", "math", "random", "uuid", "tempfile", "pathlib", "typing", "dataclasses", "collections", "itertools", "functools"}
    for imp in imports:
        name = imp[0] or imp[1]
        assert name in stdlib, f"Non-stdlib import found: {name}"