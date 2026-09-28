"""Cryptographic Token Manager.

Implements signed tokens using only the Python standard library
(hmac, hashlib, base64, json).
"""
import base64
import hashlib
import hmac
import json


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_bytes: bytes, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), payload_bytes, hashlib.sha256
    ).digest()


def create_secure_token(payload: dict, secret: str) -> str:
    """Create a signed token: base64(json payload) + '.' + base64(HMAC-SHA256 signature)."""
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded_payload = _b64encode(payload_bytes)
    signature = hmac.new(
        secret.encode("utf-8"), payload_bytes, hashlib.sha256
    ).digest()
    return f"{encoded_payload}.{_b64encode(signature)}"


def verify_secure_token(token: str, secret: str):
    """Verify and decode a signed token. Return the payload dict, or None if invalid."""
    if not isinstance(token, str) or "." not in token:
        return None
    try:
        payload_part, signature_part = token.split(".", 1)
        payload_bytes = _b64decode(payload_part)
        expected_signature = hmac.new(
            secret.encode("utf-8"), payload_bytes, hashlib.sha256
        ).digest()
        actual_signature = _b64decode(signature_part)
        if not hmac.compare_digest(expected_signature, actual_signature):
            return None
        return json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None