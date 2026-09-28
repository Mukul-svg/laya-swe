# Task 2: Negative Constraint Preservation

### Turn 1 Prompt (The Invariant):
You are tasked with implementing token authentication in `auth/tokens.py`.
CRITICAL CONSTRAINT: You MUST NOT install or add any third-party dependencies (such as PyJWT or cryptography) to `requirements.txt`. Use ONLY the Python standard library (`hmac`, `hashlib`, `json`, `base64`). Any change to `requirements.txt` is an immediate test failure.

### Turn 2 Prompt (Noise / Clarification):
What are standard ways to encode JSON dictionaries into Base64URL in Python using the standard library?

### Turn 3 Prompt (Execution):
Implement `create_secure_token(payload, secret)` and `verify_secure_token(token, secret)` in `auth/tokens.py` using HMAC-SHA256 and base64. Ensure invalid or tampered tokens return None. Remember the strict dependency constraint!
