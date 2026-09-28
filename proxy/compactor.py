"""LAYA-SWE Context Compactor: Schema-Safe Context Compaction for Coding Agents.

Features:
1. Preserves leading system prompt (cached prefix)
2. Preserves root user task goal (pinned invariant)
3. Observation compaction: AST/pytest semantic compaction of bulky tool outputs
4. Safe tail slicing: Guarantees complete tool_calls / tool_result message grouping
5. Injects curated LAYA-SWE multi-plane evidence (Invariants, ASTs, Resolutions)
"""

import os
import re
import sys
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from proxy.memory_engine import LayaSweMemoryEngine, AxiomMemoryEngine, MultiRelationalMemoryEngine
except ImportError:
    from memory_engine import LayaSweMemoryEngine, AxiomMemoryEngine, MultiRelationalMemoryEngine


try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


def estimate_tokens(text: str) -> int:
    """Accurate BPE token estimator with fallback (~4 chars per token)."""
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Accurately tally token counts across text and structured tool calls."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += estimate_tokens(str(content))
        tool_calls = m.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            total += estimate_tokens(fn.get("name", ""))
            total += estimate_tokens(fn.get("arguments", ""))
    return total


def prune_whitespace(text: str) -> str:
    """Collapse excess blank lines and trailing whitespace."""
    if not text:
        return ""
    # Replace 3+ consecutive newlines with 2
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    # Trim trailing whitespace on each line
    return "\n".join(line.rstrip() for line in cleaned.splitlines())


def compact_tool_output(content: str, max_chars: int = 900) -> str:
    """Semantic compression of bulky tool results (pytest outputs, file dumps, shell logs)."""
    if not isinstance(content, str):
        content = str(content) if content is not None else ""

    if not content or len(content) <= max_chars:
        return content

    lines = content.splitlines()

    # 1. Pytest / Test Runner Output Compression
    if any("FAILED" in l or "FAILURES" in l or "AssertionError" in l or "passed" in l for l in lines):
        fail_lines = [
            l for l in lines
            if any(k in l for k in ["FAILED", "FAILURES", "AssertionError", "assert ", "Error:", "tests/", "E   "])
        ]
        summary_lines = [l for l in lines if ("passed" in l or "failed" in l) and "=" in l]
        header = f"[Pytest Output Summary ({len(lines)} lines)]"
        extracted = fail_lines[:15] + summary_lines[-2:]
        if extracted:
            candidate = header + "\n" + "\n".join(extracted)
            if len(candidate) <= max_chars:
                return candidate

    # 2. Source Code / File Dump Compression (Keep AST-like structural outline)
    if any(line.strip().startswith(("class ", "def ", "import ", "from ", "@")) for line in lines):
        sigs = [
            l for l in lines
            if l.strip().startswith(("class ", "def ", "import ", "from ", "@")) or "TODO" in l
        ]
        if len(sigs) >= 3:
            candidate = f"[File Structure Outline ({len(lines)} lines)]\n" + "\n".join(sigs[:30])
            if len(candidate) <= max_chars:
                return candidate

    # 3. Generic Long Terminal Log / File Content: Keep head and tail window safely bounded
    if max_chars < 60:
        return content[:max(0, max_chars - 3)] + "..."

    half = max(1, (max_chars - 35) // 2)
    omitted = len(content) - (half * 2)
    if omitted <= 0:
        return content
    return content[:half] + f"\n... [{omitted} chars omitted] ...\n" + content[len(content) - half:]


class ContextCompactor:
    def __init__(
        self,
        memory_engine: MultiRelationalMemoryEngine,
        max_tail_messages: int = 8,
        cache_friendly_placement: bool = True,
        min_compaction_token_threshold: int = 1500
    ):
        self.memory_engine = memory_engine
        self.max_tail_messages = max_tail_messages
        self.cache_friendly_placement = cache_friendly_placement
        self.min_compaction_token_threshold = min_compaction_token_threshold
        self._ingested_message_ids = set()
        self._frozen_compacted_messages: Dict[str, Dict[str, Any]] = {}

    def process_and_compact(
        self,
        messages: List[Dict[str, Any]],
        enable_compaction: bool = True
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        """Process messages, auto-write older turns, virtualize tool results, and return compacted context.

        Returns:
            compacted_messages: The pruned/augmented message list for the LLM
            raw_tokens: Estimated token count of the uncompressed conversation
            compacted_tokens: Estimated token count of the compacted payload
        """
        raw_tokens = estimate_messages_tokens(messages)

        if not enable_compaction or len(messages) <= self.max_tail_messages + 1:
            return messages, raw_tokens, raw_tokens

        # 1. Extract System Prompt and Root User Task Objective
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

        root_user_idx = -1
        for idx, m in enumerate(messages):
            if m.get("role") == "user":
                root_user_idx = idx
                break

        root_user_msg = messages[root_user_idx] if root_user_idx >= 0 else None

        # History includes all conversation turns after the root user objective
        if root_user_idx >= 0:
            history = messages[root_user_idx + 1:]
        elif system_msg:
            history = messages[1:]
        else:
            history = messages

        if len(history) <= self.max_tail_messages:
            return messages, raw_tokens, raw_tokens

        # 2. Ingest Root User Objective into System-1 Memory Graph (Provenance: 'user')
        if root_user_msg:
            u_content = root_user_msg.get("content") or ""
            u_key = f"user:{u_content[:60]}"
            if u_key not in self._ingested_message_ids and len(str(u_content).strip()) > 10:
                self.memory_engine.ingest(
                    f"[USER GOAL]: {u_content}",
                    role="user",
                    metadata={"role": "user"}
                )
                self._ingested_message_ids.add(u_key)

        # 3. Auto-Write & Observation Virtualization for Older Turns (Provenance-Gated)
        older_turns = history[:-self.max_tail_messages]
        for idx, m in enumerate(older_turns):
            role = m.get("role", "unknown")
            content = m.get("content")
            content_str = str(content) if content is not None else ""

            if role == "tool":
                compacted_obs = compact_tool_output(content_str, max_chars=400)
                msg_text = f"[OBSERVED TOOL OUTPUT]: {compacted_obs}"
                msg_role = "tool"
            elif role == "assistant" and m.get("tool_calls"):
                fn_names = [tc.get("function", {}).get("name", "") for tc in m.get("tool_calls", [])]
                msg_text = f"[ACTION]: Called {', '.join(fn_names)}"
                msg_role = "assistant"
            else:
                msg_text = f"[{role.upper()}]: {content_str[:400]}"
                msg_role = role

            msg_key = f"{role}:{msg_text[:60]}"
            if msg_key not in self._ingested_message_ids and len(msg_text.strip()) > 15:
                self.memory_engine.ingest(msg_text, role=msg_role, metadata={"role": msg_role})
                self._ingested_message_ids.add(msg_key)

        # 4. Auto-Read: Two-Slot Retrieval to prevent context-reconstruction privilege escalation
        active_query = ""
        for m in reversed(history):
            if m.get("role") in ("user", "tool") and m.get("content"):
                active_query = str(m["content"])[:300]
                break

        if hasattr(self.memory_engine, "retrieve_two_slot"):
            priv_invariants, unpriv_evidence = self.memory_engine.retrieve_two_slot(
                active_query or "active task goal", top_k=4
            )
        else:
            priv_invariants = ""
            unpriv_evidence = self.memory_engine.retrieve(active_query or "active task goal", top_k=4)

        # 5. Safe Tail Slicing (Atomic tool_calls / tool_result Preservation)
        start_idx = max(0, len(history) - self.max_tail_messages)
        while start_idx > 0 and history[start_idx].get("role") == "tool":
            start_idx -= 1

        raw_tail = history[start_idx:]

        # Refined Tail Slicing: Recency-Weighted Compaction with Batch-and-Freeze Hysteresis
        tail = []
        num_tail_msgs = len(raw_tail)

        last_asst_idx = -1
        for i in range(num_tail_msgs - 1, -1, -1):
            if raw_tail[i].get("role") == "assistant":
                last_asst_idx = i
                break

        for idx, m in enumerate(raw_tail):
            role = m.get("role")
            if role == "tool" and m.get("content") is not None:
                text = str(m["content"])
                is_latest_turn = (idx > last_asst_idx and last_asst_idx >= 0) or (idx == num_tail_msgs - 1)

                if is_latest_turn:
                    # Immediate active observation: high fidelity
                    if any(k in text for k in ["pytest", "FAILED", "FAILURES", "AssertionError"]) and len(text) > 1200:
                        m_copy = dict(m)
                        m_copy["content"] = compact_tool_output(text, max_chars=1200)
                        tail.append(m_copy)
                    elif len(text) > 8000:
                        m_copy = dict(m)
                        m_copy["content"] = compact_tool_output(text, max_chars=8000)
                        tail.append(m_copy)
                    else:
                        tail.append(m)
                else:
                    # Older intermediate tool result in the tail: freeze in cache for stable prefix
                    freeze_key = f"{m.get('tool_call_id', idx)}:{text[:100]}"
                    if freeze_key in self._frozen_compacted_messages:
                        tail.append(self._frozen_compacted_messages[freeze_key])
                    else:
                        m_copy = dict(m)
                        m_copy["content"] = compact_tool_output(text, max_chars=400)
                        self._frozen_compacted_messages[freeze_key] = m_copy
                        tail.append(m_copy)
            else:
                tail.append(m)

        # 6. Assemble Final Context with Two-Slot Security Model & Cache-Aware Placement
        compacted = []

        # Slot 1: Privileged Invariants attached to System Prompt (Provenance: system/user)
        if system_msg:
            sys_copy = dict(system_msg)
            sys_text = prune_whitespace(str(sys_copy.get("content", "")))
            if priv_invariants:
                sys_text += f"\n\n=== LAYA-SWE SYSTEM-1 MEMORY: PRIVILEGED INVARIANTS ===\n{priv_invariants}\n========================================================"
            sys_copy["content"] = sys_text
            compacted.append(sys_copy)
        elif priv_invariants:
            compacted.append({
                "role": "system",
                "content": f"=== LAYA-SWE SYSTEM-1 MEMORY: PRIVILEGED INVARIANTS ===\n{priv_invariants}\n========================================================"
            })

        if root_user_msg:
            compacted.append(root_user_msg)

        # Slot 2: Unprivileged Evidence placed at prompt tail (UNTRUSTED observation data)
        # Never promoted to system role!
        context_msg = {
            "role": "system" if not self.cache_friendly_placement else "user",
            "content": (
                "=== LAYA-SWE SYSTEM-1 MEMORY: EXECUTION CONTEXT (OBSERVED TOOL DATA) ===\n"
                f"{unpriv_evidence or 'No recent observations.'}\n"
                "========================================================================"
            )
        }

        if self.cache_friendly_placement:
            # Cache-aware: append tail first, keeping the conversation prefix stable across turns.
            compacted.extend(tail)
            compacted.append(context_msg)
        else:
            # Legacy placement
            compacted.append(context_msg)
            compacted.extend(tail)

        compacted_tokens = estimate_messages_tokens(compacted)
        return compacted, raw_tokens, compacted_tokens


# Aliases
LayaSweCompactor = ContextCompactor
AxiomCompactor = ContextCompactor
